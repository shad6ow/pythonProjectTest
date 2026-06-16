# -*- coding: utf-8 -*-
"""
SGCC Phase 4 - Point B v2: 异常"阶段定位"(stage localization) 算法升级 (Decision 11)

本脚本不追求全局检测分数, 也【不声称】能超过 G3 baseline。
成功标准 = "异常发生在哪几个月" 的定位质量 (localization quality) 相对基线的提升,
           而非全局用户级 AUC / F1。

方法概述:
1. 自监督正常流形 (self-supervised normal manifold): 复用 sgcc_phase4_self_supervised
   的 NormalPatternTransformer, 通过掩码重构 + 未来窗口预测学习正常用电规律,
   得到逐月隐表示 h_{i,m} 与逐月偏离 d_{i,m} (重构误差 + 预测误差, 按用户标准化)。
2. 弱监督注意力-MIL 定位头 (weakly-supervised attention-MIL): 在 h 上做
   linear+softmax 得到逐月注意力 a_{i,m}; 用户分数 s_i = sum_m a_{i,m} * d_{i,m},
   经可学习 scale/bias + sigmoid 做 BCE。MIL 头使用【全部】用户的弱用户级标签训练。
3. 合成异常注入基准 (synthetic injection benchmark): 取正常用户复制后在已知随机月份窗口
   注入 sudden_drop / sustained_low / zero / slow_drift 四种形状, 得到 ground-truth 月份掩码,
   用 IoU / point-adjusted F1 / precision / recall 评估定位质量, 对比三种来源:
     (a) MIL 注意力 a, (b) 原始偏离 d 的阈值 (post-hoc 基线), (c) 均匀注意力基线。

所有打印 / JSON 数字均来自真实计算, 不做任何夸大或硬编码。
"""

import argparse
import gc
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, TensorDataset

from sgcc_phase4_self_supervised import (
    DEVICE,
    SEED,
    NormalPatternTransformer,
    _format_metrics,
    build_monthly_sequences,
    classification_metrics,
    fill_missing,
    load_sgcc,
    make_month_mask,
    make_month_weights,
    reconstruction_loss,
)

np.random.seed(SEED)
torch.manual_seed(SEED)

INJECTION_SHAPES = ("sudden_drop", "sustained_low", "zero", "slow_drift")


def standardize_channels(seq, mean=None, std=None, clip=5.0):
    """逐通道 z-score + 裁剪, 防止个别未归一化通道使自监督损失量级爆炸。"""
    if mean is None:
        flat = seq.reshape(-1, seq.shape[2])
        mean = flat.mean(axis=0)
        std = flat.std(axis=0) + 1e-6
    out = np.clip((seq - mean) / std, -clip, clip).astype(np.float32)
    return out, mean, std


# ----------------------------------------------------------------------------
# 模型: 包装自监督编码器 + attention-MIL 定位头
# ----------------------------------------------------------------------------
class LocalizationMIL(nn.Module):
    """包装 NormalPatternTransformer, 暴露逐月偏离 d 与注意力定位头。"""

    def __init__(self, base: NormalPatternTransformer, d_model: int):
        super().__init__()
        self.base = base
        self.attn = nn.Linear(d_model, 1)
        nn.init.zeros_(self.attn.weight)
        nn.init.zeros_(self.attn.bias)  # 内容注意力初始为 0, 让 deviation 引导项先主导
        self.dev_gain = nn.Parameter(torch.tensor(2.0))  # deviation 引导增益, 破冷启动循环
        self.log_temp = nn.Parameter(torch.tensor(0.0))  # 可学习 softmax 温度, 允许注意力锐化
        self.score_scale = nn.Parameter(torch.tensor(1.0))
        self.score_bias = nn.Parameter(torch.tensor(0.0))

    def deviation(self, x, month_mask=None):
        """逐月偏离 d_{i,m} = 重构误差 + 预测误差, 并给出按用户标准化的 d_std。"""
        h = self.base.encode(x, month_mask)
        recon = self.base.recon_head(h)
        rec_err = ((recon - x) ** 2).mean(dim=2)  # (B, M)
        pred = self.base.pred_head(h[:, :-1])
        pred_err = torch.zeros_like(rec_err)
        pred_err[:, 1:] = ((pred - x[:, 1:, :]) ** 2).mean(dim=2)
        pred_err[:, 0] = pred_err[:, 1]
        d = rec_err + pred_err
        mu = d.mean(dim=1, keepdim=True)
        sd = d.std(dim=1, keepdim=True) + 1e-6
        d_std = (d - mu) / sd
        return h, d, d_std

    def forward(self, x):
        h, d, d_std = self.deviation(x, None)
        # deviation 引导: 内容注意力 + 偏离引导项, 初始即跟随偏离, 破冷启动
        attn_logits = self.attn(h).squeeze(-1) + self.dev_gain * d_std  # (B, M)
        temp = torch.clamp(self.log_temp.exp(), 0.05, 20.0)
        a = torch.softmax(attn_logits / temp, dim=1)
        s = (a * d_std).sum(dim=1)  # 用户级 MIL 聚合分数
        logit = self.score_scale * s + self.score_bias
        return logit, a, d, d_std


# ----------------------------------------------------------------------------
# 自监督分支损失 (masked recon + future prediction), 复用现有 helper
# ----------------------------------------------------------------------------
def self_supervised_loss(base, xb, month_weights, mask_ratio):
    mask = make_month_mask(xb.size(0), xb.size(1), mask_ratio, xb.device, 0.0)
    h = base.encode(xb, mask)
    recon = base.recon_head(h)
    rec_loss = reconstruction_loss(recon, xb, mask, month_weights)
    pred = base.pred_head(h[:, :-1])
    pred_err = ((pred - xb[:, 1:, :]) ** 2).mean(dim=2)
    pred_loss = (pred_err * month_weights[None, 1:]).sum() / (month_weights[1:].sum() * xb.size(0))
    return rec_loss + pred_loss


# ----------------------------------------------------------------------------
# 合成异常注入基准
# ----------------------------------------------------------------------------
def inject_anomalies(raw_normal, days_per_month, rng):
    """对正常用户原始日序列副本注入异常, 返回 (注入后序列, 月份 ground-truth 掩码, 形状标签)。"""
    injected = raw_normal.copy().astype(np.float32)
    n_users, t_days = injected.shape
    n_months = t_days // days_per_month
    gt_mask = np.zeros((n_users, n_months), dtype=np.int32)
    shapes = []
    for i in range(n_users):
        shape = INJECTION_SHAPES[rng.integers(len(INJECTION_SHAPES))]
        wlen = int(rng.integers(2, min(6, n_months) + 1))
        wlen = min(wlen, n_months)
        start = int(rng.integers(0, n_months - wlen + 1))
        ds = start * days_per_month
        de = (start + wlen) * days_per_month
        seg = injected[i, ds:de].copy()
        base_level = float(np.median(injected[i])) + 1e-6
        if shape == "sudden_drop":
            seg = seg * float(rng.uniform(0.05, 0.25))
        elif shape == "sustained_low":
            seg = np.full_like(seg, base_level * float(rng.uniform(0.2, 0.45)))
        elif shape == "zero":
            seg = np.zeros_like(seg)
        elif shape == "slow_drift":
            ramp = np.linspace(1.0, float(rng.uniform(0.05, 0.25)), len(seg)).astype(np.float32)
            seg = seg * ramp
        injected[i, ds:de] = seg
        gt_mask[i, start:start + wlen] = 1
        shapes.append(shape)
    return injected, gt_mask, shapes


def _point_adjust(pred, gt):
    """point-adjusted: 若某 ground-truth 连续段内有任一预测命中, 则整段记为命中。"""
    out = pred.copy()
    m = 0
    M = len(gt)
    while m < M:
        if gt[m] == 1:
            j = m
            while j < M and gt[j] == 1:
                j += 1
            if pred[m:j].any():
                out[m:j] = 1
            m = j
        else:
            m += 1
    return out


def smooth_rows(mat, k):
    """对每行做长度 k 的滑动平均 (k<=1 时原样返回)。用于弱形态平滑变体。"""
    if k is None or k <= 1:
        return mat
    kernel = np.ones(k, dtype=np.float64) / k
    out = np.empty_like(mat, dtype=np.float64)
    for i in range(mat.shape[0]):
        out[i] = np.convolve(mat[i].astype(np.float64), kernel, mode="same")
    return out


def per_user_iou(score_mat, gt_mask):
    """返回每个用户的 IoU 数组 (mean+std 阈值), 供配对显著性检验使用。"""
    ious = []
    for i in range(score_mat.shape[0]):
        s = score_mat[i].astype(np.float64)
        thr = s.mean() + s.std()
        pred = (s >= thr).astype(np.int32)
        gt = gt_mask[i].astype(np.int32)
        inter = int((pred & gt).sum())
        union = int((pred | gt).sum())
        ious.append(inter / union if union > 0 else 1.0)
    return np.asarray(ious, dtype=np.float64)


def localization_metrics(score_mat, gt_mask):
    """对每个用户用 mean+std 阈值得到预测月份掩码, 计算 IoU / point-adjusted P/R/F1。"""
    n_users = score_mat.shape[0]
    ious = []
    pa_pred_all, gt_all = [], []
    for i in range(n_users):
        s = score_mat[i].astype(np.float64)
        thr = s.mean() + s.std()
        pred = (s >= thr).astype(np.int32)
        gt = gt_mask[i].astype(np.int32)
        inter = int((pred & gt).sum())
        union = int((pred | gt).sum())
        ious.append(inter / union if union > 0 else 1.0)
        pa_pred_all.append(_point_adjust(pred, gt))
        gt_all.append(gt)
    pa_pred = np.concatenate(pa_pred_all)
    gt_flat = np.concatenate(gt_all)
    tp = int(((pa_pred == 1) & (gt_flat == 1)).sum())
    fp = int(((pa_pred == 1) & (gt_flat == 0)).sum())
    fn = int(((pa_pred == 0) & (gt_flat == 1)).sum())
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return {
        "mean_iou": float(np.mean(ious)),
        "pa_precision": float(precision),
        "pa_recall": float(recall),
        "pa_f1": float(f1),
        "n_users": int(n_users),
    }


# ----------------------------------------------------------------------------
# 训练
# ----------------------------------------------------------------------------
def train_localization(seq, labels, args):
    print("--- 训练 attention-MIL 定位模型 (全部用户, 弱用户标签) ---")
    n_months = seq.shape[1]
    base = NormalPatternTransformer(
        feat_dim=seq.shape[2],
        d_model=args.d_model,
        nhead=args.nhead,
        num_layers=args.num_layers,
        dim_ff=args.dim_ff,
        dropout=args.dropout,
        max_len=n_months + 2,
    )
    model = LocalizationMIL(base, args.d_model).to(DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    month_weights = make_month_weights(n_months, DEVICE)

    x_all = torch.FloatTensor(seq)
    y_all = torch.FloatTensor(labels.astype(np.float32))
    loader = DataLoader(TensorDataset(x_all, y_all), batch_size=args.batch_size, shuffle=True, num_workers=0)
    n_pos = float((labels == 1).sum())
    n_neg = float((labels == 0).sum())
    pos_weight = torch.tensor(n_neg / max(n_pos, 1.0), device=DEVICE)  # 类别平衡: 自然不平衡下保住 bag 分类器聚焦异常月的激励
    print(f"  正样本={int(n_pos)} 负样本={int(n_neg)} pos_weight={pos_weight.item():.3f}")
    bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    for epoch in range(1, args.epochs + 1):
        model.train()
        agg = {"loss": 0.0, "bce": 0.0, "ss": 0.0, "tv": 0.0, "sp": 0.0, "neg": 0.0}
        seen = 0
        for xb, yb in loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            logit, a, _, _ = model(xb)
            l_bce = bce(logit, yb)
            # 自监督正常流形只在正常用户上训练, 避免异常用户污染正常流形、压低异常月份偏离
            normal_rows = (yb == 0)
            if normal_rows.any():
                l_ss = self_supervised_loss(model.base, xb[normal_rows], month_weights, args.mask_ratio)
            else:
                l_ss = torch.zeros((), device=xb.device)
            l_tv = (a[:, 1:] - a[:, :-1]).abs().sum(dim=1).mean()
            l_sp = a.abs().sum(dim=1).mean()
            neg_mask = (yb == 0).float()
            l_neg = ((a ** 2).sum(dim=1) * neg_mask).sum() / torch.clamp(neg_mask.sum(), min=1.0)
            loss = (
                l_bce
                + args.lambda_ss * l_ss
                + args.lambda_tv * l_tv
                + args.lambda_sp * l_sp
                + args.lambda_neg * l_neg
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            bs = xb.size(0)
            seen += bs
            agg["loss"] += loss.item() * bs
            agg["bce"] += l_bce.item() * bs
            agg["ss"] += l_ss.item() * bs
            agg["tv"] += l_tv.item() * bs
            agg["sp"] += l_sp.item() * bs
            agg["neg"] += l_neg.item() * bs
        for k in agg:
            agg[k] /= max(seen, 1)
        print(
            f"  epoch={epoch:03d} loss={agg['loss']:.5f} bce={agg['bce']:.5f} "
            f"ss={agg['ss']:.5f} tv={agg['tv']:.5f} sp={agg['sp']:.5f} neg={agg['neg']:.5f}"
        )
    return model


# ----------------------------------------------------------------------------
# 推理: 全量用户 attention / deviation / user_score
# ----------------------------------------------------------------------------
def infer_all(model, seq, args):
    model.eval()
    loader = DataLoader(TensorDataset(torch.FloatTensor(seq)), batch_size=args.eval_batch_size, shuffle=False)
    attn_all, dev_all, score_all = [], [], []
    with torch.no_grad():
        for (xb,) in loader:
            xb = xb.to(DEVICE)
            logit, a, _, d_std = model(xb)
            attn_all.append(a.cpu().numpy())
            dev_all.append(d_std.cpu().numpy())
            score_all.append(torch.sigmoid(logit).cpu().numpy())
    return (
        np.concatenate(attn_all, axis=0).astype(np.float32),
        np.concatenate(dev_all, axis=0).astype(np.float32),
        np.concatenate(score_all, axis=0).astype(np.float32),
    )


def attention_to_interval(score_vec):
    """从逐月分数提取一个连续预测区间 [start, end] 及置信度。"""
    s = score_vec.astype(np.float64)
    thr = s.mean() + s.std()
    above = s >= thr
    if not above.any():
        peak = int(s.argmax())
        return peak, peak, float(s[peak])
    idx = np.where(above)[0]
    start, end = int(idx.min()), int(idx.max())
    return start, end, float(s[start:end + 1].mean())


# ----------------------------------------------------------------------------
# 主流程
# ----------------------------------------------------------------------------
def subsample_users(labels, max_users, rng):
    if not max_users or max_users >= len(labels):
        return np.arange(len(labels))
    pos = np.where(labels == 1)[0]
    neg = np.where(labels == 0)[0]
    n_pos = min(len(pos), max(20, max_users // 2))
    n_neg = max_users - n_pos
    n_neg = min(len(neg), n_neg)
    sel_pos = rng.choice(pos, n_pos, replace=False) if n_pos < len(pos) else pos
    sel_neg = rng.choice(neg, n_neg, replace=False) if n_neg < len(neg) else neg
    idx = np.concatenate([sel_pos, sel_neg])
    rng.shuffle(idx)
    return idx


def main():
    parser = argparse.ArgumentParser(description="Point B v2: attention-MIL 异常阶段定位 + 合成注入基准")
    parser.add_argument("--data", default=r"C:\Users\wb.zhoushujie\PyCharmMiscProject\data set.csv")
    parser.add_argument(
        "--output-dir",
        default=r"C:\Users\wb.zhoushujie\PyCharmMiscProject\results\phase4_pointb_v2_localization",
    )
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--eval-batch-size", type=int, default=1024)
    parser.add_argument("--d-model", type=int, default=64)
    parser.add_argument("--nhead", type=int, default=4)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--dim-ff", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--mask-ratio", type=float, default=0.25)
    parser.add_argument("--days-per-month", type=int, default=30)
    parser.add_argument("--lambda-ss", type=float, default=1.0, help="自监督重构+预测损失权重")
    parser.add_argument("--lambda-tv", type=float, default=0.05, help="注意力时间平滑 (total variation) 权重")
    parser.add_argument("--lambda-sp", type=float, default=0.01, help="注意力稀疏 L1 权重")
    parser.add_argument("--lambda-neg", type=float, default=0.1, help="正常用户注意力 L2 抑制权重")
    parser.add_argument("--n-inject", type=int, default=500, help="合成注入基准的正常用户数")
    parser.add_argument("--max-users", type=int, default=0, help="子采样用户数 (0=全部), 便于 smoke 测试")
    parser.add_argument("--smoke", action="store_true", help="快速冒烟: epochs=1, max-users=400, n-inject=80")
    parser.add_argument("--seed", type=int, default=SEED, help="随机种子 (numpy/torch/注入 rng), 多种子鲁棒性用")
    parser.add_argument("--seed-suffix", action="store_true", help="将输出写入 output_dir/seed<seed> 子目录, 便于多种子聚合")
    parser.add_argument("--attn-smooth", type=int, default=0, help="弱形态平滑变体: 注意力/偏离滑动平均窗口 (0/1=关闭)")
    args = parser.parse_args()

    if args.smoke:
        args.epochs = 1
        args.max_users = args.max_users or 400
        args.n_inject = min(args.n_inject, 80)

    t0 = time.time()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    output_dir = Path(args.output_dir)
    if args.seed_suffix:
        output_dir = output_dir / f"seed{args.seed}"
    output_dir.mkdir(parents=True, exist_ok=True)

    cons_no, labels, raw_vals, dates = load_sgcc(args.data)
    raw_vals, nan_mask = fill_missing(raw_vals)

    sel = subsample_users(labels, args.max_users, rng)
    cons_no, labels = cons_no[sel], labels[sel]
    raw_vals, nan_mask = raw_vals[sel], nan_mask[sel]

    seq, meta = build_monthly_sequences(raw_vals, nan_mask, labels, dates, args.days_per_month)
    seq, ch_mean, ch_std = standardize_channels(seq)
    n_months = seq.shape[1]

    model = train_localization(seq, labels, args)
    ckpt_path = output_dir / "pointb_v2_localization_model.pth"
    torch.save(model.state_dict(), ckpt_path)

    # 全量用户推理 + 导出特征
    attn, dev, user_score = infer_all(model, seq, args)
    intervals = [attention_to_interval(attn[i]) for i in range(len(labels))]
    feat = {
        "CONS_NO": cons_no,
        "FLAG": labels,
        "user_score": user_score,
        "pred_interval_start": [iv[0] for iv in intervals],
        "pred_interval_end": [iv[1] for iv in intervals],
        "interval_confidence": [iv[2] for iv in intervals],
    }
    for m in range(n_months):
        feat[f"att_m{m + 1:02d}"] = attn[:, m]
    feat_df = pd.DataFrame(feat)
    feat_path = output_dir / "pointb_v2_localization_features.csv"
    feat_df.to_csv(feat_path, index=False, encoding="utf-8-sig")
    print(f"  特征文件: {feat_path}")

    # 用户级 AUC (sanity, 非主目标)
    user_auc = None
    if np.unique(labels).size > 1:
        user_auc = classification_metrics(labels, user_score)

    # 正常 vs 异常 平均注意力分离度
    max_attn = attn.max(axis=1)
    sep = {}
    if np.unique(labels).size > 1:
        sep = {
            "max_attn_mean_abnormal": float(max_attn[labels == 1].mean()),
            "max_attn_mean_normal": float(max_attn[labels == 0].mean()),
            "max_attn_separation": float(max_attn[labels == 1].mean() - max_attn[labels == 0].mean()),
        }

    # ---- 合成注入基准 ----
    print("--- 合成异常注入基准 ---")
    normal_pos = np.where(labels == 0)[0]
    k = min(args.n_inject, len(normal_pos))
    inj_src = rng.choice(normal_pos, k, replace=False) if k < len(normal_pos) else normal_pos
    injected_raw, gt_mask, shape_labels = inject_anomalies(raw_vals[inj_src], args.days_per_month, rng)

    # 在 (原始正常 + 注入) 群体上重建序列以保持跨用户统计真实, 取注入部分
    combined_raw = np.vstack([raw_vals, injected_raw])
    combined_nan = np.vstack([nan_mask, np.zeros_like(injected_raw, dtype=bool)])
    combined_labels = np.concatenate([labels, np.ones(k, dtype=np.int64)])
    seq_comb, _ = build_monthly_sequences(combined_raw, combined_nan, combined_labels, dates, args.days_per_month)
    seq_comb, _, _ = standardize_channels(seq_comb, ch_mean, ch_std)
    inj_seq = seq_comb[len(raw_vals):]

    inj_attn, inj_dev, _ = infer_all(model, inj_seq, args)
    if args.attn_smooth and args.attn_smooth > 1:
        inj_attn = smooth_rows(inj_attn, args.attn_smooth)
        inj_dev = smooth_rows(inj_dev, args.attn_smooth)
    uniform = np.ones_like(inj_attn) / inj_attn.shape[1]

    # 公平基线: 随机连续区间, 长度=该用户真实异常月数, 位置随机 (隔离"知道在哪")
    rand_interval = np.zeros_like(inj_attn)
    n_m = rand_interval.shape[1]
    for i in range(rand_interval.shape[0]):
        L = int(gt_mask[i].sum())
        if L <= 0:
            continue
        L = min(L, n_m)
        start = int(rng.integers(0, n_m - L + 1))
        rand_interval[i, start:start + L] = 1.0

    shapes_arr = np.array(shape_labels)
    inj_metrics = {"overall": {}, "by_shape": {}}
    sources = {
        "mil_attention": inj_attn,
        "deviation_baseline": inj_dev,
        "uniform_baseline": uniform,
        "random_interval": rand_interval,
    }
    for name, mat in sources.items():
        inj_metrics["overall"][name] = localization_metrics(mat, gt_mask)
    for shape in INJECTION_SHAPES:
        mask = shapes_arr == shape
        if mask.sum() == 0:
            continue
        inj_metrics["by_shape"][shape] = {
            name: localization_metrics(mat[mask], gt_mask[mask]) for name, mat in sources.items()
        }

    # 每用户 IoU 数组 (供聚合器做配对 Wilcoxon 检验)
    per_user = {name: per_user_iou(mat, gt_mask).tolist() for name, mat in sources.items()}

    metrics = {
        "note": "success = localization quality vs baselines, NOT global AUC/F1; G3 comparison not claimed.",
        "seed": int(args.seed),
        "attn_smooth": int(args.attn_smooth),
        "synthetic_injection_localization": inj_metrics,
        "sanity_user_auc": user_auc,
        "attention_separation": sep,
        "n_injected": int(k),
        "injection_shape_counts": {s: int((shapes_arr == s).sum()) for s in INJECTION_SHAPES},
        "per_user_iou": per_user,
        "per_user_shape": shapes_arr.tolist(),
    }
    metrics_path = output_dir / "pointb_v2_localization_metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(_format_metrics(metrics), f, ensure_ascii=False, indent=2)

    summary = {
        **meta,
        "feature_file": str(feat_path),
        "metrics_file": str(metrics_path),
        "checkpoint": str(ckpt_path),
        "epochs": args.epochs,
        "lambda_ss": args.lambda_ss,
        "lambda_tv": args.lambda_tv,
        "lambda_sp": args.lambda_sp,
        "lambda_neg": args.lambda_neg,
        "elapsed_sec": float(time.time() - t0),
    }
    with open(output_dir / "pointb_v2_localization_summary.json", "w", encoding="utf-8") as f:
        json.dump(_format_metrics(summary), f, ensure_ascii=False, indent=2)

    print("--- 完成 (定位质量为主目标, 非全局检测分数) ---")
    print(json.dumps(_format_metrics(metrics), ensure_ascii=False, indent=2))

    del raw_vals, nan_mask, combined_raw, combined_nan
    gc.collect()


if __name__ == "__main__":
    main()
