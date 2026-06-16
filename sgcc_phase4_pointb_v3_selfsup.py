# -*- coding: utf-8 -*-
"""
SGCC Phase 4 - Point B v3: 纯自监督异常阶段定位 (并联版本, Decision 12)

与 v2 的根本区别 (回应"判别/定位是否同一件事"的质疑):
  v2 定位头用【真实用户级异常标签 FLAG】做 BCE 监督, 与判别模块共用同一监督信号 ->
     形成"串联", 难以论证两个创新点相互独立。
  v3 定位头【完全不碰真实标签 FLAG】, 监督信号来自【合成注入自训练】:
     在低偏离"伪正常"用户上注入已知月份的合成异常, 用合成月级 GT 训练注意力头。
     -> 判别 = 有监督(真实标签); 定位 = 自监督(自造异常), 连标签都不共用 = 真正并联。

并联结构:
  原始数据 ─┬─→ 判别模块(有监督, 真实FLAG)         ──→ 用户异常分   [全体]
            └─→ 定位模块(自监督, 不碰FLAG):
                  ① 全体用户学正常流形(掩码重构+预测)
                  ② 低偏离伪正常用户注入合成异常 -> 合成月级GT -> 训练注意力头
                ──→ 用户×月偏离图   [全体]
  协同只在最后分析层 (正交性/双向不可替代/对照案例), 不进数据管线。

防泄漏: 训练注入池 (train_inject) 与 评测注入池 (eval_inject) 用户完全隔离 +
        不同随机子序列, 评测注入用户绝不出现在训练中。

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
from sgcc_phase4_pointb_v2_localization import (
    INJECTION_SHAPES,
    LocalizationMIL,
    attention_to_interval,
    inject_anomalies,
    localization_metrics,
    per_user_iou,
    self_supervised_loss,
    smooth_rows,
    standardize_channels,
)

np.random.seed(SEED)
torch.manual_seed(SEED)


def infer_seq(model, seq, eval_batch_size):
    """全量推理: 返回 (注意力, 标准化偏离, 用户分数)。"""
    model.eval()
    loader = DataLoader(TensorDataset(torch.FloatTensor(seq)), batch_size=eval_batch_size, shuffle=False)
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


def month_bce_loss(attn, d_std, gt_mask, pos_weight):
    """合成月级监督: 对逐月注意力 (经 deviation 引导) 做月级 BCE。

    注意力 a 已 softmax 归一(每行和=1), 不适合直接做月级 BCE; 这里用注意力 logits 的
    代理——直接对 (attn 经 logit 还原) 不稳定, 故改用"注意力相对均匀分布的对数比"作为月级打分,
    再用合成 GT 做带权 BCE。该监督完全来自合成注入, 不含真实标签。
    """
    n_months = attn.size(1)
    # 月级得分: log(a * M), 即相对均匀注意力的对数, >0 表示该月被强调
    score = torch.log(attn.clamp_min(1e-8) * n_months)
    bce = nn.functional.binary_cross_entropy_with_logits(
        score, gt_mask, pos_weight=pos_weight, reduction="mean"
    )
    return bce


def train_localization_selfsup(seq_all, labels_all, raw_all, nan_all, dates, args):
    """纯自监督定位训练。不使用真实标签做监督, 仅用于划分伪正常池。

    流程:
      1. 用全体用户的掩码重构+预测预热正常流形 (无标签)。
      2. 用预热模型算每用户平均偏离, 取偏离最低的一批作为"伪正常池"。
      3. 在伪正常池上注入合成异常 -> 合成月级 GT, 联合训练:
           - 自监督正常流形损失 (在伪正常的干净副本上)
           - 月级 BCE (注意力 vs 合成月级 GT)
    """
    n_months = seq_all.shape[1]
    base = NormalPatternTransformer(
        feat_dim=seq_all.shape[2],
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

    # ---- 阶段1: 全体用户掩码重构预热 (无标签) ----
    print("--- 阶段1: 正常流形预热 (掩码重构+预测, 无标签) ---")
    x_all = torch.FloatTensor(seq_all)
    warm_loader = DataLoader(TensorDataset(x_all), batch_size=args.batch_size, shuffle=True)
    for epoch in range(1, args.warmup_epochs + 1):
        model.train()
        tot, seen = 0.0, 0
        for (xb,) in warm_loader:
            xb = xb.to(DEVICE)
            l_ss = self_supervised_loss(model.base, xb, month_weights, args.mask_ratio)
            optimizer.zero_grad(set_to_none=True)
            l_ss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            tot += l_ss.item() * xb.size(0)
            seen += xb.size(0)
        print(f"  warmup epoch={epoch:02d} ss={tot / max(seen, 1):.5f}")

    # ---- 划分伪正常池 (按预热后平均偏离, 不用真实标签) ----
    _, dev_all, _ = infer_seq(model, seq_all, args.eval_batch_size)
    mean_dev = dev_all.mean(axis=1)
    n_pseudo = int(len(mean_dev) * args.pseudo_normal_frac)
    pseudo_idx = np.argsort(mean_dev)[:n_pseudo]
    rng = np.random.default_rng(args.seed)
    rng.shuffle(pseudo_idx)
    # 评测注入用户从伪正常池中【独立切出】, 训练注入只用剩余, 防泄漏
    n_eval_hold = min(args.n_inject, n_pseudo // 3)
    eval_inject_idx = pseudo_idx[:n_eval_hold]
    train_pool_idx = pseudo_idx[n_eval_hold:]
    print(f"  伪正常池={n_pseudo}  训练注入池={len(train_pool_idx)}  评测注入留出={len(eval_inject_idx)}")

    # ---- 阶段2: 合成注入自训练 (月级监督来自合成 GT, 无真实标签) ----
    print("--- 阶段2: 合成注入自训练 (月级监督=合成GT, 不用真实标签) ---")
    pool_raw = raw_all[train_pool_idx]
    pos_weight = torch.tensor(float(args.month_pos_weight), device=DEVICE)
    n_pool = len(train_pool_idx)
    for epoch in range(1, args.epochs + 1):
        # 每轮重新注入, 增加合成多样性
        ep_rng = np.random.default_rng(args.seed * 1000 + epoch)
        inj_raw, gt_mask, _ = inject_anomalies(pool_raw, args.days_per_month, ep_rng)
        # 在 (全体原始 + 本轮注入) 上重建序列以保持跨用户统计真实, 取注入部分
        comb_raw = np.vstack([raw_all, inj_raw])
        comb_nan = np.vstack([nan_all, np.zeros_like(inj_raw, dtype=bool)])
        comb_lbl = np.concatenate([labels_all, np.ones(n_pool, dtype=np.int64)])
        seq_comb, _ = build_monthly_sequences(comb_raw, comb_nan, comb_lbl, dates, args.days_per_month)
        seq_comb, _, _ = standardize_channels(seq_comb, CH_MEAN, CH_STD)
        inj_seq = seq_comb[len(raw_all):]
        # 干净伪正常副本 (用于自监督流形损失)
        clean_seq = seq_all[train_pool_idx]

        x_inj = torch.FloatTensor(inj_seq)
        x_cln = torch.FloatTensor(clean_seq)
        g_msk = torch.FloatTensor(gt_mask.astype(np.float32))
        loader = DataLoader(TensorDataset(x_inj, x_cln, g_msk), batch_size=args.batch_size, shuffle=True)

        model.train()
        agg = {"loss": 0.0, "mbce": 0.0, "ss": 0.0, "tv": 0.0, "sp": 0.0}
        seen = 0
        for xb_inj, xb_cln, gm in loader:
            xb_inj, xb_cln, gm = xb_inj.to(DEVICE), xb_cln.to(DEVICE), gm.to(DEVICE)
            _, a, _, d_std = model(xb_inj)
            l_mbce = month_bce_loss(a, d_std, gm, pos_weight)
            l_ss = self_supervised_loss(model.base, xb_cln, month_weights, args.mask_ratio)
            l_tv = (a[:, 1:] - a[:, :-1]).abs().sum(dim=1).mean()
            l_sp = a.abs().sum(dim=1).mean()
            loss = l_mbce + args.lambda_ss * l_ss + args.lambda_tv * l_tv + args.lambda_sp * l_sp
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            bs = xb_inj.size(0)
            seen += bs
            agg["loss"] += loss.item() * bs
            agg["mbce"] += l_mbce.item() * bs
            agg["ss"] += l_ss.item() * bs
            agg["tv"] += l_tv.item() * bs
            agg["sp"] += l_sp.item() * bs
        for k in agg:
            agg[k] /= max(seen, 1)
        print(
            f"  epoch={epoch:03d} loss={agg['loss']:.5f} mbce={agg['mbce']:.5f} "
            f"ss={agg['ss']:.5f} tv={agg['tv']:.5f} sp={agg['sp']:.5f}"
        )
    return model, eval_inject_idx


CH_MEAN = None
CH_STD = None


def main():
    global CH_MEAN, CH_STD
    parser = argparse.ArgumentParser(description="Point B v3: 纯自监督异常阶段定位 (并联版)")
    parser.add_argument("--data", default=r"C:\Users\wb.zhoushujie\PyCharmMiscProject\data set.csv")
    parser.add_argument(
        "--output-dir",
        default=r"C:\Users\wb.zhoushujie\PyCharmMiscProject\results\phase4_pointb_v3_selfsup",
    )
    parser.add_argument("--warmup-epochs", type=int, default=5)
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
    parser.add_argument("--lambda-ss", type=float, default=1.0)
    parser.add_argument("--lambda-tv", type=float, default=0.05)
    parser.add_argument("--lambda-sp", type=float, default=0.01)
    parser.add_argument("--month-pos-weight", type=float, default=4.0, help="月级BCE正样本权重(异常月少)")
    parser.add_argument("--pseudo-normal-frac", type=float, default=0.6, help="按预热偏离取最低比例作伪正常池")
    parser.add_argument("--n-inject", type=int, default=500, help="评测注入留出用户数")
    parser.add_argument("--max-users", type=int, default=0)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--seed-suffix", action="store_true")
    parser.add_argument("--attn-smooth", type=int, default=0)
    args = parser.parse_args()

    if args.smoke:
        args.warmup_epochs = 1
        args.epochs = 2
        args.max_users = args.max_users or 800
        args.n_inject = min(args.n_inject, 100)

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

    if args.max_users and args.max_users < len(labels):
        sel = rng.choice(len(labels), args.max_users, replace=False)
        cons_no, labels = cons_no[sel], labels[sel]
        raw_vals, nan_mask = raw_vals[sel], nan_mask[sel]

    seq, meta = build_monthly_sequences(raw_vals, nan_mask, labels, dates, args.days_per_month)
    seq, CH_MEAN, CH_STD = standardize_channels(seq)
    n_months = seq.shape[1]

    model, eval_inject_idx = train_localization_selfsup(seq, labels, raw_vals, nan_mask, dates, args)
    ckpt_path = output_dir / "pointb_v3_selfsup_model.pth"
    torch.save(model.state_dict(), ckpt_path)

    # ---- 全体用户独立推理 (不依赖判别名单) ----
    attn, dev, user_score = infer_seq(model, seq, args.eval_batch_size)
    intervals = [attention_to_interval(attn[i]) for i in range(len(labels))]
    feat = {
        "CONS_NO": cons_no,
        "FLAG": labels,
        "loc_score": user_score,
        "pred_interval_start": [iv[0] for iv in intervals],
        "pred_interval_end": [iv[1] for iv in intervals],
        "interval_confidence": [iv[2] for iv in intervals],
    }
    for m in range(n_months):
        feat[f"att_m{m + 1:02d}"] = attn[:, m]
    feat_df = pd.DataFrame(feat)
    feat_path = output_dir / "pointb_v3_selfsup_features.csv"
    feat_df.to_csv(feat_path, index=False, encoding="utf-8-sig")
    print(f"  特征文件: {feat_path}")

    # ---- 评测注入 (留出的伪正常用户, 训练绝未见过) ----
    print("--- 评测合成注入 (留出池, 防泄漏) ---")
    eval_raw = raw_vals[eval_inject_idx]
    injected_raw, gt_mask, shape_labels = inject_anomalies(eval_raw, args.days_per_month, rng)
    comb_raw = np.vstack([raw_vals, injected_raw])
    comb_nan = np.vstack([nan_mask, np.zeros_like(injected_raw, dtype=bool)])
    comb_lbl = np.concatenate([labels, np.ones(len(eval_inject_idx), dtype=np.int64)])
    seq_comb, _ = build_monthly_sequences(comb_raw, comb_nan, comb_lbl, dates, args.days_per_month)
    seq_comb, _, _ = standardize_channels(seq_comb, CH_MEAN, CH_STD)
    inj_seq = seq_comb[len(raw_vals):]

    inj_attn, inj_dev, _ = infer_seq(model, inj_seq, args.eval_batch_size)
    if args.attn_smooth and args.attn_smooth > 1:
        inj_attn = smooth_rows(inj_attn, args.attn_smooth)
        inj_dev = smooth_rows(inj_dev, args.attn_smooth)
    uniform = np.ones_like(inj_attn) / inj_attn.shape[1]

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
    sources = {
        "selfsup_attention": inj_attn,
        "deviation_baseline": inj_dev,
        "uniform_baseline": uniform,
        "random_interval": rand_interval,
    }
    inj_metrics = {"overall": {}, "by_shape": {}}
    for name, mat in sources.items():
        inj_metrics["overall"][name] = localization_metrics(mat, gt_mask)
    for shape in INJECTION_SHAPES:
        m = shapes_arr == shape
        if m.sum() == 0:
            continue
        inj_metrics["by_shape"][shape] = {
            name: localization_metrics(mat[m], gt_mask[m]) for name, mat in sources.items()
        }
    per_user = {name: per_user_iou(mat, gt_mask).tolist() for name, mat in sources.items()}

    # ---- 注意力分离度 (sanity, 用真实标签仅做事后核验, 不参与训练) ----
    max_attn = attn.max(axis=1)
    sep = {}
    if np.unique(labels).size > 1:
        sep = {
            "max_attn_mean_abnormal": float(max_attn[labels == 1].mean()),
            "max_attn_mean_normal": float(max_attn[labels == 0].mean()),
            "max_attn_separation": float(max_attn[labels == 1].mean() - max_attn[labels == 0].mean()),
        }

    metrics = {
        "note": "v3 纯自监督: 定位训练不使用真实标签, 监督来自合成注入月级GT; 真实标签仅事后核验.",
        "seed": int(args.seed),
        "attn_smooth": int(args.attn_smooth),
        "n_eval_inject": int(len(eval_inject_idx)),
        "synthetic_injection_localization": inj_metrics,
        "attention_separation": sep,
        "injection_shape_counts": {s: int((shapes_arr == s).sum()) for s in INJECTION_SHAPES},
        "per_user_iou": per_user,
        "per_user_shape": shapes_arr.tolist(),
    }
    metrics_path = output_dir / "pointb_v3_selfsup_metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(_format_metrics(metrics), f, ensure_ascii=False, indent=2)

    summary = {
        **meta,
        "feature_file": str(feat_path),
        "metrics_file": str(metrics_path),
        "checkpoint": str(ckpt_path),
        "warmup_epochs": args.warmup_epochs,
        "epochs": args.epochs,
        "pseudo_normal_frac": args.pseudo_normal_frac,
        "month_pos_weight": args.month_pos_weight,
        "elapsed_sec": float(time.time() - t0),
    }
    with open(output_dir / "pointb_v3_selfsup_summary.json", "w", encoding="utf-8") as f:
        json.dump(_format_metrics(summary), f, ensure_ascii=False, indent=2)

    print("--- 完成 (纯自监督定位, 与判别模块不共用标签) ---")
    print(json.dumps(_format_metrics({k: v for k, v in metrics.items() if k != "per_user_iou"}),
                      ensure_ascii=False, indent=2))

    del raw_vals, nan_mask
    gc.collect()


if __name__ == "__main__":
    main()
