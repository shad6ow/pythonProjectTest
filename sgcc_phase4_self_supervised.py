# -*- coding: utf-8 -*-
"""
SGCC Phase 4: 自监督正常用电模式建模 + 异常偏离特征导出

点B目标:
1. 复用点A月度多通道序列构造
2. 仅用正常用户/低风险用户训练正常模式模型
3. 通过掩码重构、未来窗口预测学习正常用电规律
4. 输出重构误差、预测误差、隐空间正常原型距离、异常月份定位
5. 生成可与点A GBDT/RMT/Transformer特征融合的CSV特征文件
"""

import argparse
import gc
import json
import os
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.stats import rankdata
from sklearn.cluster import KMeans
from sklearn.metrics import average_precision_score, f1_score, precision_recall_curve, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from torch.utils.data import DataLoader, TensorDataset


SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.set_num_threads(min(os.cpu_count() or 4, 8))
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _scale_ch(arr2d, clip=5.0):
    flat = arr2d.reshape(-1, 1)
    flat = RobustScaler().fit_transform(flat).reshape(arr2d.shape)
    return np.clip(flat, -clip, clip).astype(np.float32)


def _scale_ch_per_user(arr2d, clip=5.0):
    med = np.median(arr2d, axis=1, keepdims=True)
    q1 = np.percentile(arr2d, 25, axis=1, keepdims=True)
    q3 = np.percentile(arr2d, 75, axis=1, keepdims=True)
    iqr = np.clip(q3 - q1, 1e-6, None)
    return np.clip((arr2d - med) / iqr, -clip, clip).astype(np.float32)

def _format_metrics(obj):
    if isinstance(obj, dict):
        return {k: _format_metrics(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_format_metrics(v) for v in obj]
    if isinstance(obj, float):
        return round(obj, 5)
    return obj


def classification_metrics(labels, scores):
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    auc_raw = roc_auc_score(labels, scores)
    if auc_raw < 0.5:
        scores = -scores
        auc_raw = roc_auc_score(labels, scores)
    precision, recall, thresholds = precision_recall_curve(labels, scores)
    f1_values = 2 * precision * recall / (precision + recall + 1e-12)
    best_idx = int(np.nanargmax(f1_values))
    best_threshold = float(thresholds[max(best_idx - 1, 0)]) if len(thresholds) else float(np.median(scores))
    preds = (scores >= best_threshold).astype(np.int64)
    return {
        "auc": float(auc_raw),
        "pr_auc": float(average_precision_score(labels, scores)),
        "best_f1": float(f1_score(labels, preds)),
        "best_threshold": best_threshold,
        "positive_rate_at_threshold": float(preds.mean()),
    }

def load_sgcc(data_path):
    print("--- Step 1: 加载数据 ---")
    header = pd.read_csv(data_path, nrows=0).columns
    dtype = {c: "float32" for c in header if c not in ["CONS_NO", "FLAG"]}
    df = pd.read_csv(data_path, dtype=dtype)
    cons_no = df["CONS_NO"].values if "CONS_NO" in df.columns else np.arange(len(df))
    labels = df["FLAG"].values.astype(np.int64)
    date_cols = [c for c in df.columns if c not in ["CONS_NO", "FLAG"]]
    raw_vals = df[date_cols].values.astype(np.float32)
    del df
    gc.collect()
    dates = pd.to_datetime(date_cols, format="%m/%d/%Y")
    print(f"  用户数={raw_vals.shape[0]}, 天数={raw_vals.shape[1]}, 设备={DEVICE}")
    return cons_no, labels, raw_vals, dates


def fill_missing(raw_vals):
    print("--- Step 2: 缺失填充 ---")
    nan_mask = np.isnan(raw_vals)
    col_mean = np.nan_to_num(np.nanmean(raw_vals, axis=0), nan=0.0)
    chunk_size = 5000
    for s in range(0, raw_vals.shape[0], chunk_size):
        e = min(s + chunk_size, raw_vals.shape[0])
        part = pd.DataFrame(raw_vals[s:e])
        part = part.interpolate(method="linear", axis=1, limit_direction="both")
        part = part.fillna(pd.Series(col_mean))
        raw_vals[s:e] = part.values.astype(np.float32)
        del part
        gc.collect()
    return raw_vals, nan_mask


def build_monthly_sequences(raw_vals, nan_mask, labels, dates, days_per_month=30):
    print("--- Step 3: 构建月度多通道序列 ---")
    n_users, t_days = raw_vals.shape
    n_months = t_days // days_per_month
    x_raw = raw_vals[:, : n_months * days_per_month]
    nan_mask = nan_mask[:, : n_months * days_per_month]

    mo_mean = np.zeros((n_users, n_months), dtype=np.float32)
    mo_std = np.zeros((n_users, n_months), dtype=np.float32)
    mo_max = np.zeros((n_users, n_months), dtype=np.float32)
    mo_zero = np.zeros((n_users, n_months), dtype=np.float32)
    mo_nan = np.zeros((n_users, n_months), dtype=np.float32)

    for m in range(n_months):
        s, e = m * days_per_month, (m + 1) * days_per_month
        part = x_raw[:, s:e]
        mo_mean[:, m] = part.mean(axis=1)
        mo_std[:, m] = part.std(axis=1)
        mo_max[:, m] = part.max(axis=1)
        mo_zero[:, m] = (part == 0).mean(axis=1)
        mo_nan[:, m] = nan_mask[:, s:e].mean(axis=1)

    half = n_months // 2
    baseline_mean = mo_mean[:, :6].mean(axis=1, keepdims=True) + 1e-3
    mo_vs_base = (mo_mean - baseline_mean) / (np.abs(baseline_mean) + 1e-3)
    mo_cumdev = np.cumsum(mo_mean - baseline_mean, axis=1)
    mo_pct = np.zeros((n_users, n_months), dtype=np.float32)
    for m in range(n_months):
        mo_pct[:, m] = (rankdata(mo_mean[:, m]) / n_users).astype(np.float32)
    mo_rank_mean = mo_pct.mean(axis=1, keepdims=True)
    mo_rank_dev = mo_pct - mo_rank_mean
    rank_drop = mo_pct[:, :half].mean(axis=1) - mo_pct[:, half:].mean(axis=1)
    rank_tile = np.tile(rank_drop[:, None], (1, n_months))
    user_median = np.median(mo_mean, axis=1, keepdims=True) + 1e-3
    mo_self_ratio = mo_mean / user_median
    mo_log_ratio = np.log1p(np.maximum(mo_mean, 0)) - np.log1p(np.maximum(baseline_mean, 0))
    mo_diff1 = np.diff(mo_mean, axis=1, prepend=mo_mean[:, :1])
    mo_diff2 = np.diff(mo_diff1, axis=1, prepend=mo_diff1[:, :1])

    mo_roll3_mean = np.zeros_like(mo_mean)
    mo_roll3_std = np.zeros_like(mo_mean)
    for m in range(n_months):
        ws = max(0, m - 2)
        mo_roll3_mean[:, m] = mo_mean[:, ws : m + 1].mean(axis=1)
        mo_roll3_std[:, m] = mo_mean[:, ws : m + 1].std(axis=1) + 1e-6
    mo_local_zscore = np.clip((mo_mean - mo_roll3_mean) / mo_roll3_std, -5, 5)
    mo_global_median = np.median(mo_mean, axis=0, keepdims=True) + 1e-3
    mo_global_dev = np.log1p(np.maximum(mo_mean, 0)) - np.log1p(np.maximum(mo_global_median, 0))

    # 简化版 ISCT 局部群体偏离轨迹：按用户平均用电分层后，对月均值做层内中位数偏离。
    user_avg = x_raw.mean(axis=1)
    strata = pd.qcut(user_avg, q=8, labels=False, duplicates="drop")
    isct_dev_monthly = np.zeros((n_users, n_months), dtype=np.float32)
    for k in range(int(strata.max()) + 1):
        mask = strata == k
        if mask.sum() < 2:
            continue
        layer_median = np.median(mo_mean[mask], axis=0, keepdims=True)
        isct_dev_monthly[mask] = ((mo_mean[mask] - layer_median) / (np.abs(layer_median) + 1e-3)).astype(np.float32)

    seq = np.stack(
        [
            _scale_ch_per_user(mo_mean),
            _scale_ch_per_user(mo_std),
            _scale_ch_per_user(mo_max),
            mo_zero.astype(np.float32),
            mo_nan.astype(np.float32),
            mo_pct.astype(np.float32),
            _scale_ch(mo_rank_dev),
            _scale_ch(rank_tile),
            _scale_ch_per_user(mo_vs_base),
            _scale_ch_per_user(mo_cumdev),
            _scale_ch(mo_log_ratio),
            _scale_ch_per_user(mo_diff1),
            _scale_ch(mo_diff2),
            mo_self_ratio.astype(np.float32),
            mo_local_zscore.astype(np.float32),
            _scale_ch(mo_global_dev),
            _scale_ch_per_user(isct_dev_monthly),
        ],
        axis=2,
    ).astype(np.float32)

    scalar_feats = seq.mean(axis=1).astype(np.float32)
    scalar_tiled = np.tile(scalar_feats[:, None, :], (1, n_months, 1))
    seq = np.concatenate([seq, scalar_tiled], axis=2).astype(np.float32)
    meta = {
        "n_users": int(n_users),
        "n_months": int(n_months),
        "feat_dim": int(seq.shape[2]),
        "positive_count": int((labels == 1).sum()),
        "negative_count": int((labels == 0).sum()),
    }
    print(f"  月度序列={seq.shape}")
    return seq, meta


class NormalPatternTransformer(nn.Module):
    def __init__(self, feat_dim, d_model=64, nhead=4, num_layers=2, dim_ff=128, dropout=0.1, max_len=40):
        super().__init__()
        self.input_proj = nn.Linear(feat_dim, d_model)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, d_model))
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div[: d_model // 2])
        self.register_buffer("pe", pe.unsqueeze(0))
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_ff,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers, enable_nested_tensor=False)
        self.recon_head = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, feat_dim))
        self.pred_head = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, feat_dim))
        self.latent_norm = nn.LayerNorm(d_model)

    def encode(self, x, month_mask=None):
        h = self.input_proj(x)
        if month_mask is not None:
            h = torch.where(month_mask[:, :, None], self.mask_token.expand_as(h), h)
        h = h + self.pe[:, : h.size(1), :]
        h = self.encoder(h)
        return h

    def forward(self, x, month_mask=None):
        h = self.encode(x, month_mask)
        recon = self.recon_head(h)
        pooled = self.latent_norm(h.mean(dim=1))
        future_base = h[:, :-1]
        pred_next = self.pred_head(future_base)
        return recon, pred_next, pooled


def _normal_reference_percentile(values, normal_idx):
    values = np.asarray(values, dtype=np.float64)
    ref = np.sort(values[normal_idx])
    pct = np.searchsorted(ref, values, side="right") / max(len(ref), 1)
    return pct.astype(np.float32)


def _normal_reference_zscore(values, normal_idx):
    values = np.asarray(values, dtype=np.float64)
    ref = values[normal_idx]
    med = np.median(ref)
    mad = np.median(np.abs(ref - med)) + 1e-6
    return ((values - med) / (1.4826 * mad)).astype(np.float32)


def _cluster_reference_dist(latent, normal_idx, n_clusters):
    normal_latent = latent[normal_idx]
    k = min(max(int(n_clusters), 1), len(normal_latent))
    if k <= 1:
        center = normal_latent.mean(axis=0, keepdims=True)
        dist = np.sqrt(((latent - center) ** 2).sum(axis=1))
        return dist.astype(np.float32), np.zeros(len(latent), dtype=np.int32)
    km = KMeans(n_clusters=k, random_state=SEED, n_init=10)
    km.fit(normal_latent)
    diff = latent[:, None, :] - km.cluster_centers_[None, :, :]
    dists = np.sqrt((diff ** 2).sum(axis=2))
    nearest = dists.argmin(axis=1).astype(np.int32)
    dist = dists[np.arange(len(latent)), nearest]
    return dist.astype(np.float32), nearest


def make_month_weights(seq_len, device, late_weight=1.0, sustained_weight=1.0, late_strategy="step"):
    weights = torch.ones(seq_len, device=device)
    if late_weight > 1.0:
        late_start = seq_len * 2 // 3
        if late_strategy == "exponential":
            steps = max(seq_len - late_start, 1)
            ramp = torch.linspace(0.0, 1.0, steps, device=device)
            weights[late_start:] *= torch.pow(torch.tensor(float(late_weight), device=device), ramp)
        else:
            weights[late_start:] *= late_weight
    if sustained_weight > 1.0:
        s, e = 25, min(34, seq_len)
        if s < e:
            weights[s:e] *= sustained_weight
    return weights


def make_month_mask(batch_size, seq_len, mask_ratio, device, late_bias=0.0):
    if late_bias > 0.0:
        prob = torch.full((seq_len,), mask_ratio, device=device)
        late_start = seq_len * 2 // 3
        prob[late_start:] = torch.clamp(prob[late_start:] * (1.0 + late_bias), max=0.95)
        s, e = 25, min(34, seq_len)
        if s < e:
            prob[s:e] = torch.clamp(prob[s:e] * (1.0 + late_bias), max=0.95)
        mask = torch.rand(batch_size, seq_len, device=device) < prob[None, :]
    else:
        mask = torch.rand(batch_size, seq_len, device=device) < mask_ratio
    empty = ~mask.any(dim=1)
    if empty.any():
        idx = torch.randint(0, seq_len, (int(empty.sum()),), device=device)
        mask[empty, idx] = True
    return mask


def parse_future_horizons(raw):
    horizons = []
    for part in str(raw).split(","):
        part = part.strip()
        if part:
            horizons.append(int(part))
    return tuple(sorted(set(h for h in horizons if h > 0)))


def weighted_mse(values, target, month_weights):
    per_month = ((values - target) ** 2).mean(dim=2)
    return (per_month * month_weights[None, :]).sum() / (month_weights.sum() * values.size(0))


def reconstruction_loss(recon, xb, mask, month_weights):
    per_month = ((recon - xb) ** 2).mean(dim=2)
    weights = month_weights[None, :] * mask.float()
    return (per_month * weights).sum() / torch.clamp(weights.sum(), min=1.0)


def future_prediction_loss(model, h, xb, horizons, month_weights):
    losses = []
    for horizon in horizons:
        if horizon >= xb.size(1):
            continue
        pred = model.pred_head(h[:, :-horizon])
        losses.append(weighted_mse(pred, xb[:, horizon:, :], month_weights[horizon:]))
    if not losses:
        return torch.tensor(0.0, device=xb.device)
    return torch.stack(losses).mean()


def select_normal_training_users(labels, args):
    normal_idx = np.where(labels == 0)[0]
    if args.g3_artifact and args.clean_normal_quantile < 1.0:
        g3 = np.load(args.g3_artifact, allow_pickle=True)
        g3_labels = np.asarray(g3["labels"], dtype=np.int64)
        if len(g3_labels) != len(labels) or not np.array_equal(g3_labels, labels):
            raise ValueError("G3 artifact labels 与当前 SGCC 数据不一致，不能用于 clean-normal 选择。")
        g3_score = np.asarray(g3["oof_ensemble"], dtype=np.float64)
        normal_score = g3_score[normal_idx]
        cutoff = float(np.quantile(normal_score, args.clean_normal_quantile))
        normal_idx = normal_idx[normal_score <= cutoff]
        print(
            f"  clean-normal: 使用 label=0 且 G3 score lowest {args.clean_normal_quantile:.0%} 用户训练, "
            f"n={len(normal_idx)}, cutoff={cutoff:.6f}"
        )
    else:
        print(f"  normal training users: 使用全部 label=0 用户, n={len(normal_idx)}")
    if args.max_train_users and len(normal_idx) > args.max_train_users:
        normal_idx = np.random.choice(normal_idx, args.max_train_users, replace=False)
    return normal_idx


def train_self_supervised(seq, labels, args):
    print("--- Step 4: 训练自监督正常模式模型 ---")
    normal_idx = select_normal_training_users(labels, args)
    tr_idx, va_idx = train_test_split(normal_idx, test_size=0.15, random_state=SEED)
    x_tr = torch.FloatTensor(seq[tr_idx])
    x_va = torch.FloatTensor(seq[va_idx])
    train_loader = DataLoader(TensorDataset(x_tr), batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(TensorDataset(x_va), batch_size=args.batch_size, shuffle=False, num_workers=0)

    model = NormalPatternTransformer(
        feat_dim=seq.shape[2],
        d_model=args.d_model,
        nhead=args.nhead,
        num_layers=args.num_layers,
        dim_ff=args.dim_ff,
        dropout=args.dropout,
        max_len=seq.shape[1] + 2,
    ).to(DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    best_state, best_val = None, float("inf")
    patience_left = args.patience

    month_weights = make_month_weights(
        seq.shape[1],
        DEVICE,
        args.late_window_weight,
        args.sustained_window_weight,
        args.late_weight_strategy,
    )
    future_horizons = parse_future_horizons(args.future_horizons)

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss, seen = 0.0, 0
        for (xb,) in train_loader:
            xb = xb.to(DEVICE)
            mask = make_month_mask(xb.size(0), xb.size(1), args.mask_ratio, DEVICE, args.late_mask_bias)
            h = model.encode(xb, mask)
            recon = model.recon_head(h)
            mask_loss = reconstruction_loss(recon, xb, mask, month_weights)
            pred_loss = future_prediction_loss(model, h, xb, future_horizons, month_weights)
            loss = mask_loss + args.pred_weight * pred_loss
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item() * xb.size(0)
            seen += xb.size(0)
        train_loss /= max(seen, 1)

        model.eval()
        val_loss, seen = 0.0, 0
        with torch.no_grad():
            for (xb,) in val_loader:
                xb = xb.to(DEVICE)
                mask = make_month_mask(xb.size(0), xb.size(1), args.mask_ratio, DEVICE, args.late_mask_bias)
                h = model.encode(xb, mask)
                recon = model.recon_head(h)
                mask_loss = reconstruction_loss(recon, xb, mask, month_weights)
                pred_loss = future_prediction_loss(model, h, xb, future_horizons, month_weights)
                loss = mask_loss + args.pred_weight * pred_loss
                val_loss += loss.item() * xb.size(0)
                seen += xb.size(0)
        val_loss /= max(seen, 1)
        print(f"  epoch={epoch:03d} train_loss={train_loss:.5f} val_loss={val_loss:.5f}")
        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience_left = args.patience
        else:
            patience_left -= 1
            if patience_left <= 0:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, normal_idx, {"best_val_loss": float(best_val), "normal_train_users": int(len(normal_idx))}


def evaluate_deviation_features(model, seq, labels, cons_no, normal_idx, output_dir, args):
    print("--- Step 5: 导出异常偏离特征 ---")
    model.eval()
    x_all = torch.FloatTensor(seq)
    loader = DataLoader(TensorDataset(x_all), batch_size=args.eval_batch_size, shuffle=False, num_workers=0)
    rec_month_all, pred_month_all, latents = [], [], []
    with torch.no_grad():
        for (xb,) in loader:
            xb = xb.to(DEVICE)
            # 评估重构用固定全月份掩码：每次只遮蔽一个月份，取对角式误差。
            recon_sum = torch.zeros(xb.size(0), xb.size(1), device=DEVICE)
            for m in range(xb.size(1)):
                mask = torch.zeros(xb.size(0), xb.size(1), dtype=torch.bool, device=DEVICE)
                mask[:, m] = True
                recon, _, _ = model(xb, mask)
                recon_sum[:, m] = ((recon[:, m, :] - xb[:, m, :]) ** 2).mean(dim=1)
            _, pred_next, latent = model(xb, None)
            pred_month = torch.zeros(xb.size(0), xb.size(1), device=DEVICE)
            pred_month[:, 1:] = ((pred_next - xb[:, 1:, :]) ** 2).mean(dim=2)
            pred_month[:, 0] = pred_month[:, 1]
            rec_month_all.append(recon_sum.cpu().numpy())
            pred_month_all.append(pred_month.cpu().numpy())
            latents.append(latent.cpu().numpy())

    rec_month = np.concatenate(rec_month_all, axis=0).astype(np.float32)
    pred_month = np.concatenate(pred_month_all, axis=0).astype(np.float32)
    latent = np.concatenate(latents, axis=0).astype(np.float32)
    proto = latent[normal_idx].mean(axis=0, keepdims=True)
    latent_dist = np.sqrt(((latent - proto) ** 2).sum(axis=1)).astype(np.float32)
    cluster_latent_dist, nearest_proto = _cluster_reference_dist(latent, normal_idx, args.n_prototypes)

    rec_mean = rec_month.mean(axis=1)
    rec_max = rec_month.max(axis=1)
    pred_mean = pred_month.mean(axis=1)
    pred_max = pred_month.max(axis=1)
    rec_user_scale = rec_month / (np.median(rec_month, axis=1, keepdims=True) + 1e-6)
    pred_user_scale = pred_month / (np.median(pred_month, axis=1, keepdims=True) + 1e-6)
    rec_scale_mean = rec_user_scale.mean(axis=1).astype(np.float32)
    rec_scale_max = rec_user_scale.max(axis=1).astype(np.float32)
    pred_scale_mean = pred_user_scale.mean(axis=1).astype(np.float32)
    pred_scale_max = pred_user_scale.max(axis=1).astype(np.float32)
    late_start = seq.shape[1] * 2 // 3
    rec_late_mean = rec_month[:, late_start:].mean(axis=1)
    pred_late_mean = pred_month[:, late_start:].mean(axis=1)
    combined_month = rec_month + pred_month
    top_month = combined_month.argmax(axis=1).astype(np.int32)
    top_month_score = combined_month.max(axis=1).astype(np.float32)
    threshold = combined_month.mean(axis=1, keepdims=True) + 2.0 * combined_month.std(axis=1, keepdims=True)
    abnormal_mask = combined_month > threshold
    abnormal_month_count = abnormal_mask.sum(axis=1).astype(np.int32)
    max_consec = np.zeros(len(labels), dtype=np.int32)
    cur = np.zeros(len(labels), dtype=np.int32)
    for m in range(seq.shape[1]):
        cur = (cur + 1) * abnormal_mask[:, m]
        max_consec = np.maximum(max_consec, cur)

    feature_df = pd.DataFrame(
        {
            "CONS_NO": cons_no,
            "FLAG": labels,
            "ss_rec_mean": rec_mean,
            "ss_rec_max": rec_max,
            "ss_pred_mean": pred_mean,
            "ss_pred_max": pred_max,
            "ss_rec_scale_mean": rec_scale_mean,
            "ss_rec_scale_max": rec_scale_max,
            "ss_pred_scale_mean": pred_scale_mean,
            "ss_pred_scale_max": pred_scale_max,
            "ss_rec_ref_pct": _normal_reference_percentile(rec_mean, normal_idx),
            "ss_pred_ref_pct": _normal_reference_percentile(pred_mean, normal_idx),
            "ss_rec_ref_z": _normal_reference_zscore(rec_mean, normal_idx),
            "ss_pred_ref_z": _normal_reference_zscore(pred_mean, normal_idx),
            "ss_rec_late_mean": rec_late_mean,
            "ss_pred_late_mean": pred_late_mean,
            "ss_latent_dist": latent_dist,
            "ss_cluster_latent_dist": cluster_latent_dist,
            "ss_cluster_latent_ref_pct": _normal_reference_percentile(cluster_latent_dist, normal_idx),
            "ss_nearest_proto": nearest_proto,
            "ss_top_month": top_month,
            "ss_top_month_score": top_month_score,
            "ss_abnormal_month_count": abnormal_month_count,
            "ss_max_consec_abnormal_months": max_consec,
        }
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "sgcc_phase4_self_supervised_features.csv"
    feature_df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    np.savez_compressed(
        output_dir / "sgcc_phase4_month_errors.npz",
        rec_month=rec_month,
        pred_month=pred_month,
        latent=latent,
        labels=labels,
        cons_no=cons_no,
    )
    print(f"  特征文件: {csv_path}")

    metrics = {}
    for col in [
        "ss_rec_mean",
        "ss_rec_max",
        "ss_pred_mean",
        "ss_pred_max",
        "ss_rec_scale_mean",
        "ss_rec_scale_max",
        "ss_pred_scale_mean",
        "ss_pred_scale_max",
        "ss_rec_ref_pct",
        "ss_pred_ref_pct",
        "ss_rec_ref_z",
        "ss_pred_ref_z",
        "ss_rec_late_mean",
        "ss_pred_late_mean",
        "ss_latent_dist",
        "ss_cluster_latent_dist",
        "ss_cluster_latent_ref_pct",
        "ss_top_month_score",
        "ss_abnormal_month_count",
        "ss_max_consec_abnormal_months",
    ]:
        values = feature_df[col].values.astype(float)
        if np.isfinite(values).all() and np.unique(values).size > 1:
            metrics[col] = classification_metrics(labels, values)
            m = metrics[col]
            print(
                f"  {col}: AUC={m['auc']:.5f} PR-AUC={m['pr_auc']:.5f} "
                f"BestF1={m['best_f1']:.5f} Thr={m['best_threshold']:.5f}"
            )

    score_cols = [
        "ss_rec_ref_pct",
        "ss_pred_ref_pct",
        "ss_rec_ref_z",
        "ss_pred_ref_z",
        "ss_rec_late_mean",
        "ss_pred_late_mean",
        "ss_cluster_latent_ref_pct",
    ]
    score_mat = []
    for col in score_cols:
        v = feature_df[col].values.astype(np.float64)
        lo, hi = np.percentile(v, 1), np.percentile(v, 99)
        v = np.clip(v, lo, hi)
        v = (v - v.min()) / (v.max() - v.min() + 1e-12)
        score_mat.append(v)
    feature_df["ss_combined_score"] = np.vstack(score_mat).mean(axis=0)
    feature_df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    metrics["ss_combined_score"] = classification_metrics(labels, feature_df["ss_combined_score"].values)
    m = metrics["ss_combined_score"]
    print(
        f"  ss_combined_score: AUC={m['auc']:.5f} PR-AUC={m['pr_auc']:.5f} "
        f"BestF1={m['best_f1']:.5f} Thr={m['best_threshold']:.5f}"
    )
    return csv_path, metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=r"C:\Users\wb.zhoushujie\PyCharmMiscProject\data set.csv")
    parser.add_argument("--output-dir", default=r"C:\Users\wb.zhoushujie\PyCharmMiscProject\results\phase4_self_supervised")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--eval-batch-size", type=int, default=1024)
    parser.add_argument("--d-model", type=int, default=64)
    parser.add_argument("--nhead", type=int, default=4)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--dim-ff", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--mask-ratio", type=float, default=0.25)
    parser.add_argument("--late-mask-bias", type=float, default=0.0, help="Increase mask probability in late/month25_33 windows; 0 keeps uniform masking")
    parser.add_argument("--future-horizons", default="1", help="Comma-separated prediction horizons, e.g. 3,6 for harder future-window prediction")
    parser.add_argument("--late-window-weight", type=float, default=1.0, help="Loss weight for the last third of months")
    parser.add_argument("--late-weight-strategy", choices=["step", "exponential"], default="step", help="Late-window loss weighting schedule")
    parser.add_argument("--sustained-window-weight", type=float, default=1.0, help="Loss weight for month indices 25..33")
    parser.add_argument("--pred-weight", type=float, default=0.5)
    parser.add_argument("--n-prototypes", type=int, default=8)
    parser.add_argument("--max-train-users", type=int, default=0)
    parser.add_argument("--g3-artifact", default="", help="Phase 3 true G3 artifact for clean-normal training selection")
    parser.add_argument("--clean-normal-quantile", type=float, default=1.0, help="Use label=0 users with G3 score in lowest quantile, e.g. 0.5/0.6/0.7")
    args = parser.parse_args()

    t0 = time.time()
    output_dir = Path(args.output_dir)
    cons_no, labels, raw_vals, dates = load_sgcc(args.data)
    raw_vals, nan_mask = fill_missing(raw_vals)
    seq, meta = build_monthly_sequences(raw_vals, nan_mask, labels, dates)
    del raw_vals, nan_mask
    gc.collect()

    model, normal_idx, train_meta = train_self_supervised(seq, labels, args)
    output_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = output_dir / "sgcc_phase4_self_supervised_model.pth"
    torch.save(model.state_dict(), ckpt_path)
    csv_path, metrics = evaluate_deviation_features(model, seq, labels, cons_no, normal_idx, output_dir, args)
    summary = {
        **meta,
        **train_meta,
        "feature_file": str(csv_path),
        "checkpoint": str(ckpt_path),
        "metrics": metrics,
        "elapsed_sec": float(time.time() - t0),
    }
    with open(output_dir / "sgcc_phase4_summary.json", "w", encoding="utf-8") as f:
        json.dump(_format_metrics(summary), f, ensure_ascii=False, indent=2)
    print("--- 完成 ---")
    print(json.dumps(_format_metrics(summary), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
