# -*- coding: utf-8 -*-
"""
London Smart Meters - Point B v3 纯自监督异常阶段定位跨数据集验证

目的: 验证点B不是 SGCC 特有补丁, 而是可迁移的自监督阶段定位框架。
注意口径: London 无真实窃电标签; 本实验只说明在 London 正常用电数据上注入已知月份异常后,
          点B能恢复异常阶段, 属于跨数据集/分难度合成验证, 不等同真实窃电标签验证。

流程:
  1. 加载 London daily_dataset, 过滤覆盖率>=80%的用户, 日序列插值补全;
  2. 将日序列聚合为月度多通道序列;
  3. 用全体原始用户预热正常流形 (掩码重构+未来预测, 无标签);
  4. 选低偏离伪正常池, 切出训练注入池/评测注入池;
  5. 对 Easy/Medium/Hard 三档难度分别训练合成注入注意力头并评测;
  6. 输出 IoU / point-adjusted F1, 对比 deviation / uniform / random_interval 基线。
"""

import argparse
import gc
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from scipy import stats
from scipy.stats import rankdata
from torch.utils.data import DataLoader, TensorDataset

from sgcc_phase4_self_supervised import DEVICE, SEED, NormalPatternTransformer, _format_metrics
from sgcc_phase4_pointb_v2_localization import (
    LocalizationMIL,
    localization_metrics,
    per_user_iou,
    self_supervised_loss,
    smooth_rows,
    standardize_channels,
)
from sgcc_phase4_pointb_v3_selfsup import month_bce_loss

# paired_wilcoxon 在本文件内定义。

DIFFICULTY_CONFIGS = {
    "Easy": {
        "scale_range": (0.1, 0.5),
        "low_val_pct": 10,
        "zero_period": (2, 4),
        "shift_range": (0.4, 0.7),
        "zero_day_ratio": (0.3, 0.5),
        "decay_end": (0.1, 0.3),
    },
    "Medium": {
        "scale_range": (0.5, 0.8),
        "low_val_pct": 25,
        "zero_period": (5, 10),
        "shift_range": (0.15, 0.35),
        "zero_day_ratio": (0.15, 0.25),
        "decay_end": (0.4, 0.6),
    },
    "Hard": {
        "scale_range": (0.8, 0.95),
        "low_val_pct": 40,
        "zero_period": (10, 20),
        "shift_range": (0.05, 0.15),
        "zero_day_ratio": (0.05, 0.12),
        "decay_end": (0.7, 0.9),
    },
}
ATTACK_TYPES = ("scale", "fixed_low", "periodic_zero", "mean_shift", "random_zero", "gradual_decay")


def paired_wilcoxon(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    if len(a) == 0 or len(a) != len(b):
        return {"skipped": "no paired samples"}
    diff = a - b
    if np.all(diff == 0):
        return {"skipped": "all paired differences are zero"}
    stat, p = stats.wilcoxon(a, b, alternative="two-sided")
    med = float(np.median(diff))
    return {"statistic": float(stat), "p_value": float(p), "median_diff": med, "n_pairs": int(len(a))}


def load_london(base, max_users=0, min_coverage=0.8, seed=SEED):
    print("--- Step 1: 加载 London 数据 ---")
    root = Path(base)
    daily_csv = root / "daily_dataset.csv"
    block_dir = root / "daily_dataset" / "daily_dataset"
    if daily_csv.exists():
        daily = pd.read_csv(daily_csv, usecols=["LCLid", "day", "energy_sum"])
    else:
        files = sorted(block_dir.glob("block_*.csv"))
        chunks = [pd.read_csv(f, usecols=["LCLid", "day", "energy_sum"]) for f in files]
        daily = pd.concat(chunks, ignore_index=True)
    daily["day"] = pd.to_datetime(daily["day"])
    all_dates = pd.date_range(daily["day"].min(), daily["day"].max(), freq="D")
    total_days = len(all_dates)
    user_day_counts = daily.groupby("LCLid")["day"].nunique()
    valid_users = user_day_counts[user_day_counts >= total_days * min_coverage].index
    daily = daily[daily["LCLid"].isin(valid_users)].copy()
    if max_users and len(valid_users) > max_users:
        rng = np.random.default_rng(seed)
        keep = rng.choice(np.asarray(valid_users), size=max_users, replace=False)
        daily = daily[daily["LCLid"].isin(keep)].copy()
    pivot = daily.pivot_table(index="LCLid", columns="day", values="energy_sum", aggfunc="first")
    pivot = pivot.reindex(columns=all_dates)
    cons_no = pivot.index.astype(str).to_numpy()
    raw = pivot.values.astype(np.float32)
    del daily, pivot
    gc.collect()
    print(f"  用户数={raw.shape[0]}, 天数={raw.shape[1]}, 设备={DEVICE}")
    return cons_no, raw, all_dates


def fill_missing(raw_vals):
    raw = raw_vals.copy().astype(np.float32)
    nan_mask = np.isnan(raw)
    col_m = np.nan_to_num(np.nanmean(raw, axis=0), nan=0.0)
    for i in range(0, raw.shape[0], 5000):
        c = pd.DataFrame(raw[i:i + 5000])
        c = c.interpolate(method="linear", axis=1, limit_direction="both").fillna(pd.Series(col_m))
        raw[i:i + 5000] = c.values.astype(np.float32)
    return raw, nan_mask


def build_monthly_sequences(raw_vals, nan_mask, dates, days_per_month=30):
    print("--- Step 2: 构建 London 月度多通道序列 ---")
    X = raw_vals.astype(np.float32)
    N, T = X.shape
    n_months = T // days_per_month
    X = X[:, : n_months * days_per_month]
    nan_mask = nan_mask[:, : n_months * days_per_month]
    T = X.shape[1]
    half_nm = n_months // 2
    base_m = min(6, n_months)
    dates = pd.DatetimeIndex(dates[:T])
    day_of_week = dates.dayofweek.values
    month_of_year = dates.month.values
    weekday_mask = day_of_week < 5
    weekend_mask = ~weekday_mask

    mo_mean = np.zeros((N, n_months), dtype=np.float32)
    mo_std = np.zeros_like(mo_mean)
    mo_max = np.zeros_like(mo_mean)
    mo_zero = np.zeros_like(mo_mean)
    mo_nan = np.zeros_like(mo_mean)
    for m in range(n_months):
        s, e = m * days_per_month, (m + 1) * days_per_month
        seg = X[:, s:e]
        mo_mean[:, m] = seg.mean(1)
        mo_std[:, m] = seg.std(1)
        mo_max[:, m] = seg.max(1)
        mo_zero[:, m] = (seg == 0).mean(1)
        mo_nan[:, m] = nan_mask[:, s:e].mean(1)

    bl = mo_mean[:, :base_m].mean(1, keepdims=True) + 1e-3
    mo_vs = (mo_mean - bl) / (np.abs(bl) + 1e-3)
    mo_cum = np.cumsum(mo_mean - bl, axis=1)
    mo_pct = np.zeros_like(mo_mean)
    for m in range(n_months):
        mo_pct[:, m] = (rankdata(mo_mean[:, m]) / N).astype(np.float32)
    mo_rd = mo_pct - mo_pct.mean(1, keepdims=True)
    u_med = np.median(mo_mean, axis=1, keepdims=True) + 1e-3
    mo_sr = mo_mean / u_med
    mo_lr = np.log1p(np.maximum(mo_mean, 0)) - np.log1p(np.maximum(bl, 0))
    mo_r3m = np.zeros_like(mo_mean)
    mo_r3s = np.zeros_like(mo_mean)
    for m in range(n_months):
        ws = max(0, m - 2)
        mo_r3m[:, m] = mo_mean[:, ws:m + 1].mean(1)
        mo_r3s[:, m] = mo_mean[:, ws:m + 1].std(1) + 1e-6
    mo_lz = np.clip((mo_mean - mo_r3m) / mo_r3s, -5, 5)
    mo_gm = np.median(mo_mean, axis=0, keepdims=True) + 1e-3
    mo_gd = np.log1p(np.maximum(mo_mean, 0)) - np.log1p(np.maximum(mo_gm, 0))
    mo_d1 = np.diff(mo_mean, axis=1, prepend=mo_mean[:, :1])
    mo_d2 = np.diff(mo_d1, axis=1, prepend=mo_d1[:, :1])
    mo_cv = mo_std / (mo_mean + 1e-6)

    # 简单日历通道聚合为月级: weekday/weekend 比值
    mo_wdm = np.zeros_like(mo_mean)
    mo_wem = np.zeros_like(mo_mean)
    for m in range(n_months):
        s, e = m * days_per_month, (m + 1) * days_per_month
        wm = weekday_mask[s:e]
        wem = weekend_mask[s:e]
        if wm.sum() > 0:
            mo_wdm[:, m] = X[:, s:e][:, wm].mean(1)
        if wem.sum() > 0:
            mo_wem[:, m] = X[:, s:e][:, wem].mean(1)
    mo_wwr = mo_wdm / (mo_wem + 1e-6)

    # 轻量但覆盖主要语义的 16 通道
    seq = np.stack([
        mo_mean, mo_std, mo_max, mo_zero, mo_nan,
        mo_vs, mo_cum, mo_pct, mo_rd, mo_sr, mo_lr,
        mo_lz, mo_gd, mo_d1, mo_d2, mo_cv,
        mo_wwr,
    ], axis=2).astype(np.float32)
    print(f"  月度序列={seq.shape}")
    return seq, {"n_users": int(N), "n_months": int(n_months), "n_channels": int(seq.shape[2])}


def inject_attacks_with_gt(raw_clean, config, days_per_month, rng, start_month=None):
    """对每个用户注入一个短异常阶段, 返回 raw, gt_month_mask, attack_types。

    分类实验可把异常注入整个后半段; 定位实验不能这么做, 否则"全月份均匀注意力"会因为
    GT 很长而获得虚高 IoU。这里按阶段定位口径使用 3~6 个月短窗口, 位置在观测窗口后半段随机。
    """
    raw = raw_clean.copy().astype(np.float32)
    N, T = raw.shape
    n_months = T // days_per_month
    gt = np.zeros((N, n_months), dtype=np.int32)
    types = []
    for i in range(N):
        wlen = int(rng.integers(3, min(7, n_months) + 1))
        lo = n_months // 2 if start_month is None else int(start_month)
        hi = max(lo, n_months - wlen)
        s_month = int(rng.integers(lo, hi + 1))
        e_month = s_month + wlen
        attack_start = s_month * days_per_month
        attack_end = e_month * days_per_month
        gt[i, s_month:e_month] = 1

        atype = ATTACK_TYPES[i % len(ATTACK_TYPES)]
        seg = raw[i, attack_start:attack_end].copy()
        if atype == "scale":
            s = rng.uniform(*config["scale_range"])
            raw[i, attack_start:attack_end] = seg * s
        elif atype == "fixed_low":
            pos = seg[seg > 0]
            lv = np.percentile(pos, config["low_val_pct"]) if len(pos) else 0.1
            raw[i, attack_start:attack_end] = lv
        elif atype == "periodic_zero":
            lo_p, hi_p = config["zero_period"]
            p = int(rng.integers(lo_p, hi_p + 1))
            idx = np.arange(0, len(seg), p)
            raw[i, attack_start + idx] = 0.0
        elif atype == "mean_shift":
            shift = rng.uniform(*config["shift_range"]) * float(np.mean(seg))
            raw[i, attack_start:attack_end] = np.maximum(seg - shift, 0)
        elif atype == "random_zero":
            zr = rng.uniform(*config["zero_day_ratio"])
            k = max(1, int(len(seg) * zr))
            idx = rng.choice(len(seg), size=k, replace=False)
            raw[i, attack_start + idx] = 0.0
        else:
            de = rng.uniform(*config["decay_end"])
            decay = np.linspace(1.0, de, len(seg)).astype(np.float32)
            raw[i, attack_start:attack_end] = seg * decay
        types.append(atype)
    return raw, gt, np.array(types)


def infer_seq(model, seq, eval_batch_size):
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
    return np.concatenate(attn_all, 0), np.concatenate(dev_all, 0), np.concatenate(score_all, 0)


def diff_offset(name):
    return {"Easy": 101, "Medium": 202, "Hard": 303}[name]


def train_for_difficulty(base_model_state, seq_clean, raw_vals, nan_mask, dates, train_idx, eval_idx,
                         ch_mean, ch_std, config, args, difficulty):
    n_months = seq_clean.shape[1]
    model = LocalizationMIL(
        NormalPatternTransformer(
            feat_dim=seq_clean.shape[2],
            d_model=args.d_model,
            nhead=args.nhead,
            num_layers=args.num_layers,
            dim_ff=args.dim_ff,
            dropout=args.dropout,
            max_len=n_months + 2,
        ),
        args.d_model,
    ).to(DEVICE)
    model.load_state_dict(base_model_state)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    month_weights = make_month_weights(n_months, DEVICE)
    pos_weight = torch.tensor(float(args.month_pos_weight), device=DEVICE)

    pool_raw = raw_vals[train_idx]
    clean_seq = seq_clean[train_idx]
    rng = np.random.default_rng(args.seed + diff_offset(difficulty))
    print(f"--- {difficulty}: 合成注入自训练 ---")
    for epoch in range(1, args.epochs + 1):
        ep_rng = np.random.default_rng(args.seed * 1000 + epoch + diff_offset(difficulty))
        inj_raw, gt_mask, _ = inject_attacks_with_gt(pool_raw, config, args.days_per_month, ep_rng)
        comb_raw = np.vstack([raw_vals, inj_raw])
        comb_nan = np.vstack([nan_mask, np.zeros_like(inj_raw, dtype=bool)])
        seq_comb, _ = build_monthly_sequences(comb_raw, comb_nan, dates, args.days_per_month)
        seq_comb, _, _ = standardize_channels(seq_comb, ch_mean, ch_std)
        inj_seq = seq_comb[len(raw_vals):]
        loader = DataLoader(
            TensorDataset(torch.FloatTensor(inj_seq), torch.FloatTensor(clean_seq), torch.FloatTensor(gt_mask.astype(np.float32))),
            batch_size=args.batch_size,
            shuffle=True,
        )
        model.train()
        agg = {"loss": 0.0, "mbce": 0.0, "ss": 0.0}
        seen = 0
        for xb_inj, xb_clean, gm in loader:
            xb_inj, xb_clean, gm = xb_inj.to(DEVICE), xb_clean.to(DEVICE), gm.to(DEVICE)
            _, a, _, _ = model(xb_inj)
            l_mbce = month_bce_loss(a, None, gm, pos_weight)
            l_ss = self_supervised_loss(model.base, xb_clean, month_weights, args.mask_ratio)
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
        print(f"  {difficulty} epoch={epoch:02d} loss={agg['loss']/seen:.5f} mbce={agg['mbce']/seen:.5f} ss={agg['ss']/seen:.5f}")

    # 评测留出用户
    eval_rng = np.random.default_rng(args.seed * 777 + diff_offset(difficulty))
    eval_raw, gt_mask, attack_types = inject_attacks_with_gt(raw_vals[eval_idx], config, args.days_per_month, eval_rng)
    comb_raw = np.vstack([raw_vals, eval_raw])
    comb_nan = np.vstack([nan_mask, np.zeros_like(eval_raw, dtype=bool)])
    seq_comb, _ = build_monthly_sequences(comb_raw, comb_nan, dates, args.days_per_month)
    seq_comb, _, _ = standardize_channels(seq_comb, ch_mean, ch_std)
    inj_seq = seq_comb[len(raw_vals):]
    inj_attn, inj_dev, _ = infer_seq(model, inj_seq, args.eval_batch_size)
    if args.attn_smooth and args.attn_smooth > 1:
        inj_attn = smooth_rows(inj_attn, args.attn_smooth)
        inj_dev = smooth_rows(inj_dev, args.attn_smooth)
    uniform = np.ones_like(inj_attn) / inj_attn.shape[1]
    rand_interval = np.zeros_like(inj_attn)
    rng2 = np.random.default_rng(args.seed * 888 + diff_offset(difficulty))
    for i in range(rand_interval.shape[0]):
        L = int(gt_mask[i].sum())
        start = int(rng2.integers(0, rand_interval.shape[1] - L + 1))
        rand_interval[i, start:start + L] = 1.0
    sources = {
        "selfsup_attention": inj_attn,
        "deviation_baseline": inj_dev,
        "uniform_baseline": uniform,
        "random_interval": rand_interval,
    }
    overall = {name: localization_metrics(mat, gt_mask) for name, mat in sources.items()}
    by_type = {}
    for typ in ATTACK_TYPES:
        m = attack_types == typ
        if m.sum():
            by_type[typ] = {name: localization_metrics(mat[m], gt_mask[m]) for name, mat in sources.items()}
    per_user = {name: per_user_iou(mat, gt_mask).tolist() for name, mat in sources.items()}
    sig = {
        "selfsup_vs_deviation": paired_wilcoxon(per_user["selfsup_attention"], per_user["deviation_baseline"]),
        "selfsup_vs_uniform": paired_wilcoxon(per_user["selfsup_attention"], per_user["uniform_baseline"]),
        "selfsup_vs_random_interval": paired_wilcoxon(per_user["selfsup_attention"], per_user["random_interval"]),
    }
    return {
        "overall": overall,
        "by_attack_type": by_type,
        "attack_type_counts": {typ: int((attack_types == typ).sum()) for typ in ATTACK_TYPES},
        "paired_significance": sig,
        "per_user_iou": per_user,
    }, model.state_dict()


def make_month_weights(seq_len, device):
    return torch.ones(seq_len, device=device)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=r"C:\Users\wb.zhoushujie\PyCharmMiscProject\london_smart_meters")
    ap.add_argument("--output-dir", default=r"C:\Users\wb.zhoushujie\PyCharmMiscProject\results\london_pointb_v3_localization")
    ap.add_argument("--max-users", type=int, default=3000)
    ap.add_argument("--n-eval", type=int, default=500)
    ap.add_argument("--pseudo-normal-frac", type=float, default=0.65)
    ap.add_argument("--warmup-epochs", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--eval-batch-size", type=int, default=1024)
    ap.add_argument("--d-model", type=int, default=64)
    ap.add_argument("--nhead", type=int, default=4)
    ap.add_argument("--num-layers", type=int, default=2)
    ap.add_argument("--dim-ff", type=int, default=128)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--mask-ratio", type=float, default=0.25)
    ap.add_argument("--days-per-month", type=int, default=30)
    ap.add_argument("--lambda-ss", type=float, default=1.0)
    ap.add_argument("--lambda-tv", type=float, default=0.05)
    ap.add_argument("--lambda-sp", type=float, default=0.01)
    ap.add_argument("--month-pos-weight", type=float, default=2.0)
    ap.add_argument("--attn-smooth", type=int, default=0)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    if args.smoke:
        args.max_users = min(args.max_users, 800)
        args.n_eval = min(args.n_eval, 120)
        args.warmup_epochs = 1
        args.epochs = 2

    t0 = time.time()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cons_no, raw, dates = load_london(args.base, args.max_users, seed=args.seed)
    raw, nan_mask = fill_missing(raw)
    seq, meta = build_monthly_sequences(raw, nan_mask, dates, args.days_per_month)
    seq, ch_mean, ch_std = standardize_channels(seq)
    n_months = seq.shape[1]

    # 预热正常流形
    print("--- Step 3: London 正常流形预热 (无标签) ---")
    base = NormalPatternTransformer(
        feat_dim=seq.shape[2], d_model=args.d_model, nhead=args.nhead,
        num_layers=args.num_layers, dim_ff=args.dim_ff, dropout=args.dropout, max_len=n_months + 2,
    )
    model = LocalizationMIL(base, args.d_model).to(DEVICE)
    opt = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    month_weights = make_month_weights(n_months, DEVICE)
    loader = DataLoader(TensorDataset(torch.FloatTensor(seq)), batch_size=args.batch_size, shuffle=True)
    for epoch in range(1, args.warmup_epochs + 1):
        model.train()
        tot, seen = 0.0, 0
        for (xb,) in loader:
            xb = xb.to(DEVICE)
            l_ss = self_supervised_loss(model.base, xb, month_weights, args.mask_ratio)
            opt.zero_grad(set_to_none=True)
            l_ss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tot += l_ss.item() * xb.size(0)
            seen += xb.size(0)
        print(f"  warmup epoch={epoch:02d} ss={tot/max(seen,1):.5f}")

    # 伪正常池 + 留出
    _, dev, _ = infer_seq(model, seq, args.eval_batch_size)
    mean_dev = dev.mean(axis=1)
    n_pseudo = int(len(mean_dev) * args.pseudo_normal_frac)
    pseudo_idx = np.argsort(mean_dev)[:n_pseudo]
    rng = np.random.default_rng(args.seed)
    rng.shuffle(pseudo_idx)
    n_eval = min(args.n_eval, n_pseudo // 3)
    eval_idx = pseudo_idx[:n_eval]
    train_idx = pseudo_idx[n_eval:]
    print(f"  伪正常池={n_pseudo} 训练注入池={len(train_idx)} 评测留出={len(eval_idx)}")

    results = {
        "note": "London 无真实窃电标签; 本实验为分难度合成注入阶段定位验证, 不等同真实窃电定位。",
        "dataset": meta,
        "seed": int(args.seed),
        "difficulty": {},
    }
    state = model.state_dict()
    for difficulty in ("Easy", "Medium", "Hard"):
        res, _ = train_for_difficulty(
            state, seq, raw, nan_mask, dates, train_idx, eval_idx,
            ch_mean, ch_std, DIFFICULTY_CONFIGS[difficulty], args, difficulty,
        )
        results["difficulty"][difficulty] = res
        print(f"  {difficulty} IoU selfsup={res['overall']['selfsup_attention']['mean_iou']:.5f} "
              f"random={res['overall']['random_interval']['mean_iou']:.5f}")

    results["elapsed_sec"] = float(time.time() - t0)
    out = out_dir / "london_pointb_v3_localization_results.json"
    out.write_text(json.dumps(_format_metrics(results), ensure_ascii=False, indent=2), encoding="utf-8")
    print("--- 完成 London 点B v3 分难度定位实验 ---")
    print(json.dumps(_format_metrics({"difficulty": {
        k: {src: v["overall"][src]["mean_iou"] for src in ("selfsup_attention", "deviation_baseline", "random_interval")}
        for k, v in results["difficulty"].items()
    }, "out": str(out)}), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
