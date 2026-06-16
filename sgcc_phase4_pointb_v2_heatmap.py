# -*- coding: utf-8 -*-
"""
SGCC Phase4 Point B v2 - 月度注意力 × 原始月度用电曲线 可视化

对确定性选出的真实 G3 阳性用户，绘制每月注意力叠加在原始月度用电曲线上，
并阴影标注预测异常区间。纯定性/示意 —— 不存在月级真值，绝不计算 IoU/F1。
"""
import argparse
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sgcc_phase4_self_supervised import load_sgcc

DAYS_PER_MONTH = 30
N_MONTHS = 34
ATT_COLS = [f"att_m{m:02d}" for m in range(1, N_MONTHS + 1)]

DEFAULT_FEATURES = "results/phase4_pointb_v2_localization/pointb_v2_localization_features.csv"
DEFAULT_DATA = "data set.csv"
DEFAULT_OUT = "results/phase4_pointb_v2_evidence/heatmaps"


def slugify(s):
    """CONS_NO 安全文件名片段。"""
    return re.sub(r"[^0-9A-Za-z]", "_", str(s))[:40]


def select_cases(df, top, contrast):
    """确定性选案例。
    - primary: FLAG==1 中按 interval_confidence 降序 top 个，tie-break CONS_NO 升序。
    - contrast: FLAG==1 中按 user_score 升序 bottom contrast 个，tie-break CONS_NO 升序。
    返回 [(row, group)]，去重（primary 优先）。
    """
    pos = df[df["FLAG"] == 1].copy()
    pos["CONS_NO"] = pos["CONS_NO"].astype(str)

    primary = pos.sort_values(
        ["interval_confidence", "CONS_NO"], ascending=[False, True]
    ).head(top)
    contrast_df = pos.sort_values(
        ["user_score", "CONS_NO"], ascending=[True, True]
    ).head(contrast)

    cases = []
    seen = set()
    for _, r in primary.iterrows():
        cases.append((r, "primary"))
        seen.add(r["CONS_NO"])
    for _, r in contrast_df.iterrows():
        if r["CONS_NO"] in seen:
            continue
        cases.append((r, "contrast"))
        seen.add(r["CONS_NO"])
    return cases


def monthly_means(daily):
    """日序列 -> 34 个 30 天块均值，末块可短，忽略 NaN。"""
    out = np.full(N_MONTHS, np.nan, dtype=float)
    for m in range(N_MONTHS):
        s, e = m * DAYS_PER_MONTH, (m + 1) * DAYS_PER_MONTH
        block = daily[s:e]
        if block.size == 0:
            continue
        out[m] = np.nanmean(block)
    return out


def render_user(row, group, daily, out_dir):
    """绘制单用户图并返回 png 路径。"""
    cons_no = str(row["CONS_NO"])
    att = row[ATT_COLS].to_numpy(dtype=float)
    curve = monthly_means(daily)
    months = np.arange(1, N_MONTHS + 1)

    fig, ax_left = plt.subplots(figsize=(12, 5))

    # 右轴：注意力柱状
    ax_right = ax_left.twinx()
    ax_right.bar(months, att, width=0.7, color="#f4a261", alpha=0.55,
                 label="attention", zorder=1)
    ax_right.set_ylabel("monthly attention", color="#e76f51")
    ax_right.tick_params(axis="y", labelcolor="#e76f51")

    # 左轴：原始月度用电
    ax_left.plot(months, curve, color="#264653", marker="o", linewidth=1.8,
                 label="raw monthly consumption", zorder=3)
    ax_left.set_xlabel("month index (1..34)")
    ax_left.set_ylabel("raw monthly mean consumption", color="#264653")
    ax_left.tick_params(axis="y", labelcolor="#264653")
    ax_left.set_zorder(ax_right.get_zorder() + 1)
    ax_left.patch.set_visible(False)
    ax_left.set_xlim(0.5, N_MONTHS + 0.5)

    # 阴影：预测区间（0-indexed -> 绘图月索引 +1），含右端点
    p_start = int(row["pred_interval_start"])
    p_end = int(row["pred_interval_end"])
    ax_left.axvspan(p_start + 0.5, p_end + 1.5, color="#e63946", alpha=0.15,
                    label="predicted anomalous interval", zorder=0)

    title = (f"CONS_NO={cons_no} | FLAG={int(row['FLAG'])} | "
             f"user_score={float(row['user_score']):.4f} | "
             f"interval_confidence={float(row['interval_confidence']):.4f} | "
             f"group={group}")
    ax_left.set_title(title, fontsize=10)

    # 合并图例
    h1, l1 = ax_left.get_legend_handles_labels()
    h2, l2 = ax_right.get_legend_handles_labels()
    ax_left.legend(h1 + h2, l1 + l2, loc="upper right", fontsize=8)

    fig.text(0.5, 0.01,
             "Qualitative / illustrative — no month-level ground truth",
             ha="center", fontsize=9, style="italic", color="gray")

    fig.tight_layout(rect=(0, 0.03, 1, 1))
    png_path = out_dir / f"heatmap_{group}_{slugify(cons_no)}.png"
    fig.savefig(png_path, dpi=120)
    plt.close(fig)
    return png_path


def main():
    ap = argparse.ArgumentParser(description="Point B v2 注意力×用电曲线可视化")
    ap.add_argument("--features", default=DEFAULT_FEATURES)
    ap.add_argument("--data", default=DEFAULT_DATA)
    ap.add_argument("--out-dir", default=DEFAULT_OUT)
    ap.add_argument("--top", type=int, default=6)
    ap.add_argument("--contrast", type=int, default=3)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.features)
    df["CONS_NO"] = df["CONS_NO"].astype(str)
    cases = select_cases(df, args.top, args.contrast)
    print(f"选出 {len(cases)} 个案例 (primary+contrast 去重后)")

    # 加载原始日数据并建立 CONS_NO -> 行索引
    cons_no, labels, raw_vals, dates = load_sgcc(args.data)
    cons_no = np.asarray(cons_no).astype(str)
    idx_map = {c: i for i, c in enumerate(cons_no)}

    records = []
    for row, group in cases:
        c = str(row["CONS_NO"])
        if c not in idx_map:
            print(f"[WARN] CONS_NO={c} 不在数据中，跳过")
            continue
        daily = raw_vals[idx_map[c]].astype(float)
        png_path = render_user(row, group, daily, out_dir)
        print(f"  已渲染 {group}: {c} -> {png_path}")
        records.append({
            "CONS_NO": c,
            "group": group,
            "user_score": float(row["user_score"]),
            "interval_confidence": float(row["interval_confidence"]),
            "pred_interval_start": int(row["pred_interval_start"]),
            "pred_interval_end": int(row["pred_interval_end"]),
            "png_path": str(png_path),
        })

    index_path = out_dir / "heatmap_index.csv"
    pd.DataFrame(records).to_csv(index_path, index=False, encoding="utf-8")
    print(f"索引写入 {index_path}，共 {len(records)} 条")


if __name__ == "__main__":
    main()
