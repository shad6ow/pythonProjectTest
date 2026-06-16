# -*- coding: utf-8 -*-
"""
SGCC Phase4 — Point A (G3) × Point B v2 互补性证据。
将 A 点用户级异常分数/标签 与 B 点 v2 定位输出按 CONS_NO 关联，产出:
  (a) join 覆盖报告
  (b) G3 正样本队列中获得 "可信连续 B 区间" 的覆盖表
  (c) 3 个确定性 case study
  (d) A->B 流水线示意图
注意: 本脚本不主张 B 点能提升 G3 的全局 AUC/F1。
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sgcc_phase4_self_supervised import load_sgcc

DATA_DEFAULT = r"C:/Users/wb.zhoushujie/PyCharmMiscProject/data set.csv"
NPZ_DEFAULT = r"C:/Users/wb.zhoushujie/PyCharmMiscProject/results/sgcc_phase3_g3_artifacts.npz"
FEATURES_DEFAULT = (
    r"C:/Users/wb.zhoushujie/PyCharmMiscProject/results/"
    r"phase4_pointb_v2_localization/pointb_v2_localization_features.csv"
)
OUT_DEFAULT = r"C:/Users/wb.zhoushujie/PyCharmMiscProject/results/phase4_pointb_v2_evidence"

G3_AUC = 0.875983
G3_F1 = 0.516921
ATT_COLS = [f"att_m{m:02d}" for m in range(1, 35)]  # att_m01..att_m34


def parse_args():
    p = argparse.ArgumentParser(description="Point A × Point B v2 互补性证据")
    p.add_argument("--g3-npz", default=NPZ_DEFAULT)
    p.add_argument("--data", default=DATA_DEFAULT)
    p.add_argument("--features", default=FEATURES_DEFAULT)
    p.add_argument("--out-dir", default=OUT_DEFAULT)
    return p.parse_args()


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- 1) 加载 A 点 G3 工件 ---
    print("--- 加载 G3 npz ---")
    npz = np.load(args.g3_npz, allow_pickle=True)
    labels_npz = npz["labels"].astype(np.int64)
    g3_score = npz["oof_ensemble"].astype(np.float64)
    print(f"  npz labels={labels_npz.shape}, oof_ensemble={g3_score.shape}")

    # --- 2) 通过 load_sgcc 恢复 CONS_NO (与 npz 行同序) ---
    print("--- 通过 load_sgcc 恢复 CONS_NO ---")
    cons_no, labels_ls, _raw, _dates = load_sgcc(args.data)
    labels_ls = np.asarray(labels_ls).astype(np.int64)

    # 标签对齐校验
    if not np.array_equal(labels_ls, labels_npz):
        n_mismatch = int(np.sum(labels_ls != labels_npz)) if labels_ls.shape == labels_npz.shape else -1
        msg = (
            f"标签对齐校验失败! load_sgcc labels shape={labels_ls.shape}, "
            f"npz labels shape={labels_npz.shape}, mismatch={n_mismatch}. 终止, 不伪造 join。"
        )
        print(msg)
        raise SystemExit(msg)
    print("  标签对齐校验通过: load_sgcc 标签 == npz 标签")

    df_a = pd.DataFrame(
        {
            "CONS_NO": np.asarray(cons_no),
            "g3_score": g3_score,
            "g3_label": labels_npz,
        }
    )

    # --- 3) 与 B 点 v2 特征关联 ---
    print("--- 加载 B 点特征并关联 ---")
    df_b = pd.read_csv(args.features)
    df_a["CONS_NO"] = df_a["CONS_NO"].astype(str)
    df_b["CONS_NO"] = df_b["CONS_NO"].astype(str)

    merged = df_a.merge(df_b, on="CONS_NO", how="inner")
    matched = len(merged)
    unmatched_a = len(df_a) - df_a["CONS_NO"].isin(df_b["CONS_NO"]).sum()
    unmatched_b = len(df_b) - df_b["CONS_NO"].isin(df_a["CONS_NO"]).sum()
    print(f"  matched={matched}, unmatched_A={unmatched_a}, unmatched_B={unmatched_b}")

    join_cols = [
        "CONS_NO", "g3_score", "g3_label", "FLAG", "user_score",
        "pred_interval_start", "pred_interval_end", "interval_confidence",
    ]
    merged[join_cols].to_csv(out_dir / "ab_join.csv", index=False)

    # --- 4) 覆盖表: G3 正样本队列 ---
    pos = merged[merged["g3_label"] == 1].copy()
    n_pos = len(pos)
    conf_median = float(pos["interval_confidence"].median()) if n_pos else float("nan")
    contiguous = pos["pred_interval_end"] >= pos["pred_interval_start"]
    confident = pos["interval_confidence"] > conf_median
    confident_contig = pos[confident & contiguous]
    n_conf_contig = len(confident_contig)
    frac_conf_contig = (n_conf_contig / n_pos) if n_pos else float("nan")
    print(
        f"  G3 正样本={n_pos}, 置信中位数={conf_median:.6f}, "
        f"可信连续区间数={n_conf_contig}, 占比={frac_conf_contig:.4f}"
    )

    # 正样本 pred_interval_start 月份直方图
    fig, ax = plt.subplots(figsize=(8, 4.5))
    bins = np.arange(-0.5, 34.5, 1)
    ax.hist(pos["pred_interval_start"].dropna(), bins=bins, color="#3b7dd8", edgecolor="white")
    ax.set_xlabel("pred_interval_start (month index 0..33)")
    ax.set_ylabel("count")
    ax.set_title(f"B-interval start month for G3-positive cohort (n={n_pos})")
    ax.set_xlim(-0.5, 33.5)
    fig.tight_layout()
    fig.savefig(out_dir / "ab_interval_position_hist.png", dpi=130)
    plt.close(fig)

    # --- 5) Case studies: g3_label==1, 按 g3_score desc, CONS_NO 升序打破平局 ---
    cases_pool = pos.sort_values(
        by=["g3_score", "CONS_NO"], ascending=[False, True]
    )
    case_records = []
    for _, row in cases_pool.head(3).iterrows():
        att_vals = row[ATT_COLS].to_numpy(dtype=np.float64)
        top3_idx = np.argsort(att_vals)[::-1][:3]
        rec = {
            "CONS_NO": str(row["CONS_NO"]),
            "g3_score": float(row["g3_score"]),
            "b_user_score": float(row["user_score"]),
            "pred_interval": [int(row["pred_interval_start"]), int(row["pred_interval_end"])],
            "interval_confidence": float(row["interval_confidence"]),
            "top3_attention_months": [int(i) for i in top3_idx],  # 0-indexed month positions
            "top3_attention_values": [float(att_vals[i]) for i in top3_idx],
        }
        case_records.append(rec)
        print(
            f"  case CONS_NO={rec['CONS_NO']} g3={rec['g3_score']:.4f} "
            f"b={rec['b_user_score']:.4f} interval={rec['pred_interval']} "
            f"conf={rec['interval_confidence']:.4f} top3mon={rec['top3_attention_months']}"
        )
    with open(out_dir / "ab_case_studies.json", "w", encoding="utf-8") as f:
        json.dump(case_records, f, ensure_ascii=False, indent=2)

    # --- 6) A->B 流水线示意图 ---
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.axis("off")
    box_a = dict(boxstyle="round,pad=0.5", facecolor="#dbe9ff", edgecolor="#3b7dd8", linewidth=2)
    box_b = dict(boxstyle="round,pad=0.5", facecolor="#ffe6cc", edgecolor="#d8843b", linewidth=2)
    ax.text(
        0.22, 0.5,
        "Point A (G3 ensemble)\nWHO is anomalous\n(user-level score / flag)",
        ha="center", va="center", fontsize=11, bbox=box_a,
    )
    ax.text(
        0.78, 0.5,
        "Point B v2\n(self-supervised manifold\n+ attention-MIL)\nWHEN (which months)\n& WHAT morphology",
        ha="center", va="center", fontsize=11, bbox=box_b,
    )
    ax.annotate(
        "", xy=(0.52, 0.5), xytext=(0.40, 0.5),
        arrowprops=dict(arrowstyle="-|>", color="#444", linewidth=2.5),
    )
    ax.text(0.46, 0.58, "localize / characterize", ha="center", va="bottom", fontsize=9, color="#444")
    ax.set_title("A -> B complementarity pipeline", fontsize=13)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    fig.tight_layout()
    fig.savefig(out_dir / "ab_pipeline_schematic.png", dpi=130)
    plt.close(fig)

    # --- 7) 汇总 ---
    summary = {
        "join_coverage": {
            "df_a_rows": int(len(df_a)),
            "df_b_rows": int(len(df_b)),
            "matched": int(matched),
            "unmatched_A": int(unmatched_a),
            "unmatched_B": int(unmatched_b),
        },
        "coverage_table": {
            "g3_positive_cohort": int(n_pos),
            "interval_confidence_median_in_cohort": conf_median,
            "confident_contiguous_count": int(n_conf_contig),
            "confident_contiguous_fraction": frac_conf_contig,
            "definition": "interval_confidence > cohort median AND pred_interval_end >= pred_interval_start",
        },
        "case_study_refs": [r["CONS_NO"] for r in case_records],
        "g3_reference": {"AUC": G3_AUC, "F1": G3_F1},
        "note": (
            "Point B localizes/characterizes anomalies; it does NOT improve G3 global "
            "AUC/F1 (AUC 0.875983, F1 0.516921 unchanged)."
        ),
    }
    with open(out_dir / "ab_complementarity_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("--- 输出文件 ---")
    for name in [
        "ab_join.csv", "ab_interval_position_hist.png", "ab_case_studies.json",
        "ab_pipeline_schematic.png", "ab_complementarity_summary.json",
    ]:
        print(f"  {out_dir / name}")
    print("完成。")


if __name__ == "__main__":
    main()
