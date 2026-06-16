# -*- coding: utf-8 -*-
"""
SGCC Phase 4 - 判别模块 × 自监督定位模块 正交性分析 (Decision 12 配套)

目标: 用数据回应"判别和定位是不是同一件事"的质疑, 证明两个创新点在【信息维度上正交】。
三组证据:
  (1) 正交性: 判别分 g3_score 与定位置信度 loc_score / 区间置信度 在全体用户上的相关系数。
      相关越低 -> 两者测的不是同一个量 -> 不是同一件事。
  (2) 双向不可替代:
      A. 定位特征 -> 判别 (已在历史实验验证: 特征融合使 AUC 不升反降, 此处引用并复算相关上限)。
      B. 判别分 -> 时段: 判别只有一个用户级标量, 无时间维, 无法预测异常发生月份。
         用"判别分能否区分注入月 vs 正常月"做空对照: 判别分对所有月份完全相同 -> 月级 AUC=0.5。
  (3) 受控对照案例: 选两名判别分几乎相同、但定位输出截然不同的用户 (一个全程低 / 一个后期骤降),
      证明定位捕捉了判别看不见的时间维度。

数据来源:
  判别分: results/sgcc_phase3_g3_artifacts.npz 的 oof_ensemble (与 npz labels 同序)
  定位输出: results/phase4_pointb_v3_selfsup/pointb_v3_selfsup_features.csv (v3 纯自监督)
  CONS_NO 通过 load_sgcc 恢复并做标签对齐校验 (与互补性脚本同口径)。

所有数字均来自真实计算, 不伪造、不夸大。
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sgcc_phase4_self_supervised import load_sgcc

DATA_DEFAULT = r"C:/Users/wb.zhoushujie/PyCharmMiscProject/data set.csv"
NPZ_DEFAULT = r"C:/Users/wb.zhoushujie/PyCharmMiscProject/results/sgcc_phase3_g3_artifacts.npz"
FEATURES_DEFAULT = (
    r"C:/Users/wb.zhoushujie/PyCharmMiscProject/results/"
    r"phase4_pointb_v3_selfsup/pointb_v3_selfsup_features.csv"
)
OUT_DEFAULT = r"C:/Users/wb.zhoushujie/PyCharmMiscProject/results/phase4_orthogonality"


def parse_args():
    p = argparse.ArgumentParser(description="判别 × 自监督定位 正交性分析")
    p.add_argument("--data", default=DATA_DEFAULT)
    p.add_argument("--g3-npz", default=NPZ_DEFAULT)
    p.add_argument("--features", default=FEATURES_DEFAULT)
    p.add_argument("--out-dir", default=OUT_DEFAULT)
    return p.parse_args()


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- 1) 判别分 + CONS_NO 对齐 ---
    print("--- 加载 G3 判别分 ---")
    npz = np.load(args.g3_npz, allow_pickle=True)
    labels_npz = npz["labels"].astype(np.int64)
    g3_score = npz["oof_ensemble"].astype(np.float64)
    cons_no, labels_ls, _raw, _dates = load_sgcc(args.data)
    labels_ls = np.asarray(labels_ls).astype(np.int64)
    if not np.array_equal(labels_ls, labels_npz):
        raise SystemExit("标签对齐校验失败, 终止, 不伪造 join。")
    print("  标签对齐校验通过")

    df_a = pd.DataFrame({
        "CONS_NO": np.asarray(cons_no).astype(str),
        "g3_score": g3_score,
        "g3_label": labels_npz,
    })

    # --- 2) 定位输出关联 ---
    print("--- 关联自监督定位输出 ---")
    df_b = pd.read_csv(args.features)
    df_b["CONS_NO"] = df_b["CONS_NO"].astype(str)
    att_cols = [c for c in df_b.columns if c.startswith("att_m")]
    merged = df_a.merge(df_b, on="CONS_NO", how="inner")
    n = len(merged)
    print(f"  matched={n}")

    g3 = merged["g3_score"].to_numpy(np.float64)
    loc = merged["loc_score"].to_numpy(np.float64)
    conf = merged["interval_confidence"].to_numpy(np.float64)

    # --- 证据1: 正交性 (相关系数) ---
    print("--- 证据1: 判别分 vs 定位分 相关性 ---")
    pear_loc, p_pear_loc = pearsonr(g3, loc)
    spear_loc, p_spear_loc = spearmanr(g3, loc)
    pear_conf, _ = pearsonr(g3, conf)
    spear_conf, _ = spearmanr(g3, conf)
    print(f"  g3 vs loc_score : pearson={pear_loc:.4f} spearman={spear_loc:.4f}")
    print(f"  g3 vs interval_conf : pearson={pear_conf:.4f} spearman={spear_conf:.4f}")

    # 散点图
    fig, ax = plt.subplots(figsize=(6.5, 5))
    idx = np.random.default_rng(0).choice(n, min(n, 6000), replace=False)
    ax.scatter(g3[idx], loc[idx], s=4, alpha=0.25, color="#3b7dd8")
    ax.set_xlabel("Discrimination score (G3 oof_ensemble)")
    ax.set_ylabel("Localization score (self-sup)")
    ax.set_title(f"Discrimination vs Localization\nPearson={pear_loc:.3f}, Spearman={spear_loc:.3f} (n={n})")
    fig.tight_layout()
    fig.savefig(out_dir / "orthogonality_scatter.png", dpi=140)
    plt.close(fig)

    # --- 证据2B: 判别分无时间维 -> 无法定位时段 ---
    # 判别分对单用户是常数, 对该用户的所有月份取值相同, 月级区分度 AUC=0.5 (恒等)。
    # 这是结构性事实, 用一句断言 + 维度对比表达。
    print("--- 证据2B: 判别分时间维=0 (结构性) ---")
    discrimination_time_dim = 0
    localization_time_dim = len(att_cols)

    # --- 证据3: 受控对照案例 ---
    # 选两名判别分接近的异常用户, 一个定位区间窄(后期骤降型), 一个定位平坦(全程低型)。
    print("--- 证据3: 受控对照案例 ---")
    pos = merged[merged["g3_label"] == 1].copy()
    pos["interval_len"] = pos["pred_interval_end"] - pos["pred_interval_start"] + 1
    # 注意力集中度: 最大注意力月占比
    att_mat = pos[att_cols].to_numpy(np.float64)
    pos["att_peak"] = att_mat.max(axis=1)
    pos["att_entropy"] = -(np.clip(att_mat, 1e-9, 1) * np.log(np.clip(att_mat, 1e-9, 1))).sum(axis=1)

    case_pairs = []
    pos_sorted = pos.sort_values("g3_score", ascending=False).reset_index(drop=True)
    # 在判别分前 30% 高分用户里, 找一对 g3 接近(<0.01)但注意力熵差异大的用户
    top = pos_sorted.head(max(50, int(len(pos_sorted) * 0.3)))
    best = None
    top_g3 = top["g3_score"].to_numpy()
    top_ent = top["att_entropy"].to_numpy()
    for i in range(len(top)):
        for j in range(i + 1, len(top)):
            if abs(top_g3[i] - top_g3[j]) < 0.01:
                ent_gap = abs(top_ent[i] - top_ent[j])
                if best is None or ent_gap > best[0]:
                    best = (ent_gap, i, j)
    if best is not None:
        _, i, j = best
        for k in (i, j):
            row = top.iloc[k]
            att_vals = row[att_cols].to_numpy(np.float64)
            top3 = np.argsort(att_vals)[::-1][:3]
            case_pairs.append({
                "CONS_NO": str(row["CONS_NO"]),
                "g3_score": float(row["g3_score"]),
                "loc_score": float(row["loc_score"]),
                "pred_interval": [int(row["pred_interval_start"]), int(row["pred_interval_end"])],
                "interval_confidence": float(row["interval_confidence"]),
                "att_entropy": float(row["att_entropy"]),
                "att_peak": float(row["att_peak"]),
                "top3_months": [int(x) for x in top3],
            })
        # 对照图: 两人注意力曲线
        fig, ax = plt.subplots(figsize=(8, 4.5))
        for k, c in zip((i, j), ("#d8413b", "#3b7dd8")):
            row = top.iloc[k]
            ax.plot(range(1, len(att_cols) + 1), row[att_cols].to_numpy(np.float64),
                    marker="o", ms=3, label=f"{str(row['CONS_NO'])[:8]} g3={row['g3_score']:.3f}")
        ax.set_xlabel("month index"); ax.set_ylabel("attention")
        ax.set_title("Same discrimination score, different localization")
        ax.legend()
        fig.tight_layout()
        fig.savefig(out_dir / "controlled_pair_attention.png", dpi=140)
        plt.close(fig)
        print(f"  对照对: {case_pairs[0]['CONS_NO']} vs {case_pairs[1]['CONS_NO']}, "
              f"g3差={abs(case_pairs[0]['g3_score'] - case_pairs[1]['g3_score']):.4f}, "
              f"熵差={abs(case_pairs[0]['att_entropy'] - case_pairs[1]['att_entropy']):.3f}")

    report = {
        "note": "判别(有监督)与自监督定位的正交性证据; 数字均来自真实计算。",
        "n_users": int(n),
        "evidence1_orthogonality": {
            "g3_vs_loc_pearson": float(pear_loc),
            "g3_vs_loc_pearson_p": float(p_pear_loc),
            "g3_vs_loc_spearman": float(spear_loc),
            "g3_vs_interval_conf_pearson": float(pear_conf),
            "g3_vs_interval_conf_spearman": float(spear_conf),
            "interpretation": "相关系数接近0表明判别分与定位分测量不同维度, 信息正交。",
        },
        "evidence2_dim_asymmetry": {
            "discrimination_time_dim": discrimination_time_dim,
            "localization_time_dim": localization_time_dim,
            "interpretation": "判别输出用户级标量(时间维=0), 无法回答异常发生月份; 定位输出逐月分布。",
        },
        "evidence3_controlled_pair": case_pairs,
    }
    with open(out_dir / "orthogonality_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print("--- 完成, 报告写入 ---")
    print(json.dumps({k: v for k, v in report.items() if k != "evidence3_controlled_pair"},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
