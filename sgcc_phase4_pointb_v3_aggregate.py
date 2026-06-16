# -*- coding: utf-8 -*-
"""
SGCC Phase 4 - Point B v3: 纯自监督定位多随机种子聚合 + 配对显著性检验

跨多个随机种子聚合 pointb_v3_selfsup_metrics.json:
1. 标量指标 (selfsup_attention mean_iou / pa_f1, attention_separation) 的 mean/std/95%CI;
2. 池化各 seed 的 per_user_iou 做配对 Wilcoxon 检验
   (selfsup_attention vs deviation_baseline / uniform / random_interval);
3. 逐形状 selfsup_attention mean_iou 的跨 seed 均值, 含弱形状 slow_drift。

发现规则: 同时收集 seed*/ 子目录 与 根目录的 metrics 文件 (seed42 写在根目录)。
所有数字均来自真实读取的 JSON, 不做夸大。
"""

import argparse
import json
from pathlib import Path

import numpy as np
from scipy import stats

NOTE = "v3 纯自监督定位: 训练不使用真实标签, 与判别模块构成并联。"
SHAPES = ["sudden_drop", "sustained_low", "zero", "slow_drift"]
MAIN_SRC = "selfsup_attention"


def find_metric_files(root: Path):
    files = sorted(root.glob("seed*/pointb_v3_selfsup_metrics.json"))
    single = root / "pointb_v3_selfsup_metrics.json"
    if single.exists():
        files = [single] + files
    return files


def summarize(values):
    arr = np.asarray(values, dtype=float)
    n = len(arr)
    if n == 0:
        return None
    mean = float(arr.mean())
    std = float(arr.std(ddof=1)) if n >= 2 else 0.0
    if n >= 2:
        lo, hi = stats.t.interval(0.95, n - 1, loc=mean, scale=stats.sem(arr))
        ci = [float(lo), float(hi)]
    else:
        ci = None
    return {"mean": mean, "std": std, "ci95": ci, "n": n}


def paired_wilcoxon(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    if len(a) == 0 or len(a) != len(b):
        return {"skipped": "no paired samples"}
    diff = a - b
    if np.all(diff == 0):
        return {"skipped": "all paired differences are zero"}
    stat, p = stats.wilcoxon(a, b, alternative="two-sided")
    med = float(np.median(diff))
    direction = "selfsup higher" if med > 0 else ("baseline higher" if med < 0 else "tie")
    return {
        "statistic": float(stat),
        "p_value": float(p),
        "median_diff": med,
        "direction": direction,
        "n_pairs": int(len(a)),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-root", default="results/phase4_pointb_v3_selfsup")
    ap.add_argument("--output", default="results/phase4_pointb_v3_selfsup/pointb_v3_multiseed_stats.json")
    args = ap.parse_args()

    root = Path(args.input_root)
    files = find_metric_files(root)
    if not files:
        print(f"[聚合] 未在 {root} 找到 metrics 文件, 退出。")
        return 0
    print(f"[聚合] 发现 {len(files)} 个 seed metrics 文件。")
    metrics = [json.loads(f.read_text(encoding="utf-8")) for f in files]

    def overall(m, src, key):
        return m.get("synthetic_injection_localization", {}).get("overall", {}).get(src, {}).get(key)

    def collect(getter):
        return [v for v in (getter(m) for m in metrics) if v is not None]

    scalars = {
        "selfsup_mean_iou": summarize(collect(lambda m: overall(m, MAIN_SRC, "mean_iou"))),
        "selfsup_pa_f1": summarize(collect(lambda m: overall(m, MAIN_SRC, "pa_f1"))),
        "deviation_baseline_mean_iou": summarize(collect(lambda m: overall(m, "deviation_baseline", "mean_iou"))),
        "uniform_baseline_mean_iou": summarize(collect(lambda m: overall(m, "uniform_baseline", "mean_iou"))),
        "random_interval_mean_iou": summarize(collect(lambda m: overall(m, "random_interval", "mean_iou"))),
        "max_attn_separation": summarize(
            collect(lambda m: m.get("attention_separation", {}).get("max_attn_separation"))),
    }

    by_shape = {}
    for shape in SHAPES:
        vals = collect(lambda m, s=shape: m.get("synthetic_injection_localization", {})
                       .get("by_shape", {}).get(s, {}).get(MAIN_SRC, {}).get("mean_iou"))
        by_shape[shape] = summarize(vals)

    def pool(src):
        out = []
        for m in metrics:
            arr = m.get("per_user_iou", {}).get(src)
            if arr:
                out.extend(arr)
        return out

    main_iou = pool(MAIN_SRC)
    significance = {
        "selfsup_vs_deviation": paired_wilcoxon(main_iou, pool("deviation_baseline")) if main_iou else {"skipped": "no data"},
        "selfsup_vs_uniform": paired_wilcoxon(main_iou, pool("uniform_baseline")) if main_iou else {"skipped": "no data"},
        "selfsup_vs_random_interval": paired_wilcoxon(main_iou, pool("random_interval")) if main_iou else {"skipped": "no data"},
        "pooled_n_users": len(main_iou),
    }

    result = {
        "note": NOTE,
        "n_seeds": len(files),
        "seed_files": [str(f) for f in files],
        "scalar_metrics": scalars,
        "selfsup_mean_iou_by_shape": by_shape,
        "paired_significance": significance,
    }
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"[聚合] seeds={len(files)}, 池化 per_user n={significance['pooled_n_users']}")
    for k, s in scalars.items():
        if s:
            ci = f"CI95=[{s['ci95'][0]:.4f},{s['ci95'][1]:.4f}]" if s["ci95"] else "CI95=null(n=1)"
            print(f"  {k}: mean={s['mean']:.5f} std={s['std']:.5f} {ci}")
    print("  by_shape selfsup mean_iou:")
    for shape, s in by_shape.items():
        if s:
            print(f"    {shape}: mean={s['mean']:.5f} std={s['std']:.5f} (n={s['n']})")
    for name in ("selfsup_vs_deviation", "selfsup_vs_uniform", "selfsup_vs_random_interval"):
        sig = significance[name]
        if "skipped" in sig:
            print(f"  {name}: skipped ({sig['skipped']})")
        else:
            print(f"  {name}: p={sig['p_value']:.4g} median_diff={sig['median_diff']:.5f} -> {sig['direction']}")
    print(f"[聚合] 已写入 {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
