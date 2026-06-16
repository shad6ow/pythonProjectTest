# -*- coding: utf-8 -*-
"""
SGCC Phase 4 - Point B v2: 多随机种子定位指标聚合 + 配对显著性检验

跨多个随机种子聚合 pointb_v2_localization_metrics.json:
1. 标量指标 (mil_attention mean_iou / pa_f1, attention_separation) 的 mean/std/95%CI;
2. 把各 seed 的 per_user_iou 池化后做配对 Wilcoxon 检验
   (mil_attention vs uniform_baseline, vs deviation_baseline);
3. 逐形状 mil_attention mean_iou 的跨 seed 均值, 暴露弱形状边界。

所有数字均来自真实读取的 JSON, 不做夸大。
"""

import argparse
import json
from pathlib import Path

import numpy as np
from scipy import stats

NOTE = "no claim that Point B beats G3 global AUC/F1"
SHAPES = ["sudden_drop", "sustained_low", "zero", "slow_drift"]


def find_metric_files(root: Path):
    """优先发现 seed*/ 子目录, 否则退化为单个根 metrics 文件。"""
    files = sorted(root.glob("seed*/pointb_v2_localization_metrics.json"))
    if not files:
        single = root / "pointb_v2_localization_metrics.json"
        if single.exists():
            files = [single]
    return files


def summarize(values):
    """返回 mean/std/95%CI; n<2 时 CI=null。"""
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
    """配对 Wilcoxon; 全零差或样本不足时返回 skip 说明。"""
    a, b = np.asarray(a, float), np.asarray(b, float)
    if len(a) == 0 or len(a) != len(b):
        return {"skipped": "no paired samples"}
    diff = a - b
    if np.all(diff == 0):
        return {"skipped": "all paired differences are zero"}
    stat, p = stats.wilcoxon(a, b, alternative="two-sided")
    med = float(np.median(diff))
    direction = "mil_attention higher" if med > 0 else ("baseline higher" if med < 0 else "tie")
    return {
        "statistic": float(stat),
        "p_value": float(p),
        "median_diff": med,
        "direction": direction,
        "n_pairs": int(len(a)),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-root", default="results/phase4_pointb_v2_localization")
    ap.add_argument("--output",
                    default="results/phase4_pointb_v2_evidence/pointb_v2_multiseed_stats.json")
    args = ap.parse_args()

    root = Path(args.input_root)
    files = find_metric_files(root)
    if not files:
        print(f"[聚合] 未在 {root} 下找到任何 metrics 文件 (seed*/ 或根目录), 退出。")
        return 0
    print(f"[聚合] 发现 {len(files)} 个 seed metrics 文件。")

    metrics = [json.loads(f.read_text(encoding="utf-8")) for f in files]

    def overall(m, src, key):
        return m.get("synthetic_injection_localization", {}).get("overall", {}) \
                .get(src, {}).get(key)

    def collect(getter):
        return [v for v in (getter(m) for m in metrics) if v is not None]

    scalars = {
        "mil_attention_mean_iou": summarize(collect(lambda m: overall(m, "mil_attention", "mean_iou"))),
        "mil_attention_pa_f1": summarize(collect(lambda m: overall(m, "mil_attention", "pa_f1"))),
        "max_attn_separation": summarize(
            collect(lambda m: m.get("attention_separation", {}).get("max_attn_separation"))),
        "deviation_baseline_mean_iou": summarize(collect(lambda m: overall(m, "deviation_baseline", "mean_iou"))),
        "uniform_baseline_mean_iou": summarize(collect(lambda m: overall(m, "uniform_baseline", "mean_iou"))),
        "random_interval_mean_iou": summarize(collect(lambda m: overall(m, "random_interval", "mean_iou"))),
    }

    # 逐形状 mil_attention mean_iou 跨 seed 均值
    by_shape = {}
    for shape in SHAPES:
        vals = collect(lambda m, s=shape: m.get("synthetic_injection_localization", {})
                       .get("by_shape", {}).get(s, {}).get("mil_attention", {}).get("mean_iou"))
        by_shape[shape] = summarize(vals)

    # 池化 per_user_iou 做配对检验
    def pool(src):
        out = []
        for m in metrics:
            arr = m.get("per_user_iou", {}).get(src)
            if arr:
                out.extend(arr)
        return out

    mil = pool("mil_attention")
    uniform = pool("uniform_baseline")
    deviation = pool("deviation_baseline")
    random_interval = pool("random_interval")
    significance = {
        "mil_vs_uniform": paired_wilcoxon(mil, uniform) if mil else {"skipped": "no per_user_iou data"},
        "mil_vs_deviation": paired_wilcoxon(mil, deviation) if mil else {"skipped": "no per_user_iou data"},
        "mil_vs_random_interval": paired_wilcoxon(mil, random_interval)
        if (mil and random_interval) else {"skipped": "no random_interval per_user_iou data"},
        "pooled_n_users": len(mil),
    }

    result = {
        "note": NOTE,
        "n_seeds": len(files),
        "seed_files": [str(f) for f in files],
        "scalar_metrics": scalars,
        "mil_attention_mean_iou_by_shape": by_shape,
        "paired_significance": significance,
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")

    # 人类可读摘要
    print(f"[聚合] note: {NOTE}")
    print(f"[聚合] seeds={len(files)}, 池化 per_user n={significance['pooled_n_users']}")
    for k, s in scalars.items():
        if s:
            ci = f"CI95={s['ci95']}" if s["ci95"] else "CI95=null(n=1)"
            print(f"  {k}: mean={s['mean']:.5f} std={s['std']:.5f} {ci}")
    print("  by_shape mil_attention mean_iou:")
    for shape, s in by_shape.items():
        if s:
            print(f"    {shape}: mean={s['mean']:.5f} (n={s['n']})")
    for name in ("mil_vs_uniform", "mil_vs_deviation", "mil_vs_random_interval"):
        sig = significance[name]
        if "skipped" in sig:
            print(f"  {name}: skipped ({sig['skipped']})")
        else:
            print(f"  {name}: p={sig['p_value']:.4g} median_diff={sig['median_diff']:.5f} "
                  f"-> {sig['direction']}")
    print(f"[聚合] 已写入 {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
