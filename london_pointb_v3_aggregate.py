# -*- coding: utf-8 -*-
"""聚合 London PointB v3 分难度定位多随机种子结果。"""
import argparse, json
from pathlib import Path
import numpy as np
from scipy import stats

DIFS = ["Easy", "Medium", "Hard"]
SRCS = ["selfsup_attention", "deviation_baseline", "uniform_baseline", "random_interval"]
TYPES = ["scale", "fixed_low", "periodic_zero", "mean_shift", "random_zero", "gradual_decay"]


def summarize(vals):
    a = np.asarray(vals, dtype=float)
    if len(a) == 0:
        return None
    mean = float(a.mean())
    std = float(a.std(ddof=1)) if len(a) >= 2 else 0.0
    ci = None
    if len(a) >= 2:
        lo, hi = stats.t.interval(0.95, len(a) - 1, loc=mean, scale=stats.sem(a))
        ci = [float(lo), float(hi)]
    return {"mean": mean, "std": std, "ci95": ci, "n": int(len(a))}


def find_files(root):
    root = Path(root)
    files = []
    single = root / "london_pointb_v3_localization_results.json"
    if single.exists():
        files.append(single)
    files.extend(sorted(root.glob("seed*/london_pointb_v3_localization_results.json")))
    return files


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-root", default=r"results/london_pointb_v3_localization_short")
    ap.add_argument("--output", default=r"results/london_pointb_v3_localization_short/london_pointb_v3_multiseed_stats.json")
    args = ap.parse_args()
    files = find_files(args.input_root)
    if not files:
        print("no files")
        return 0
    ms = [json.loads(Path(f).read_text(encoding="utf-8")) for f in files]
    out = {"note": "London 无真实窃电标签; 分难度合成注入阶段定位多种子聚合。", "n_seeds": len(files), "seed_files": [str(f) for f in files], "difficulty": {}}
    for d in DIFS:
        out["difficulty"][d] = {"overall": {}, "by_attack_type": {}}
        for src in SRCS:
            out["difficulty"][d]["overall"][src] = {
                "mean_iou": summarize([m["difficulty"][d]["overall"][src]["mean_iou"] for m in ms]),
                "pa_f1": summarize([m["difficulty"][d]["overall"][src]["pa_f1"] for m in ms]),
            }
        for typ in TYPES:
            out["difficulty"][d]["by_attack_type"][typ] = {}
            for src in SRCS:
                vals = []
                for m in ms:
                    node = m["difficulty"][d].get("by_attack_type", {}).get(typ, {}).get(src)
                    if node:
                        vals.append(node["mean_iou"])
                out["difficulty"][d]["by_attack_type"][typ][src] = {"mean_iou": summarize(vals)}
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"seeds={len(files)}")
    for d in DIFS:
        print(d)
        for src in SRCS:
            s = out["difficulty"][d]["overall"][src]["mean_iou"]
            ci = s["ci95"]
            ci_s = f"[{ci[0]:.4f},{ci[1]:.4f}]" if ci else "null"
            print(f"  {src}: IoU={s['mean']:.5f} std={s['std']:.5f} CI={ci_s}")
    print("out", args.output)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
