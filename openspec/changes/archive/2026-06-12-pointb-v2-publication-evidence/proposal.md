## Why

Point B v2 (self-supervised normal manifold + weakly-supervised attention-MIL anomaly-stage localization) has been validated on a synthetic injection benchmark (full-data IoU 0.146 > uniform 0.119 > deviation 0.097, attention separation +0.199, sanity AUC 0.879) and archived. However, the current evidence is insufficient for SCI/Nature-level publication: it relies on a single-seed point estimate, has no qualitative localization evidence on REAL G3-positive users, has documented per-shape weaknesses (sudden_drop / slow_drift IoU near or below uniform), and lacks an explicit joint narrative showing how Point B complements Point A (G3). This change closes those evidence gaps so the method becomes publication-grade and honestly bounded.

## What Changes

- Add a real-anomaly qualitative localization study: select representative real G3-positive users, render month-attention heatmaps over their raw consumption curves, and document whether the localized months are plausible (qualitative, no fabricated labels).
- Add statistical robustness: run Point B v2 across multiple seeds (>=5), report mean +/- std and confidence intervals for IoU / point-adjusted F1 / attention separation, and run paired significance tests vs uniform and deviation baselines.
- Add weak-shape handling: either improve localization on sudden_drop / slow_drift (e.g., multi-scale or smoothing of the deviation/attention) OR, if not improved, explicitly and honestly bound the method's applicability to drop-to-zero / sustained-low morphology — measured, not assumed.
- Add an A+B joint complementarity narrative: a pipeline figure plus concrete case studies where A answers "who is anomalous" and B answers "when / what morphology", quantifying the added value of localization on top of G3.
- All metrics must be reproducibly computed; no claim that Point B beats G3 global AUC/F1.

## Capabilities

### New Capabilities
- `pointb-localization-evidence`: Real-user qualitative localization study and the A+B joint complementarity narrative (case selection, attention-vs-rawcurve visualization, pipeline/case-study artifacts).
- `pointb-statistical-validation`: Multi-seed statistical robustness and significance testing for Point B v2 localization metrics, plus measured weak-shape boundary characterization.

### Modified Capabilities
<!-- No existing spec requirements change; the archived change's specs remain the contract for the v2 algorithm itself. -->

## Impact

- Code: `sgcc_phase4_pointb_v2_localization.py` (add `--seed`/multi-seed support, real-user attention export, optional weak-shape variant flag); new lightweight analysis/visualization scripts for heatmaps, multi-seed aggregation, and the A+B case study.
- Data/artifacts: consumes existing `results/phase4_pointb_v2_localization/pointb_v2_localization_features.csv`, the v2 model `.pth`, and the Point A baseline `results/sgcc_phase3_g3_artifacts.npz`. Produces new figures/tables under `results/phase4_pointb_v2_evidence/`.
- Dependencies: matplotlib (figures) and scipy/statsmodels (paired significance tests) within the existing `.venv`; verify availability before use.
- No change to the Point A (G3) baseline or its metrics.
