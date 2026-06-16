## Context

Point B v2 is archived and validated on a synthetic injection benchmark. The trained model and a per-user export already exist:
- `results/phase4_pointb_v2_localization/pointb_v2_localization_features.csv` with columns `CONS_NO, FLAG, user_score, pred_interval_start, pred_interval_end, interval_confidence, att_m01..att_m34` (34-month attention vector per user).
- `results/phase4_pointb_v2_localization/pointb_v2_localization_model.pth` and `pointb_v2_localization_metrics.json`.
- Point A baseline at `results/sgcc_phase3_g3_artifacts.npz` (G3 user scores/labels; AUC 0.875983, F1 0.516921).

The v2 script currently fixes the global `SEED` (imported from `sgcc_phase4_self_supervised`), so all runs are identical — there is no variance estimate. `matplotlib 3.10.8` and `scipy 1.17.1` are already in `.venv`; no new dependency is required.

The gap to publication grade is evidence quality, not algorithm capability: (1) no real-user qualitative localization, (2) single-seed point estimate with no significance test, (3) measured-but-unaddressed weak shapes, (4) no explicit A+B complementarity story.

## Goals / Non-Goals

**Goals:**
- Produce real-user attention-vs-raw-curve localization visualizations for representative G3-positive users.
- Quantify localization metric variance across >=5 seeds with confidence intervals and paired significance tests vs uniform and deviation baselines.
- Either improve sudden_drop / slow_drift localization or honestly bound the method's applicability with measured numbers.
- Produce an A+B pipeline figure and concrete case studies quantifying B's added localization value on top of G3.

**Non-Goals:**
- Changing the v2 algorithm's core mechanism (manifold + attention-MIL) — that contract is fixed by the archived change.
- Any claim that Point B improves G3 global AUC/F1.
- Fabricating month-level ground-truth on real data — real-user evidence stays qualitative; quantitative GT comes only from synthetic injection.

## Decisions

**Decision 1: Reproducible multi-seed via a `--seed` CLI, aggregated externally.**
Add `--seed` to `sgcc_phase4_pointb_v2_localization.py` (default = current `SEED`), seeding numpy/torch and the injection `rng`. Run the script N>=5 times with distinct seeds; a separate lightweight aggregator reads each run's `pointb_v2_localization_metrics.json` (written to per-seed output dirs) and reports mean +/- std + 95% CI. Rationale: keeps the core script single-responsibility and deterministic per seed; avoids embedding a multi-run loop and heavy state in the training script. Alternative considered: in-script seed loop — rejected because it complicates memory management on CPU and mixes concerns.

**Decision 2: Real-user heatmaps reuse the existing export — no retraining.**
A new visualization script reads `pointb_v2_localization_features.csv` (attention + predicted interval) and reloads raw monthly consumption for the same `CONS_NO` from `data set.csv` via the existing `load_sgcc` + `build_monthly_sequences` path. It renders, per selected user, the raw monthly curve with the attention vector overlaid and the predicted interval shaded. Rationale: the heatmap is a read-only artifact; retraining adds cost and seed noise. Case selection is deterministic: top-confidence G3-positive users plus a few G3-positive-but-low-B-score contrast cases.

**Decision 3: Paired non-parametric significance test.**
Use `scipy.stats.wilcoxon` (paired, two-sided) on per-user IoU between MIL attention and each baseline (uniform, deviation), on the synthetic injection set, aggregated per seed. Report effect direction + p-value alongside means. Rationale: per-user IoU is not normally distributed; Wilcoxon is the standard paired non-parametric choice. Alternative: paired t-test — rejected (normality not assured).

**Decision 4: Weak-shape attempt is bounded and falsifiable.**
Try ONE principled variant: temporal smoothing of the attention/deviation (short moving average) to help diffuse anomalies (slow_drift) and reduce single-month over-focus (sudden_drop). Gate: the variant must raise sudden_drop AND slow_drift IoU above uniform on the same benchmark without regressing zero/sustained_low. If it fails the gate, do NOT keep tuning — record the measured boundary and state the method is validated for drop-to-zero / sustained-low morphology. Rationale: avoids the previously-rejected trial-and-error loop; one shot with a clear gate.

**Decision 5: A+B joint artifact from existing scores.**
Join G3 user scores/labels (`sgcc_phase3_g3_artifacts.npz`) with B's `user_score` + `pred_interval` + attention by `CONS_NO`. Produce: (a) a pipeline schematic (A: who -> B: when/what morphology), (b) 2-3 case studies of real G3-flagged users showing the localized months, (c) a small table summarizing how many G3-positive users receive a confident, contiguous B interval. Rationale: complementarity is the paper's core selling point; it must be shown concretely, not asserted.

## Risks / Trade-offs

- [Multi-seed changes the injection set per seed] -> Acceptable and arguably better: it measures robustness to both model init and benchmark sampling; report this explicitly.
- [Real-user heatmaps have no ground truth, risking over-interpretation] -> Mitigation: present strictly as qualitative plausibility; never compute IoU/F1 on real users; label figures as illustrative.
- [Weak-shape variant may not help] -> Mitigation: pre-committed falsifiable gate; honest boundary statement is an acceptable, publishable outcome.
- [CPU multi-seed runtime] -> Mitigation: allow reduced `--n-inject` / `--max-users` for seed-variance runs if full-data x5 is too slow, and disclose the setting; keep at least one full-data seed for the headline number.
- [CONS_NO join mismatch between G3 npz and B export] -> Mitigation: validate join coverage and report unmatched count before producing the A+B table.
