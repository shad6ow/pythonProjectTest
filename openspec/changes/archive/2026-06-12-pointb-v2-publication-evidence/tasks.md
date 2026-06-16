## 1. Multi-seed support and statistical validation

- [x] 1.1 Add `--seed` CLI arg to `sgcc_phase4_pointb_v2_localization.py` that seeds numpy, torch, and the injection `rng` (default = current `SEED`); write per-run metrics JSON to a seed-suffixed output dir.
- [x] 1.2 Export per-user IoU arrays (MIL, uniform, deviation) into the metrics JSON so paired tests can be run post-hoc.
- [x] 1.3 Run the script across >=5 seeds (full-data headline seed + reduced-setting seeds if needed; disclose settings). DONE: 5 seeds (11/22/33/44/55), full data (max-users=0, n-inject=500, epochs=20). Aggregated mean+/-std+CI95: MIL IoU 0.159+/-0.039 [0.111,0.207]; pa-F1 0.336+/-0.084; attention separation 0.198+/-0.020 [0.172,0.223] (robustly positive).
- [x] 1.4 Create `sgcc_phase4_pointb_v2_aggregate.py` to read per-seed metrics JSONs and report mean +/- std + 95% CI for IoU, point-adjusted F1, attention separation.
- [x] 1.5 In the aggregator, run Wilcoxon signed-rank (paired) on per-user IoU for MIL-vs-uniform and MIL-vs-deviation; report effect direction + p-value. (Verified activating on smoke data: pooled per_user n=80, p-values produced.)
- [x] 1.6 Save aggregated stats + significance to `results/phase4_pointb_v2_evidence/pointb_v2_multiseed_stats.json`.
- [x] 1.7 Add a FAIR `random_interval` baseline (random contiguous interval whose length = the user's true anomaly length, random position) to the injection benchmark in the v2 script and to the aggregator (mean IoU + paired Wilcoxon vs MIL); replaces the degenerate predict-all uniform as the headline localization baseline. (Code done + smoke-verified; uniform retained for transparency.)
- [x] 1.8 Re-run 5 seeds with the updated script and re-aggregate so per_user_iou includes `random_interval`; headline = MIL vs random_interval (mean IoU + Wilcoxon). Update evidence_summary.json with the fair-baseline result. DONE: 5 seeds re-run full data. FAIR baseline random_interval mean IoU 0.113+/-0.008. MIL mean IoU 0.159 vs 0.113 = +41% relative (headline win on mean). Wilcoxon mil_vs_random_interval p=4.44e-13, median_diff=0 -> tie at per-user median (MIL bimodal hit-or-miss). evidence_summary.json updated with fair-baseline headline + honest median-tie caveat.

## 2. Weak-shape boundary (one falsifiable attempt)

- [x] 2.1 Add an optional temporal-smoothing variant (short moving average on attention/deviation) behind a CLI flag in the v2 script.
- [x] 2.2 Evaluate the variant on the same injection benchmark; record per-shape IoU. DONE (attn_smooth=3, full data): sudden_drop 0.131, sustained_low 0.119, zero 0.253, slow_drift 0.109 (uniform per-shape ~0.117-0.121).
- [x] 2.3 Apply the pre-committed gate: variant must raise sudden_drop AND slow_drift IoU above uniform without regressing zero/sustained_low; if it fails, stop tuning and document the measured applicability boundary. GATE NOT MET: sudden_drop now exceeds uniform (0.131 > 0.119, improved from 0.110) but slow_drift remains below uniform (0.109 < 0.117). Per pre-commitment: stopped tuning. Documented boundary: Point B v2 localizes drop-to-zero and sudden-drop morphology well (zero IoU ~0.30, 2.5x uniform); sustained-low marginal; gradual slow-drift is NOT reliably localized (at/below the predict-all floor).

## 3. Real-user qualitative localization

- [x] 3.1 Create `sgcc_phase4_pointb_v2_heatmap.py` that reads `pointb_v2_localization_features.csv` and reloads raw monthly consumption per `CONS_NO` via `load_sgcc` + `build_monthly_sequences`.
- [x] 3.2 Deterministically select cases: top `interval_confidence` G3-positive users plus contrast (G3-positive, low B score) users.
- [x] 3.3 Render per-user figures (raw curve + 34-month attention overlay + shaded predicted interval) to `results/phase4_pointb_v2_evidence/`, labeled qualitative/illustrative. (Smoke verified: 3 PNGs + heatmap_index.csv.)

## 4. A+B joint complementarity

- [x] 4.1 Create `sgcc_phase4_pointb_ab_complementarity.py` joining `sgcc_phase3_g3_artifacts.npz` (A scores/labels) with the B export by `CONS_NO`; validate and report matched/unmatched counts. (Verified: label-alignment check passed, matched=42372, unmatched=0.)
- [x] 4.2 Produce a coverage table: of G3-positive users, how many get a confident contiguous B interval (+ distribution of interval positions). (n_pos=3615, confident-contiguous=1807, fraction=0.500.)
- [x] 4.3 Produce 2-3 concrete case studies (real G3-flagged users, their localized months/morphology) and an A->B pipeline schematic figure.
- [x] 4.4 Write all outputs under `results/phase4_pointb_v2_evidence/`; assert no claim that B improves G3 global AUC/F1.

## 5. Verification and reporting

- [x] 5.1 `py_compile` all new/modified scripts; sanity-check that v2 still imports and runs a smoke seed. (Verified: v2 + 3 new scripts compile; smoke `--seed 123 --seed-suffix --attn-smooth 3` ran, wrote per_user_iou (80 users x 3 sources), aggregator activated Wilcoxon end-to-end.)
- [x] 5.2 Consolidate a short evidence summary (multi-seed stats + significance + weak-shape boundary + A+B coverage) referencing only verified numbers; no fabricated metrics, no G3-beating claim. DONE: `results/phase4_pointb_v2_evidence/evidence_summary.json`. Includes the honest IoU-distribution caveat (MIL wins on mean/pa-F1/attention-separation but loses to the degenerate predict-all uniform at per-user median IoU).
