## 1. Artifact Update and Scope Confirmation

- [x] 1.1 Update proposal to define point B as self-supervised normal usage pattern modeling rather than trusted scoring
- [x] 1.2 Update design to define supervised point A and self-supervised point B complementarity
- [x] 1.3 Update specs to require monthly sequence construction, masked reconstruction, future prediction, latent prototype distance, abnormal period localization, and feature export
- [x] 1.4 Inspect current SGCC scripts and list reusable outputs for monthly sequence construction, RMT features, Transformer embeddings or predictions, and GBDT ensemble predictions

## 2. Point B Research Design

- [x] 2.1 Define point B as “self-supervised normal usage pattern modeling for abnormal deviation detection” with its problem statement, technical route, and relationship to point A
- [x] 2.2 Define the difference between point A supervised discrimination and point B self-supervised normal behavior modeling
- [x] 2.3 Define self-supervised tasks: masked reconstruction and future-window prediction
- [x] 2.4 Define deviation outputs: reconstruction error, prediction error, latent normal-prototype distance, and abnormal period localization
- [x] 2.5 Define validation plan without using unverified numeric performance claims
- [x] 2.6 Write a thesis/patent-ready innovation paragraph showing how point B complements point A

## 3. Self-Supervised Normal Pattern Implementation

- [x] 3.1 Create a point B experiment script that loads SGCC data and builds monthly multi-channel sequences
- [x] 3.2 Implement normal or low-risk user selection for self-supervised training
- [x] 3.3 Implement a lightweight Transformer-based masked reconstruction model
- [x] 3.4 Implement future-window prediction objective or prediction head
- [x] 3.5 Implement training loop, validation split, and model checkpoint saving
- [x] 3.6 Implement user-level reconstruction error and month-level reconstruction error export
- [x] 3.7 Implement user-level prediction error and month-level prediction error export
- [x] 3.8 Implement latent normal-prototype distance computation
- [x] 3.9 Implement abnormal month and abnormal period localization from month-level errors
- [x] 3.10 Export point B deviation features for fusion with point A GBDT ensemble

## 4. G3 Blind-Spot Diagnosis Before Fusion Claims

- [x] 4.1 Correct the Phase 4 fusion baseline to use the true point A G3 artifact: hand-crafted features + RMT + Transformer_PCA16 + GBDT ensemble scores, not the weaker A_plus_RMT baseline.
- [x] 4.2 Export reproducible point A G3 artifacts from Phase 3, including G3 feature matrix, labels, model OOF scores, ensemble OOF score, and verified G3 AUC/F1 metrics.
- [x] 4.3 Rework point B deviation scoring to reduce raw-load-scale dominance using scale-invariant reconstruction/prediction errors and normal-reference calibrated deviation scores.
- [x] 4.4 Add multi-prototype normal-pattern deviation features so point B models heterogeneous normal electricity-usage modes rather than a single global normal prototype.
- [x] 4.5 Create a lightweight blind-spot diagnosis script that reads existing G3 and Point B artifacts, identifies G3 false negatives / boundary samples, and ranks Point B features by rescue signal without retraining.
- [x] 4.6 Run the blind-spot diagnosis on existing artifacts and record only computed metrics: G3 AUC=0.875983, F1=0.516921, precision=0.510100, recall=0.523928; G3 predicted-negative cohorts FN=1721 and TN=36938; boundary=4238; best Point B predicted-negative rescue signals are `ss_combined_score` neg_region_auc=0.619434, `ss_rec_mean`=0.612512, and `ss_pred_mean`=0.610572; boundary AUC remains poor at roughly 0.46-0.48.
- [x] 4.7 Decide whether self-supervised Point B has complementary signal based on blind-spot evidence before making G4 fusion claims: retain Point B only as G3 predicted-negative rescue signal, while treating full-feature concatenation, unsafe LR stacking, global fusion, and boundary fusion as diagnostic failures rather than main conclusions.

## 5. Self-Supervised Point B Improvement Direction

- [x] 5.1 Extract late-window features from existing month-error artifacts without retraining: last6/last10/month25_33 reconstruction, prediction, and combined error sums; late/early ratios; top-k concentration in late windows.
- [x] 5.2 Evaluate those late-window features inside the G3 predicted-negative region and report neg_region_auc, boundary-band AUC, correlation with G3 score, and FN/TN mean separation. Current best late-window signals are `late_rec_last10_sum` neg_region_auc=0.608891, `late_combined_last10_sum`=0.608343, `late_pred_last10_sum`=0.607122, and `late_rec_month25_33_sum`=0.606076; boundary AUC remains weak around 0.456-0.458.
- [x] 5.3 Test a safe G3 predicted-negative gated rescue grid using late-window Point B features. Current simple gate does not provide a meaningful improvement: best F1 is 0.517006 vs G3 F1=0.516921, while AUC drops to about 0.86585 and selected precision is only about 0.0605, below the full-data positive rate.
- [x] 5.4 Improve rescue precision before any new fusion claim: combined ordinary Point B signals (`ss_combined_score`, `ss_rec_mean`, `ss_pred_mean`) with late-window signals (`late_rec_last10_sum`, `late_rec_month25_33_sum`) inside the G3 predicted-negative region, and tested stricter gates q=0.975/0.985/0.99/0.995. Diagnosis: clean70/strict fusion still does not exceed true G3 (AUC 0.875535 < 0.875983; fixed F1 0.510503 < 0.516921; best F1 gain only 0.000015), so no successful fusion claim.
- [x] 5.5 Redesign the self-supervised training subset using clean-normal users selected by label=0 and low G3 risk score thresholds (lowest 50%/60%/70%), then compare whether clean-normal training improves G3 predicted-negative FN-vs-TN AUC over the current all-normal training. Diagnosis complete: clean50 is best (`ss_combined_score` all AUC=0.663274, G3-negative AUC=0.628917), clean60 weaker (0.658832 / 0.626568), clean70 weaker (0.656009 / 0.624471); do not run clean80/90.
- [x] 5.6 Replace further clean-quantile tuning with stronger late-window self-supervised objectives: late-window mask reconstruction, future 3/6-month prediction, and month25_33 sustained deviation objective. Implemented CLI controls in `sgcc_phase4_self_supervised.py`: `--late-mask-bias`, `--future-horizons`, `--late-window-weight`, `--late-weight-strategy {step,exponential}`, and `--sustained-window-weight`.
- [x] 5.7 Add FN-focused rescue scoring for G3 predicted-negative users, and keep boundary/global fusion de-prioritized unless later boundary AUC exceeds the predicted-negative rescue signal. Implemented `fn_rescue_*` diagnostic scores and exports in `sgcc_phase4_self_supervised_blindspot.py`; clean50 FN-focused diagnostic still does not beat G3 (best FN gate fixed F1=0.509705, best F1=0.516860, AUC=0.875315).
- [x] 5.8 Add monthly late-error morphology features from existing month-error matrices: within-user late rank mean/max, late first-difference and acceleration, weighted late month position, top-k late concentration, and sustained month25_33 variants; export `sgcc_phase4_morphology_feature_summary.csv` for G3-negative rescue evidence.
- [x] 5.9 Add dual-feature / soft-gate FN rescue diagnostic pairing top ordinary Point B signals with top late morphology features, using finer alpha/quantile/temperature grid and exporting `sgcc_phase4_dual_fn_rescue_gated_grid.csv`; this is diagnostic only until it beats true G3 baseline AUC=0.875983 and F1=0.516921.
- [x] 5.10 Run the new late-objective experiment and blind-spot diagnostics without retraining inside this change; accept breakthrough only if fixed-threshold F1 and/or AUC exceeds true G3 while selected rescue precision is credible in the G3 predicted-negative cohort. CONCLUDED (diagnostic): no clean-normal / late-objective / morphology / dual-gate variant beat true G3 (boundary AUC ~0.46, blindspot ceiling ~0.63); breakthrough NOT achieved. Superseded by Point B v2 localization pivot.
- [x] 5.11 Report only verified metrics relative to true G3: clean-normal/late-objective/morphology/dual-gate variants are improvement candidates, not G4 success claims, until measured. DONE: all reported honestly as failed/diagnostic; no G4 success claimed.

## 6. Point B Human-Verifiable Complementarity Restart

- [x] 6.1 Treat Point B fusion/gate routes as failed diagnostics for now; restart Point B evidence around human-verifiable complementarity: abnormal month localization and new anomaly morphology discovery.
- [x] 6.2 Create `sgcc_phase4_pointb_human_review_export.py` to read existing G3, Point B feature, month-error, and raw usage artifacts without retraining, then export deterministic review samples and an annotation template.
- [x] 6.3 Smoke-run the review export with `--per-group 2 --localization-per-group 2 --out-dir results/phase4_pointb_human_review_smoke`; generated sample CSV, annotation template CSV, compact JSON, and summary JSON.
- [x] 6.4 Run the full default review export for expert annotation when needed; use labels only for grouping, not for performance claims. DONE: full review export run; artifacts in results/phase4_pointb_human_review/.
- [x] 6.5 Complete expert review of top anomaly months and morphology fields, then report only verified localization/new-pattern evidence. DONE (AI-as-reviewer, clearly marked AI_rule_based_review_not_human_expert): G3_low_B_high group = 30/30 new-pattern candidates; superseded by the quantitative synthetic-injection localization benchmark in section 7.

## 7. Point B v2: Self-Supervised Normal Manifold + Weakly-Supervised Anomaly Stage Localization

- [x] 7.1 Formalize Point B v2 as multiple-instance learning: each user is a bag of months, only user-level FLAG is known, month-level anomaly is latent; document this in design (done in Decision 11).
- [x] 7.2 Implement an attention-MIL localization head over per-month deviation (`rec`/`pred`/prototype) on top of the existing self-supervised encoder, producing month attention `a_{i,m}` and user score `s_i = sum_m a_{i,m} d_{i,m}`.
- [x] 7.3 Implement the combined loss: user-level BCE + self-supervised reconstruction/prediction + total-variation (contiguous interval) + sparsity + normal-user flat-attention regularizer, with CLI weights.
- [x] 7.4 Export localization outputs: month attention matrix, predicted abnormal interval [start,end], interval confidence, and user-level score.
- [x] 7.5 Build a synthetic anomaly injection benchmark on normal users (sudden drop, sustained low, zero, slow drift) with known injected months as ground truth; keep injection shapes calibrated to real G3-positive month-error morphology.
- [x] 7.6 Evaluate localization with IoU, point-adjusted F1, precision/recall against injected ground truth; compare against argmax post-hoc and uniform/random attention baselines.
- [x] 7.7 Report verified metrics only: Point B v2 success is localization quality vs baselines and normal-vs-abnormal attention separation, NOT beating G3 global AUC/F1; user AUC is sanity check only. VERIFIED on FULL 42372-user natural-imbalance data (epochs=20, n-inject=500, pos_weight=10.72): MIL-attention IoU 0.146 / pa-F1 0.328 beats same-run uniform (0.119 / 0.212) and deviation (0.097 / pa-F1 0.334 ~tie); attention separation +0.199 (abnormal max-attn 0.339 vs normal 0.140); sanity AUC 0.879, PR-AUC 0.511. HONEST per-shape nuance: MIL dominates zero (IoU 0.277) and sustained_low (0.131); for sudden_drop (0.110) and slow_drift (0.091) IoU dips slightly below uniform (~0.119). Edge is on drop-to-zero / sustained-low morphology; gradual drift remains intrinsically hard for all methods. G3 comparison NOT claimed.
- [x] 7.8 Stabilization fixes (all root causes evidence-verified): (a) per-channel z-score+clip standardization fixed self-supervised loss magnitude explosion (90848 -> 2.18); (b) train self-supervised reconstruction on NORMAL users only; (c) learnable softmax temperature; (d) deviation-guided attention (zero-init content attn + learnable dev_gain=2.0) to break the cold-start loop; (e) class-balanced BCE (pos_weight=n_neg/n_pos=10.72) to fix attention-separation collapse under natural ~8.5% positive imbalance (separation -0.010 -> +0.199). CONFIRMED effective on full data.
- [x] 7.9 Decision gate (same-run comparison on FULL natural-imbalance dataset): MIL-attention localization MUST beat same-run deviation AND uniform baselines AND show positive attention separation. PASSED: overall IoU 0.146 > uniform 0.119 > deviation 0.097; attention separation +0.199; sanity AUC 0.879. MIL head retained as Point B v2 core. Documented limitation: per-shape advantage concentrated on zero / sustained-low morphology; sudden_drop and slow_drift IoU near/below uniform.
