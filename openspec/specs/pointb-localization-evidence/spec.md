# pointb-localization-evidence Specification

## Purpose
TBD - created by archiving change pointb-v2-publication-evidence. Update Purpose after archive.
## Requirements
### Requirement: Real-user qualitative localization visualization
The system SHALL render, for deterministically selected real G3-positive users, the per-month attention vector overlaid on the user's raw monthly consumption curve, with the predicted anomalous interval shaded, using only existing exports and `data set.csv` (no retraining). The visualization MUST be labeled as qualitative/illustrative and MUST NOT compute IoU/F1 against any real-user ground truth.

#### Scenario: Render attention-over-curve for a high-confidence G3-positive user
- **WHEN** the visualization runs on `pointb_v2_localization_features.csv` joined with raw monthly consumption for the same `CONS_NO`
- **THEN** it produces a figure per selected user showing the raw curve, the 34-month attention overlay, and the shaded `[pred_interval_start, pred_interval_end]`, saved under `results/phase4_pointb_v2_evidence/`

#### Scenario: Deterministic case selection without fabricated labels
- **WHEN** users are selected for visualization
- **THEN** selection is deterministic (e.g., top `interval_confidence` G3-positive users plus contrast cases) and no month-level anomaly label is invented for real users

### Requirement: A+B joint complementarity artifact
The system SHALL join Point A (G3) user scores/labels from `sgcc_phase3_g3_artifacts.npz` with Point B's `user_score`, predicted interval, and attention by `CONS_NO`, and produce a pipeline schematic, concrete case studies, and a coverage table quantifying how many G3-positive users receive a confident contiguous B interval. The artifact MUST NOT claim Point B improves G3 global AUC/F1.

#### Scenario: Join coverage is validated and reported
- **WHEN** the G3 artifact and the B export are joined by `CONS_NO`
- **THEN** the number of matched and unmatched users is reported before any summary table is produced

#### Scenario: Complementarity case studies are concrete and bounded
- **WHEN** the joint artifact is generated
- **THEN** it presents 2-3 real G3-flagged users with their localized months and a coverage summary, framed as A answers "who" and B answers "when/what morphology", with no global-detection improvement claim

