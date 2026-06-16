## ADDED Requirements

### Requirement: Multi-seed statistical robustness
The system SHALL support a `--seed` CLI argument on `sgcc_phase4_pointb_v2_localization.py` that deterministically seeds numpy, torch, and the injection RNG, and SHALL aggregate localization metrics across at least 5 distinct seeds, reporting mean, standard deviation, and 95% confidence interval for MIL-attention IoU, point-adjusted F1, and attention separation.

#### Scenario: Per-seed deterministic run
- **WHEN** the script is run twice with the same `--seed`
- **THEN** the reported localization metrics are identical

#### Scenario: Aggregate across >=5 seeds
- **WHEN** the aggregator reads per-seed `pointb_v2_localization_metrics.json` outputs from at least 5 seeds
- **THEN** it reports mean +/- std and 95% CI for IoU, point-adjusted F1, and attention separation

### Requirement: Paired significance testing vs baselines
The system SHALL run a paired non-parametric test (Wilcoxon signed-rank) on per-user IoU between MIL attention and each baseline (uniform, deviation), and report the effect direction and p-value alongside the aggregated means.

#### Scenario: Significance vs uniform and deviation
- **WHEN** per-user IoU values for MIL, uniform, and deviation on the synthetic injection set are available
- **THEN** the system reports a Wilcoxon p-value and effect direction for MIL-vs-uniform and MIL-vs-deviation

### Requirement: Measured weak-shape boundary
The system SHALL evaluate one principled weak-shape variant (temporal smoothing of attention/deviation) against a pre-committed gate: it MUST raise sudden_drop AND slow_drift IoU above the uniform baseline without regressing zero/sustained_low. If the gate is not met, the system SHALL record the measured boundary and state the method's validated applicability scope rather than continue tuning.

#### Scenario: Variant passes the gate
- **WHEN** the smoothed variant is evaluated on the same injection benchmark
- **THEN** if sudden_drop and slow_drift IoU both exceed uniform without zero/sustained_low regression, the variant is retained with verified numbers

#### Scenario: Variant fails the gate
- **WHEN** the smoothed variant does not meet the gate
- **THEN** no further tuning is performed and the measured per-shape boundary is documented honestly as the method's applicability scope
