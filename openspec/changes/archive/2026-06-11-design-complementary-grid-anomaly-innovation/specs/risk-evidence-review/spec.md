## ADDED Requirements

### Requirement: Monthly sequence construction
The system SHALL construct monthly multi-channel electricity usage sequences for self-supervised normal pattern modeling.

#### Scenario: Sequence fields generated
- **WHEN** point B preprocessing runs
- **THEN** the system MUST generate per-user monthly sequences containing available channels such as monthly mean, monthly standard deviation, monthly zero ratio, monthly missing ratio, baseline deviation, cumulative deviation, ranking deviation, or seasonal features

#### Scenario: Point A feature reuse
- **WHEN** monthly sequence channels overlap with point A features
- **THEN** the system MUST reuse the existing feature logic where practical rather than defining unrelated feature semantics

### Requirement: Normal-pattern training set selection
The system SHALL train the self-supervised normal pattern model using normal users or low-risk users.

#### Scenario: Label-based normal users
- **WHEN** labels are available
- **THEN** the system MUST support selecting normal-labeled users as the primary training subset

#### Scenario: Low-risk fallback
- **WHEN** model probabilities are available and label quality is uncertain
- **THEN** the system MAY support selecting low-risk users as an additional or alternative normal-pattern training subset

### Requirement: Masked reconstruction task
The system SHALL train a masked reconstruction task on monthly multi-channel sequences.

#### Scenario: Masked input reconstruction
- **WHEN** a normal training sequence is processed
- **THEN** the system MUST randomly mask months or channels and train the model to reconstruct the original values from context

#### Scenario: Reconstruction error output
- **WHEN** the trained model evaluates a user
- **THEN** the system MUST output user-level and month-level reconstruction error features

### Requirement: Future-window prediction task
The system SHALL train or compute a future-window prediction objective for normal usage pattern learning.

#### Scenario: Future prediction training
- **WHEN** a normal training sequence is processed
- **THEN** the system MUST use earlier months to predict one or more later-month windows or equivalent future targets

#### Scenario: Prediction error output
- **WHEN** the trained model evaluates a user
- **THEN** the system MUST output user-level and month-level prediction error features

### Requirement: Latent normal prototype distance
The system SHALL compute latent-space distance between each user and normal usage prototypes.

#### Scenario: Prototype construction
- **WHEN** normal users have latent representations
- **THEN** the system MUST construct at least one normal prototype from those representations using mean, median, or clustering-based centers

#### Scenario: Distance feature output
- **WHEN** a user is evaluated
- **THEN** the system MUST output a latent normal-prototype distance feature

### Requirement: Abnormal period localization
The system SHALL identify abnormal months or continuous abnormal periods from reconstruction or prediction errors.

#### Scenario: Top abnormal months
- **WHEN** month-level errors are available
- **THEN** the system MUST output the highest-error months or equivalent abnormal-period indicators for each user

#### Scenario: Period summary
- **WHEN** consecutive high-error months exist
- **THEN** the system SHOULD summarize the abnormal period length and location

### Requirement: Fusion feature export
The system SHALL export point B self-supervised deviation features for fusion with point A models.

#### Scenario: Feature file generated
- **WHEN** point B evaluation completes
- **THEN** the system MUST write a feature file containing user identifier, reconstruction error features, prediction error features, latent distance features, and abnormal-period features

#### Scenario: Scale-invariant deviation scoring
- **WHEN** point B computes reconstruction or prediction deviation scores
- **THEN** the system MUST provide scale-invariant or normal-reference-calibrated scores rather than relying only on raw magnitude-sensitive MSE values

#### Scenario: Multi-prototype normal reference
- **WHEN** latent normal-pattern distances are computed
- **THEN** the system SHOULD support multiple normal prototypes or clustered normal references and report nearest-prototype or cluster-calibrated deviation features

#### Scenario: No fabricated metrics
- **WHEN** point B outputs reports or summaries
- **THEN** the system MUST only report metrics computed from actual labels and predictions, or mark them as pending validation

### Requirement: True point A baseline fusion
The system SHALL validate point B fusion against the true point A G3 baseline, not only against a hand-crafted plus RMT baseline.

#### Scenario: G3 artifact available
- **WHEN** Phase 4 fusion runs
- **THEN** the system MUST use or generate G3 artifacts that include hand-crafted features, RMT features, Transformer_PCA16 features, GBDT OOF scores, ensemble OOF score, labels, and user identifiers

#### Scenario: Correct baseline comparison
- **WHEN** fusion metrics are reported
- **THEN** the system MUST compare G4 fusion variants against G3 = RMT + Transformer_PCA16 + GBDT Ensemble and clearly separate any weaker A_plus_RMT diagnostic result

#### Scenario: Selective point B fusion
- **WHEN** point B features are fused with G3
- **THEN** the system MUST screen or weight point B features to avoid blindly concatenating noisy full point B features

#### Scenario: Separate AUC and F1 objectives
- **WHEN** fusion is optimized or summarized
- **THEN** the system MUST report AUC-oriented validation and F1 threshold-calibrated validation separately, using only out-of-fold or held-out evidence