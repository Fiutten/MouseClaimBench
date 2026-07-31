# MIS 2.0 Calibration Protocol

## Purpose

MIS 2.0 is a methodological extension of the current Mechanistic
Identifiability Score. Its goal is not to make MouseBrainBench more permissive.
Its goal is to quantify when a non-compensatory claim gate is safe,
conservative, or unsafe under known truth.

This protocol is deliberately synthetic. It does not provide biological evidence
about mouse brain data. It provides a controlled test bed for deciding whether
MIS can be defended as a general claim-audit methodology beyond the submitted
MouseBrainBench manuscript.

## Scientific Question

The central question is:

```text
Can a conjunctive mechanistic-identifiability gate reject non-mechanistic
predictive or reproducible cases without becoming unusably conservative under
reasonable noise and sample-size regimes?
```

The answer must be expressed in false-positive and false-negative terms:

- a false positive means a non-mechanistic case passed the full MIS gate;
- a false negative means a true mechanistic case failed the full MIS gate;
- false positives are more damaging for claim auditing;
- false negatives identify conservative operating regions and data requirements.

## Current Synthetic Truth Regimes

The current suite uses synthetic regional time-series with known truth.

Positive truth regimes:

- clean directed truth;
- directed truth with moderate noise;
- directed truth with few sessions;
- directed truth with low signal-to-noise ratio.

Negative truth regimes:

- common drive with high reproducibility;
- topology-like regional specificity without temporal direction;
- temporal direction without topology specificity;
- prediction without true topology;
- noisy common drive.

These cases are intentionally simple. They are not intended to model cortical
biophysics. They isolate logical failure modes that a claim gate must not
confuse.

## Implemented Benchmarks

### Nominal calibration

Command:

```bash
mousebrainbench-mis2-synthetic
```

Artifacts:

```text
results/mis2_synthetic_calibration/summary.json
results/mis2_synthetic_calibration/summary.md
```

Current result:

```text
false_positive_rate = 0.0000
false_negative_rate = 0.1667
```

Interpretation:

The current nominal gate rejects all designed non-mechanistic cases. The false
negatives are concentrated in low-SNR directed truth. That is acceptable for a
claim-audit gate, but it must be reported as conservativeness.

### Threshold sensitivity

Command:

```bash
mousebrainbench-mis2-sensitivity
```

Artifacts:

```text
results/mis2_threshold_sensitivity/summary.json
results/mis2_threshold_sensitivity/summary.md
```

The sweep varies:

- observation noise;
- number of sessions;
- threshold profiles.

Each operating cell is classified as:

- `safe`: false-positive rate is zero and false-negative rate is low;
- `conservative`: false-positive rate is zero but false-negative rate is high;
- `dangerous`: false-positive rate is non-zero while sensitivity appears good;
- `unstable`: false-positive and false-negative rates are both problematic.

Current result:

```text
safe cells = 30
conservative cells = 30
dangerous cells = 0
unstable cells = 0
```

Interpretation:

The current gate is conservative rather than unsafe in these designed cases.
Strict topology thresholds are especially conservative. This matters because a
reviewer could correctly argue that arbitrary strictness may reject valid
mechanistic signals. MIS 2.0 should therefore be defended as a claim-protection
framework, not as a maximally sensitive detector.

## Criteria for a Stronger Methodological Paper

MIS 2.0 becomes a credible standalone methodological contribution only if future
work adds:

1. broader synthetic families with graded topology mismatch;
2. calibrated threshold selection rather than fixed heuristic multipliers;
3. uncertainty intervals over false-positive and false-negative rates;
4. comparison against compensatory scoring and correlation-only evaluation;
5. at least one empirical positive case and one empirical negative case;
6. explicit reporting of data regimes where the gate becomes conservative.

## Current Decision

The current MIS 2.0 extension is useful and technically sound as a development
line. It should not yet be presented as a universal standard. Its current value
is narrower:

```text
MouseBrainBench can now quantify whether its own mechanistic claim gate is
safe or conservative under controlled truth-known perturbations.
```

That is a solid next-step contribution, but not yet a complete second paper.

## Adversarial Claim Evaluation

The next layer is implemented by:

```bash
mousebrainbench-claim-adversarial
```

It compares three evaluators:

- `correlation_only`;
- `compensatory_score`;
- `claim_gate`.

The benchmark deliberately includes high-prediction common-drive cases,
topology-without-direction cases, direction-without-topology cases, spatially
confounded structure-function cases, and positive controls. Its purpose is to
show whether standard or compensatory evaluation authorizes claims that the
non-compensatory gate correctly blocks.

The aggregate reviewer-facing check is:

```bash
mousebrainbench-claim-attack-suite
```

This produces a risk report rather than a new scientific claim.
