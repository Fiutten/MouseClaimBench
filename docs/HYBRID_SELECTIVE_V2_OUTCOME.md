# Hybrid selective v2 outcome

## Status

This document reports a post-submission research evolution. It does not modify
the submitted MouseClaimBench manuscript, its results, or its claims. The
version-2 protocol was frozen at commit `3317d8b`. The consumed development
matrix, frozen model, implementation, and one confirmatory result are preserved
in separate commits on `hybrid-selective-v2`.

## Question

The frozen question was whether a constrained selective policy could retain
non-negotiable evidence boundaries while providing calibrated, non-trivial
risk-coverage performance on unseen claim-assessment regimes. The policy used
one logistic model per non-constant claim, isotonic calibration, abstention,
and hard support vetoes. Directional evidence came from the additive-noise
model implemented by `causal-learn==0.1.4.8`.

The model used only 3740 previously consumed cases. Deterministic, disjoint
partitions contained 2244 fitting cases, 748 calibration cases, and 748 locked
development-audit cases. The confirmatory partition contained 3600 new cases
from ten regimes, three sample sizes, three noise levels, and 40 seeds per cell.
No confirmatory case was used for fitting, calibration, or threshold selection.

## Frozen result

The primary endpoint did not pass. The constrained hybrid produced:

- zero semantic support-veto violations
- `0.9773` coverage
- `0.0472` selective error, with one-sided CP95 upper bound `0.0491`
- `0.0200` false-authorization fraction among supported decisions
- selective error within `0.02` of the unconstrained comparator

The directional condition failed. ANM attempted a direction in `2681/3600`
cases, giving `0.7447` coverage. Attempted accuracy was `0.5173` across all
regimes and `0.6515` within regimes with a declared structural direction. Both
are materially below the frozen `0.75` requirement.

The failure is structured rather than random. ANM was nearly perfect for the
direct exponential regime and strong for linear non-Gaussian data. It was
predominantly reversed for the post-nonlinear regime, close to chance under
measurement error, and frequently forced a direction under confounding and
collider selection. The latter two violate its no-hidden-confounder assumption.
The post-nonlinear regime violates the additive-noise form.

## Non-trivial audit

All-claim aggregate error is optimistic because four claims are always false
and computational reproducibility is always true. Restricting evaluation to the
six claims whose labels vary across regimes gives:

| Policy | Coverage | Selective error | False-authorization fraction |
|---|---:|---:|---:|
| Unconstrained selective logistic | 1.0000 | 0.0474 | 0.0478 |
| Constrained selective hybrid | 0.9621 | 0.0799 | 0.0260 |
| ANM-predictor ablation | 0.9755 | 0.0948 | 0.0194 |
| Uncalibrated constrained ablation | 0.9623 | 0.0791 | 0.0226 |

Hard vetoes reduce false support and guarantee semantic conformance, but they
increase selective error. Isotonic calibration did not improve confirmatory
Brier score or ten-bin ECE. The ANM predictors reduce error relative to their
predictor ablation, but they do not solve the direction failure and slightly
increase false authorization. These are mixed effects, not evidence of general
superiority.

## Publication assessment

This experiment is reproducible and scientifically useful as a negative result,
but it is not a sufficiently strong basis for the intended new Q1 claim. A
defensible statement is narrower:

> A constrained selective layer can enforce semantic support boundaries with
> non-trivial coverage, but the tested ANM gate is not a reliable general
> directional-evidence component across assumption violations.

It is not defensible to claim universal mechanistic identification, lower
overall decision error, improved human decision quality, cross-domain
generality, or independent biological validation.

## Consequence

The version-2 cases are now consumed. They may be used for error analysis and
future development, but never again as independent confirmation. Replacing ANM,
changing the direction threshold, modifying labels, or retuning the classifier
would define a version-3 method. Any version-3 claim requires another frozen,
non-overlapping benchmark or an external benchmark whose labels were not used
during redesign.
