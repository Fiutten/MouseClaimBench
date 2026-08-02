# MouseClaimBench v3: frozen scientific protocol

## Status and separation from earlier work

This protocol was frozen on 2 August 2026 before accessing the outcomes of the
new external blocks. MouseBrainBench is the submitted first paper. MouseClaimBench
v0-v2 is development work for the second paper. All previously inspected synthetic,
Allen, Sensorium, MICRONS, Tuebingen, and policy-selection results are therefore
consumed. They cannot serve as v3 confirmation.

The v3 contribution is not another claim score and it is not a new causal discovery
algorithm. It is a risk-controlled knowledge-based authorizer. The system first
checks whether the evidence required by a claim is semantically admissible. A
learned score may rank candidates that survive this gate, but it cannot repair a
missing intervention, an unresolved direction, or an absent replication block.
Learn-Then-Test then selects thresholds for the complete gated policy.

## Primary estimand

For claim family `c`, let `A_c(X)` indicate that the system authorizes support and
let `Y_c` indicate that the reference claim is true. The Semantic False
Authorization Risk is

`SFAR_c = P(Y_c = 0 | A_c(X) = 1)`.

Equivalently, `1 - SFAR_c` is the precision of authorized supports. The target is
`SFAR_c <= 0.05`. Risk is controlled separately for every declared variable claim.
The global confidence is 0.95, with Bonferroni allocation across claim families.
The complete case, not each claim decision, is the independent calibration unit.

This guarantee is population-specific. It requires calibration cases to be i.i.d.
with future cases from the certified population. It is invalid after an unaccounted
domain shift. Cross-domain CausalBench and IBL evaluations are therefore transport
audits, not guaranteed applications of a synthetic certificate.

## Directional evidence

No single direction method is treated as universally valid. Routing uses declared
assumptions and gives priority to controlled intervention evidence. DirectLiNGAM is
eligible only for acyclic linear non-Gaussian settings without hidden confounding.
ANM is eligible only for continuous additive-noise settings without hidden
confounding. Hidden confounding, material unmodelled measurement error, selection
bias, unknown assumptions, and numerical failures force abstention. Observational
direction remains evidence about identifiability under assumptions, not causal
proof.

## Independent semantic execution

The existing prioritized Python rule engine remains authoritative. A second backend
will encode the same profile as an Answer Set Program executed by clingo. Decision
equivalence is required over all 625 assignments for the four-block mechanistic
claim, all complete assignments for claims with at most four blocks, deterministic
boundary assignments for larger claims, and a fixed randomized property suite. A
single disagreement fails the corresponding primary endpoint.

## External blocks

The synthetic v3 block contains 7,200 cases from sixteen structural-equation
regimes. It is the in-distribution finite-sample confirmation. Causal Chambers adds
real controlled physical systems with experiment-level splits. CausalBench uses
K562 only for calibration and RPE1 as a locked cell-line transport audit. The IBL
Brain-Wide Map supplies a new mouse-neuroscience block split by animal. IBL can
support observational predictive or reproducibility statements, but cannot
authorize directed, causal, mechanistic, or digital-twin claims in this protocol.

## Interpretation gate

The v3 result is strong only if all primary endpoints pass. Zero semantic violations
alone is a software property, not enough for a Q1 claim. High prediction without a
simultaneous non-trivial SFAR certificate is also insufficient. External success
without a valid calibration population is reported as transfer performance, not as
a finite-sample guarantee. The exact machine-readable protocol is
`configs/benchmarks/semantic_risk_control_v3.yaml`.
