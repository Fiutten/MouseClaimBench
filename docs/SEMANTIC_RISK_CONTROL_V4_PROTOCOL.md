# Semantic risk control v4: frozen protocol

## Purpose and separation from v3

Version 4 is a new prospective study on branch `semantic-risk-control-v4`.
Version 3 remains frozen at commit `71a4d10`; its artifacts, conclusions, and
negative external result are not overwritten. The v4 design was fixed on
2026-08-02 before downloading or evaluating the twelve external datasets listed
in `configs/benchmarks/causal_chambers_v4_population.yaml`.

The scientific target is deliberately narrower than automatic truth
verification. MouseClaimBench decides whether a declared evidence contract
permits one bounded claim. A positive decision is an authorization under that
contract. An abstention is not evidence that the claim is false.

## Primary external design

The external population contains all non-image Causal Chambers datasets that
were absent from the v3 `lt_test_v1` and `wt_test_v1` audit. Assignment to target
calibration, risk lock, and final evaluation is deterministic and stratified by
chamber. The rule depends only on dataset names and a frozen SHA-256 namespace.

The physical experiment CSV is the inferential unit. Directed pair records
within an experiment share observations and variables. They are therefore not
treated as independent replicates. For each experiment, the primary risk event
is binary: at least one authorized claim is false. This deliberately stringent
endpoint allows an exact binomial upper confidence bound without pretending that
pair rows are independent. The adapter selects at most one observable direct
edge and one non-edge control per experiment using a frozen SHA-256 order. This
prevents experiments with more observable pairs from receiving more inferential
weight.

The certificate is non-degenerate. It requires an upper confidence bound on
experiment failure at or below 0.10, a lower confidence bound on authorized
experiment coverage at or above 0.10, a lower confidence bound on positive
recovery at or above 0.05, and zero semantic-gate violations. A policy that
abstains everywhere cannot pass.

## Locked sequence

1. Calibration datasets may select thresholds.
2. Risk-lock datasets test the complete contract without threshold changes.
3. Final datasets may be opened only if the risk lock passes.
4. Shift diagnostics are warnings. They cannot validate, restore, or extend a
   certificate.
5. Results remain negative if the available independent-unit count cannot
   support the promised confidence, even when point estimates look favorable.

## Comparators and router confirmation

The comparison includes a fixed 0.5 threshold, confidence-only calibration, the
evidence contract alone, unconstrained LTT, semantic LTT without an activation
floor, and the complete non-degenerate policy. These comparators isolate the
contribution of semantic constraints, statistical risk control, and positive
activation.

The association-aware direction router is confirmed in a new seed namespace and
new structural equations. Its rule is fixed before generation. Independent
regimes must receive no directional attempt, while valid routed regimes must
retain at least 0.80 attempted-direction accuracy. Association is established
only when either Pearson or Spearman association has absolute magnitude at least
0.10 and its Bonferroni-adjusted two-test p-value is below 0.01. This is a routing
precondition, not a causal or directional test.

## Validity of the knowledge profile

The profile is author-proposed. ARRIVE 2.0, NIH rigor guidance, FAIR, and NIST AI
RMF motivate transparent design, provenance, reproducibility, validity, and
scope reporting. None of those sources endorses the exact MouseClaimBench
blocks, mappings, or thresholds. Topology, direction, intervention, and
structure-function requirements are explicit methodological proposals that
still require independent domain-expert validation.

## Interpretation boundary

Passing v4 would establish bounded empirical support for a non-degenerate claim
authorization contract in the declared Causal Chambers population. It would not
establish a universal guarantee, scientific truth verification, causal validity
from association, a complete mouse-brain model, or superiority over methods
whose target decision problem is different.
