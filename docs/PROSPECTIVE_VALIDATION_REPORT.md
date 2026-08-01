# Prospective validation report

## Executive decision

The completed package is methodologically stronger than the previous internal
benchmark, but it does **not** support a strong Q1 claim that the current
non-compensatory contract is generally superior. The frozen primary endpoint
failed. This result must remain visible because the protocol, comparator, and
evaluation partition were separated before execution.

MICRONS produces a positive and useful result. The fixed
`all_pairs/readout_location` association survives directed-dyad robust
uncertainty and a network-preserving residual permutation in discovery and two
non-overlapping hold-out windows. This strengthens the local observational
case. It does not repair the failed policy-comparison endpoint because the two
experiments answer different questions.

## What was completed

The protocol was frozen at commit `23bd4c7`. A regularized probabilistic
comparator was then trained on 500 cases from the five development SEM regimes.
Its coefficients were frozen before any prospective case was generated. Nine
new regimes were evaluated over three sample sizes, three noise levels, and 40
seeds per cell. The resulting test contains 3,240 cases and 32,400 claim
decisions per policy.

The evidence contract obtained an FPR of `0.01840` and an FNR of `0.21197`. It
substantially reduced overclaiming relative to the prediction shortcut, whose
FPR was `0.27054`. It did not outperform the development-trained probabilistic
comparator, which obtained FPR `0.00556` and FNR `0.18981`. The compensatory
baseline also had fewer total errors because its FNR was lower. All three
components of the frozen primary success rule therefore failed.

The main error source is directional evidence. For the contract, `directed`
has prospective FPR `0.180` and FNR `0.588`. The learned comparator uses the
other evidence states as context and reduces directional FPR to `0.024`, with
FNR `0.566`. This does not prove that a learned policy is universally better.
It does show that the current single-block direction authorization is not a
credible centerpiece for a superiority claim.

## MICRONS result

The connected-pair coefficient is positive in all windows after adjustment for
log distance, squared log distance, pre- and post-synaptic degree, and coarse
cell-type agreement. Dyadic standard errors are `1.57` to `1.78` times the
naive HC1 errors. Nevertheless, all dyadic two-sided p-values remain below
`1.8e-7`, and all 1,000-permutation one-sided p-values equal `0.000999`.

The covariance implementation was checked in 500 null and 500 positive
simulations with sender/receiver dependence. Under the null, HC1 rejected
`20.8%` of trials, whereas the dyadic test rejected `6.6%` and achieved `93.4%`
coverage. Under the fixed positive effect, dyadic power was `89.2%`. This is a
bounded calibration, not a proof for arbitrary network dependence.

## Reviewer-facing assessment

A demanding reviewer can still make four decisive criticisms. First, the
prospective superiority endpoint is negative. Second, the knowledge profile is
author-proposed and has no independent expert content validation. Third, the
directional diagnostic is poorly calibrated in the new regimes. Fourth, all
MICRONS windows belong to one observational resource and the node-permutation
test requires residual exchangeability.

Accordingly, the current package is valuable as transparent negative evidence,
a robust MICRONS extension, and a design basis for the next version. It is not
sufficiently strong for a new Q1 paper whose central thesis is that the existing
hard contract outperforms alternative decision policies. Presenting it that way
would invite justified major criticism.

## Scientifically defensible continuation

The next version should replace the brittle binary direction block with a
calibrated directional evidence model and should separate semantic vetoes from
probabilistic authorization. A constrained hybrid can preserve non-negotiable
claim boundaries while learning how supporting blocks interact. Its primary
outcome should be selective risk at declared coverage, not raw accuracy alone.
It must be trained on all current regimes and evaluated on a new protocol with
non-overlapping generators. The present prospective cases may become
development data, but they cannot be reused as confirmatory evidence.

For a Knowledge-Based Systems submission, independent expert review of the
profile would still materially strengthen content validity. Without it, the
scope must remain computational policy validation. Human decision quality,
cross-domain generality, causal biology, independent biological replication,
and complete digital-twin status remain explicitly unsupported.

