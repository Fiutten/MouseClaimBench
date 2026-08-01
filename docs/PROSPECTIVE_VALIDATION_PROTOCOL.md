# Prospective computational validation protocol

## Purpose and temporal separation

This document freezes the second-stage computational validation before its
outcomes are generated. The machine-readable protocol is
`configs/benchmarks/prospective_claim_validation_v1.yaml`. The Git commit that
first contains both files is the temporal anchor. Results created before that
commit or after an unversioned change to the protocol are not prospective
results.

The experiment tests whether the non-compensatory claim contract reduces false
authorization on previously unused structural-equation regimes. It also tests
whether the fixed local MICRONS association survives inference that treats
directed pairs sharing a neuron as dependent. These are two distinct questions.
Success in one cannot compensate for failure in the other.

## Locked synthetic evaluation

The five regimes in the existing oracle benchmark form the development
partition. They may be used to fit the probabilistic comparator, but they may
not be reported as prospective evidence. Nine new regimes form the locked
evaluation partition. They include heavy-tailed independence, nonlinear
confounding, reverse direction, saturation, heteroscedasticity, weak effects,
distribution shift, collider selection, and a nonlinear interventional case.

Each regime is evaluated at three sample sizes and three noise levels with 40
deterministic seeds per cell. This produces 3,240 independent generated cases
and 32,400 dependent claim decisions per policy. Reference claims are declared
from each data-generating graph and intervention availability. They are never
derived from the diagnostics supplied to a policy.

The primary endpoint is aggregate false-positive rate. A successful result
requires a lower rate for the evidence contract than for every comparator. The
contract may not obtain this result by becoming indiscriminately conservative:
its false-negative rate may be no more than 0.10 above the best comparator.
Case-level paired errors must also favor the contract under a two-sided exact
sign test. Uncertainty intervals resample complete cases within every regime,
sample-size, and noise cell.

The probabilistic comparator is fitted only on the five development regimes.
Its features, regularization, threshold, and treatment of constant-label claims
are fixed in the YAML protocol. Refitting, threshold selection, or feature
selection using the prospective cases invalidates the primary comparison.

## Locked MICRONS network inference

The biological endpoint remains `all_pairs/readout_location`. No alternative
metric or stratum may replace it after inspection. The analysis uses discovery
and two non-overlapping hold-out windows. A linear coefficient for connected
pairs is adjusted using distance, squared distance, pre- and post-synaptic
degree, and coarse cell-type agreement.

Ordinary pair-level uncertainty is not acceptable because many directed pairs
share neurons. The primary standard error is therefore clustered over incident
units for directed dyads. A Freedman--Lane node-label permutation provides a
second network-preserving test. The endpoint passes only when both methods are
positive at the frozen significance level in every cohort. A positive result
remains a local observational structure--function association. It is not a
causal effect or independent biological replication.

## Interpretation boundary

This protocol can strengthen computational construct validity, prospective
error-rate evidence, comparator credibility, and network-aware uncertainty. It
cannot manufacture independent expert content validity or a human evaluation.
It does not test whether people make better decisions with the system. It also
does not support a complete digital-twin claim, whole-brain coverage, or
cross-domain validity. Those limitations remain reportable even if every
computational endpoint passes.

## Amendment policy

Implementation corrections are permitted only when a test demonstrates that
the code does not implement the written protocol. Every correction must receive
a new commit and an amendment record. Scientific rules, regimes, thresholds,
features, controls, and success criteria may not be changed in version 1.0.0
after prospective outcomes are generated. A scientifically motivated change
requires a new protocol version and a new, non-overlapping evaluation partition.

