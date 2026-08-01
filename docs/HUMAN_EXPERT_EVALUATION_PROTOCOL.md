# Human Expert Evaluation Protocol

## Scientific status

This protocol has **not been executed**. No participant, annotation, utility,
usability, time-saving, or decision-quality result is currently available. The
repository must not claim that MouseClaimBench improves human decisions until a
preregistered study has been approved, run, and analysed.

## Objective

The proposed study tests whether the MouseClaimBench report reduces unsupported
scientific-claim authorization relative to an unassisted evidence packet. It is
an evaluation of a decision-support intervention. It is not a test of neural
prediction, biological mechanism, or causal discovery.

## Reference standard

Each claim-evidence item is labelled independently by two eligible experts who
do not know which decision-support condition will later receive the item. A
third blinded expert adjudicates disagreements. The available labels are
`supportable`, `unsupported`, `insufficient_evidence`, and `out_of_scope`.
Raw agreement and nominal Krippendorff alpha are reported before adjudication.
Adjudicated labels are the study reference, not absolute scientific truth.

## Experimental design

The pilot uses a randomized within-participant crossover. Each participant sees
distinct item blocks under two conditions: an unassisted evidence packet and the
same packet format with a MouseClaimBench report. A Latin-square schedule
counterbalances condition order and item block. Items are not repeated across
conditions for the same participant, which limits recognition and carry-over.
Participants remain blind to reference labels and adjudicators remain blind to
condition assignments.

The configured pilot target is 24 eligible participants. This is a feasibility
target, not a post hoc claim of statistical power. A confirmatory sample size
must be preregistered after blinded pilot variance estimates are available.
Recruitment requires the applicable institutional ethics decision and informed
consent before any data are collected.

## Outcomes and analysis

The primary outcome is unsupported authorization at the claim-item level. The
prespecified analysis is a mixed-effects logistic regression with condition,
claim type, and period as fixed effects, and participant and item as crossed
random effects. This structure is necessary because decisions from the same
participant and decisions on the same item are not independent.

Secondary outcomes are supportable-claim rejection, decision time, calibrated
confidence, abstention, and actionability. Effect sizes and confidence intervals
must accompany every outcome. Exclusions, missingness, convergence failures,
and deviations from the preregistration must be reported. A paired item-level
or participant-level sensitivity analysis may complement the primary model but
must not replace the crossed analysis silently.

## Item construction

The item builder reads the non-authoritative deterministic candidates from
`results/llm_claim_extraction_audit/summary.json`. It selects up to eight items
per claim type, for a current target of 50 items, with a recorded random seed.
The generated package contains empty
annotation fields only. Running the builder cannot create gold labels or human
results.

```bash
python -m mousebrainbench.benchmarks.human_evaluation_protocol
```

The generated package is a preparation artifact. It may be frozen only after
the source manuscript and evidence packets are stable.
