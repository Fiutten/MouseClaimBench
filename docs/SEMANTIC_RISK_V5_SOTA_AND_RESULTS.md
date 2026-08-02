# Semantic-risk v5: evidence, novelty, and limits

## Scientific question and frozen design

Version 5 asks a narrower question than whether MouseClaimBench discovers causal
truth. It asks whether a non-compensatory evidence-to-claim authorizer can control
the probability that an independent experiment bundle contains at least one false
authorization while retaining non-zero experiment coverage and positive recovery.
The inferential unit is the seed bundle. Scenarios, pairs, and claims nested inside
that bundle are descriptive units and are never counted as independent evidence.

The global protocol, its topology-specific follow-up, the real-data transport, and
the severity sweep were frozen before their respective outcome blocks. Exact
one-sided Clopper-Pearson bounds are used only where at least 29 independent
top-level bundles exist. CausalRivers has only three geographic top-level clusters,
so no exact external certificate is reported there.

## State-of-the-art boundary checked on 2026-08-02

The surrounding field is active and leaves little room for broad framework claims.
Learn-Then-Test and risk-controlling prediction sets already provide finite-sample
risk calibration. TimeGraph already supplies graph-structured time-series
generators. CausalRivers already provides a large in-the-wild benchmark with
reference river-network graphs and a flood shift. Brouillard et al. show that
causal-discovery evaluation remains dominated by unrealistic synthetic settings,
weak metrics, and too few real applications.

Causal-Nest is the closest current Knowledge-Based Systems competitor. It combines
12 discovery methods, graph-integrity metrics, causal estimation and refutation,
and causal feature engineering. Consequently, MouseClaimBench cannot claim novelty
as a general causal workflow, integrity framework, benchmark collection, or causal
discovery system. Its defensible distinction is narrower: an executable
evidence-to-claim contract whose vetoes are non-compensatory, whose false
authorization endpoint is evaluated at the independent experiment hierarchy, and
whose deployment boundary includes activation requirements and abstention under
shift. That combination appears differentiated, but priority has not been proved
by a systematic review and publication quality still depends on external evidence.

Primary resources:

- TimeGraph, KDD 2025: https://doi.org/10.1145/3711896.3737439
- CausalRivers, ICLR 2025: https://proceedings.iclr.cc/paper_files/paper/2025/hash/a205fda871b0f6c1e18a7ad7325eb6cf-Abstract-Conference.html
- Landscape of Causal Discovery Data, CLeaR 2025: https://proceedings.mlr.press/v275/brouillard25a.html
- Causal-Nest, Knowledge-Based Systems 2026: https://doi.org/10.1016/j.knosys.2026.116005

## Results

| Evidence block | Independent units | Result | Valid interpretation |
|---|---:|---|---|
| TimeGraph v5, six claim families | 50 risk-lock bundles | No policy met the complete contract. Final block stayed closed. | The declared global population is unsupported. |
| TimeGraph v5.1, topology risk lock | 100 bundles | 1 failure, risk UCB 0.0466, coverage LCB 0.7560, recovery LCB 0.7741. | The fixed topology contract passes in the revised synthetic population. |
| TimeGraph v5.1, untouched final | 100 bundles | 1 failure, risk UCB 0.0466, coverage LCB 0.7119, recovery LCB 0.7555. | Internal confirmation of the same bounded contract. |
| Same-endpoint baselines | 100 + 100 bundles | Abstention, confidence 0.5, contract-only, and the v3 threshold all fail the complete endpoint. | The result is not reproduced by these simple comparators. |
| TimeGraph v5.1 OOD | 100 bundles | 6 failures, risk UCB 0.1150, certificate failed, shift warned. | The certificate does not transport to the declared OOD block. |
| Frozen severity sweep v5.2 | 6 levels, 100 paired bundles per level | First warning at shift 1. First simultaneous certificate loss at shift 3. | Warning preceded loss on this path only. It is not a universal detector. |
| CausalRivers transport | 3 dependent geographic clusters, 320 pair evaluations | False fractions among authorizations ranged from 0.1944 to 0.4595. Exact external inference was prohibited. | Real transport is negative/descriptive, not external validation. |
| Profile content review | 29 items, 0 external raters | Review packet complete, ratings pending. | The profile remains author-proposed. |

The six-level sweep used familywise 95% simultaneous confidence through a
Bonferroni per-level confidence of 0.991667. The warning appeared two severity
levels before loss, but only one frozen path was evaluated. The result therefore
supports boundary characterization, not sensitivity, specificity, or calibrated
alarm performance.

## Scientific judgement

The v5 work materially improves the evidence over v4. It provides a positive
hierarchy-valid synthetic confirmation, direct same-endpoint failures of four
simple comparators, a prospective degradation boundary, and a real benchmark that
exposes poor transport rather than concealing it. This is a credible methodological
core and a useful negative external result.

It is not yet a criticism-resistant strong Q1 package. The positive result is
claim-specific and synthetic. CausalRivers does not contain enough independent
top-level systems for external risk certification and shows high descriptive false
authorization fractions. The knowledge profile has no independent content-validity
evidence. The shift diagnostic has one prospective path but no calibrated operating
characteristics across heterogeneous shifts. No result establishes mouse-brain
causal validity.

The next evidence should therefore be acquired rather than tuned. The profile
packet requires at least seven eligible independent raters across the frozen
expertise strata. External risk requires a new untouched population with at least
29 genuinely independent top-level units, not additional pairs from the same river
network or mouse. A second prospective shift family is needed to estimate warning
errors rather than narrate one successful ordering. Until those blocks exist, the
release gate must remain closed for a strong Q1 claim.
