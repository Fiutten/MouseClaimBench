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
| IBL behavior v5.3 | 35 risk-lock + 35 final mice | Zero false mouse events in both splits, risk UCB 0.0820, coverage and recovery LCB 0.9180. | Positive transport for one behavioral alignment task, not causal-neural or laboratory-independent validity. |
| Orthogonal shifts v5.4 | 13 levels, 100 bundles per level | Certification failed at Student-t 2 and eight variables. Lag and confounding contrasts remained certified. | Warning behavior is family-dependent and is not a calibrated universal alarm. |
| CausalRivers transport | 3 dependent geographic clusters, 320 pair evaluations | False fractions among authorizations ranged from 0.1944 to 0.4595. Exact external inference was prohibited. | Real transport is negative/descriptive, not external validation. |
| Profile internal AI audit | 29 items, 0 external raters | 1 critical veto, 11 major revisions, 8 minor revisions, and 9 retained items. | Internal revision is required; external content validation remains pending. |

The original six-level sweep used familywise 95% simultaneous confidence through
a Bonferroni per-level confidence of 0.991667. Version 5.4 expands this evidence
to 13 evaluable levels and uses per-level confidence 0.996154. Warning behavior
varies across families. These results support boundary characterization, not a
stable sensitivity, specificity, or calibrated universal alarm.

## Scientific judgement

The v5 work materially improves the evidence over v4. It now provides a positive
hierarchy-valid synthetic confirmation, a positive task-bounded IBL transport,
direct same-endpoint comparators, and heterogeneous degradation boundaries. The
negative CausalRivers transport and the two v5.4 certificate failures remain
visible. This is a credible methodological package rather than a uniformly
favorable benchmark narrative.

It is still not criticism-proof. The positive evidence is claim-specific. IBL
mice are genuine biological units, but they share a standardized consortium task
and do not establish laboratory-level replication. Two simple IBL comparators
also pass, so that result supports external transport rather than exclusive
algorithmic superiority. The knowledge profile has no independent content-validity
evidence, and four shift families do not calibrate a deployment alarm. No result
establishes mouse-brain causal validity.

Two previously open evidence blocks are now materially improved. IBL v5.3 supplies
two locked 35-mouse evaluations, and v5.4 evaluates four orthogonal shift families.
The remaining major external-validity gap is independent content review of the
knowledge profile. The IBL result is also task-specific and does not establish
laboratory-level replication. These limits prevent a broad truth-verification or
universal-deployment claim even though the second-paper package is now stronger.
The completed internal AI audit cannot close this gap and instead identifies
specific profile revisions that should precede external recruitment.
