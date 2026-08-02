# Semantic risk v4: state of the art and scientific outcome

## Literature boundary checked on 2026-08-02

The isolated ingredients of MouseClaimBench are not new. Risk-controlling
prediction sets provide hold-out calibrated finite-sample loss control
(Bates et al., 2021, https://arxiv.org/abs/2101.02703). Learn-Then-Test reframes
model calibration as multiple testing and supports explicit finite-sample risk
guarantees without refitting the predictor (Angelopoulos et al., 2022,
https://arxiv.org/abs/2110.01052).

Scientific abstention and compositional verification are also occupied areas.
Abdaljalil et al. decompose scientific claims into minimal conditions and show
that confidence abstention reduces error at moderate coverage
(https://arxiv.org/abs/2602.14189). Liu et al. study conjunctions of asserted
constraints under a closed-world acceptance rule
(https://arxiv.org/abs/2604.10990). These works strongly overlap with claim
decomposition, non-compensatory conditions, and abstention. They do not, however,
establish the same experiment-level non-degenerate certificate implemented here.

The closest and most important threat is OPAL
(https://arxiv.org/abs/2607.27651). OPAL separates authorization from downstream
decision making, uses a precommitted contract, requires non-trivial activation,
uses target calibration, and reports positive executed value after cost. Those
ideas cannot be claimed as MouseClaimBench novelties. OPAL also has a stronger
positive external result than the present v4 study. Our remaining distinction is
the narrower combination of executable domain-knowledge contracts for scientific
claim families, non-compensatory semantic gates, exact experiment-level
activation/risk requirements, assumption-aware evidence routing, and an explicit
taxonomy of why a claim is not authorized.

The Causal Chambers source is an appropriate physical sanity check because it
provides open real-system experiments and validated ground-truth graphs. Its own
authors warn that success on these controlled systems need not transfer to larger
systems (Gamella et al., 2025, doi:10.1038/s42256-024-00964-x). Therefore, even a
positive chamber result would not establish mouse-brain biological validity.

## Eight-point outcome

1. **Target-calibrated external block.** Implemented on twelve previously unused
   non-image Causal Chambers datasets. The v4 global policy failed because only
   total abstention met risk. A locked v4.1 family policy then used 117 consumed
   experiments for development and 33 untouched usable experiments for final
   evaluation. It failed final confirmation.

2. **Non-degeneracy.** Exact one-sided Clopper-Pearson bounds jointly enforce an
   experiment-failure UCB of at most 0.10, authorized-experiment coverage LCB of
   at least 0.10, and positive-recovery LCB of at least 0.05. Universal abstention
   is formally rejected. This part works as designed.

3. **Direct baselines.** Seven policies are evaluated under the same endpoint:
   abstain-all, fixed confidence, evidence contract only, unconstrained LTT,
   semantic LTT without activation floor, confidence-only target calibration,
   and complete semantic non-degenerate LTT. None passed the final external
   certificate. This rules out an easy baseline victory but does not create a
   positive method result.

4. **Prospective router.** The original v4 screen reduced spurious attempts from
   540 to 3 but failed its zero-error criterion. A frozen v4.2 repair with new
   equations and seeds required both Pearson and Spearman association tests at
   familywise alpha 0.001. Across 2,160 cases it produced zero spurious attempts,
   0.929 attempted-direction accuracy, and 1.000 valid-route coverage. This is a
   genuine positive internal result, not causal proof.

5. **Profile validity.** Every relation now records rationale, scope, exceptions,
   rejected alternatives, and sources. ARRIVE 2.0, NIH rigor, FAIR, and NIST AI
   RMF are mapped only to their legitimate reporting or assurance roles. The
   evidence-to-claim profile remains author-proposed and lacks independent
   content validation.

6. **Prospective power.** At a 0.10 experiment-failure target and 0.95 confidence,
   at least 29 zero-failure independent units are required. The final block had
   33 usable experiments but two failures, yielding risk UCB 0.179 despite strong
   coverage and recovery.

7. **Shift diagnostics.** Holm-corrected marginal KS tests and a multivariate
   energy permutation test issue warnings only. Shift was detected in the final
   block. The warning cannot authorize, restore, or extend a certificate.

8. **Cost and failure analysis.** Artifacts include wall time, peak RSS, source
   bytes, throughput, abstention rate, missed eligible true claims, and a
   mutually exclusive failure taxonomy. These measures improve auditability but
   do not compensate for failed external risk control.

## Final scientific judgement

The engineering and methodological program is complete, and the router repair is
prospectively supported. The stronger scientific thesis is not supported. The
external authorizer failed its locked risk certificate, the profile lacks
independent content validation, and experiments nested within datasets leave a
dependence concern beyond the conditional binomial analysis.

Accordingly, this is not yet a strong Q1 second-paper package. A defensible paper
could report the framework and negative external result, but it would be a
bounded methodological contribution. A stronger Q1 claim requires a new untouched
target population, dependence-aware calibration at the dataset/experiment
hierarchy, and independent validation of profile content. The consumed final
Causal Chambers block must not be used for another confirmatory adjustment.

