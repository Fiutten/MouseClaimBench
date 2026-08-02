# Semantic risk control v3: novelty audit

Audit date: 2 August 2026.

## Scope and evidentiary status

This is a targeted prior-art audit, not a registered systematic review. It
searched current primary papers and preprints for scientific claim verification,
abstention, risk control, claim-evidence contracts, claim admissibility,
neuro-symbolic ASP, and authorization under population shift. Absence from this
audit is not proof that no related work exists. The audit must be refreshed
immediately before manuscript submission.

## Components that are not novel by themselves

1. **Finite-sample risk control.** Risk-Controlling Prediction Sets and
   Learn-Then-Test already provide general distribution-free or finite-sample
   machinery for controlling user-defined risks. MouseClaimBench applies this
   machinery. It does not invent it.

2. **Abstention in scientific claim verification.** *Knowing When Not to Answer:
   Abstention-Aware Scientific Reasoning* decomposes scientific claims into
   critical conditions, audits them with NLI, applies non-compensatory aggregation,
   and evaluates risk-coverage behavior on SciFact and PubMedQA. It does not
   provide a distribution-free guarantee, but it directly overlaps our general
   motivation and abstention narrative.

3. **All-conditions claim admissibility.** *When Verification Fails: How
   Compositionally Infeasible Claims Escape Rejection* makes acceptance conditional
   on support for every asserted constraint and demonstrates salient-constraint
   shortcuts. Therefore, requiring all critical evidence blocks is not in itself
   a new verification principle.

4. **Claim-evidence contracts and repository control planes.** ResearchLoop
   represents claims, evidence, task contracts, ledgers, and paper bindings as
   durable repository state. The Code-First Peer Review protocol likewise maps
   claims to executable artifacts and hashes. Claim-evidence traceability cannot
   be claimed as unique to MouseClaimBench.

5. **ASP plus learned components.** NeurASP and related neuro-symbolic systems
   already integrate probabilistic model outputs with Answer Set Programming.
   Our second ASP implementation is valuable as an independent conformance oracle,
   not as a new neuro-symbolic formalism.

6. **Claim-specific admissibility.** Recent domain-specific work makes the
   scientific statement, rather than only the computation, the unit of assessment.
   The PCA-biplot admissibility framework is one explicit example.

7. **Authorization under target shift.** OPAL, published as a preprint on 30 July
   2026, treats authorization as a distinct layer for adaptive science, proves a
   source-to-target impossibility boundary, requires target calibration and
   non-trivial activation, and evaluates locked scientific populations. This is
   the strongest current threat to any broad claim that MouseClaimBench first
   combines scientific authorization, controlled risk, and scope restrictions.

## Defensible differential

The remaining contribution is the following combination:

1. A typed hierarchy of neurocomputational claim families separates prediction,
   reproducibility, topology, direction, mechanism, causality, structure-function
   association, and digital-twin language.
2. A hard, non-compensatory evidence contract is composed with claim-specific LTT
   thresholds and family-wise error allocation.
3. The statistical certificate carries an executable population contract over
   population family, independent unit, evidence protocol, and reference protocol.
   An out-of-scope certificate cannot authorize support.
4. Python and clingo independently execute the scientific disposition semantics,
   with a fixed equivalence audit rather than merely sharing one implementation.
5. The complete system is stress-tested on structural-equation truth, physical
   causal systems, Perturb-seq interventions, and mouse electrophysiology. Negative
   transport is retained as a result instead of being hidden by abstention.

No work identified in this audit combines all five elements in a computational
mouse-brain claim-authority setting. This is a defensible literature gap, not proof
of absolute uniqueness.

## Novelty verdict

The generic claim "a knowledge system verifies scientific claims using evidence
contracts and abstention" is not novel enough in the current 2026 context. The
generic claim "risk-controlled scientific authorization under population shift"
is also no longer safe after OPAL.

The narrower claim remains plausible: MouseClaimBench operationalizes typed,
population-scoped, finite-sample authorization for neurocomputational model claims,
with independent symbolic conformance and explicit non-transport evidence.

This is methodologically substantial but empirically incomplete. It is a credible
Q1 candidate only if the paper foregrounds the typed scope contract and the
separation between semantic admissibility and statistical certification. It is
not yet a low-risk Q1 submission because no external domain has both valid
domain-specific calibration and non-zero authorized coverage on an untouched
population.

## Priority actions before a second-paper submission

1. **Target-calibrated positive external block.** Calibrate on exchangeable units
   from one real domain and evaluate once on a new locked population from that
   same domain. Require non-zero minimum coverage in advance. This is the highest
   priority.
2. **Non-degeneracy in the certificate.** Add a prespecified lower bound on
   coverage or recovered positive claims. An abstain-all policy must be valid but
   cannot satisfy the complete contribution claim.
3. **Direct competitor baselines.** Compare with confidence-only scientific
   abstention, hard contract only, unconstrained LTT, and the complete scoped
   policy on identical cases. Where code and data permit, reproduce the 2026
   abstention-aware framework rather than comparing only by prose.
4. **Prospective router confirmation.** Freeze the association-aware router and
   test it on new structural-equation regimes. The current 0.9422 diagnostic
   accuracy is post hoc.
5. **Profile validity without overclaiming consensus.** Either ground each
   requirement in established reporting and causal-inference standards or retain
   the profile explicitly as an author-proposed operational policy. Structural
   conformance cannot substitute for scientific content validity.
6. **Scope-shift diagnostics.** Add statistical shift diagnostics as warnings,
   while preserving the rule that diagnostics cannot by themselves validate a
   certificate in another population.
7. **Prospective power and sample-size analysis.** Determine how many independent
   units are needed to certify the target SFAR and minimum coverage before opening
   a new external test set.
8. **Efficiency and failure taxonomy.** Report runtime, memory, calibration cost,
   abstention cost, and failures partitioned into semantic inadmissibility,
   non-certifiability, score misalignment, and population mismatch.

## Primary sources reviewed

- Bates et al., *Distribution-Free, Risk-Controlling Prediction Sets*:
  https://arxiv.org/abs/2101.02703
- Angelopoulos et al., *Learn then Test: Calibrating Predictive Algorithms to
  Achieve Risk Control*: https://arxiv.org/abs/2110.01052
- Abdaljalil et al., *Knowing When Not to Answer: Abstention-Aware Scientific
  Reasoning*: https://arxiv.org/abs/2602.14189
- Liu et al., *When Verification Fails: How Compositionally Infeasible Claims
  Escape Rejection*: https://arxiv.org/abs/2604.10990
- Xia and Wang, *ResearchLoop: An Evidence-Gated Control Plane for AI-Assisted
  Research*: https://arxiv.org/abs/2605.28282
- Chen, *Review the Code, Not the Story: A Vision and Protocol for Code-First
  Peer Review*: https://arxiv.org/abs/2606.07683
- Yang et al., *NeurASP: Embracing Neural Networks into Answer Set Programming*:
  https://arxiv.org/abs/2307.07700
- Pinto and Dias, *Claim-Specific Admissibility of PCA Biplot Interpretations*:
  https://arxiv.org/abs/2607.16469
- Bi et al., *Certifying when decision-time information justifies adaptive
  experimentation*: https://arxiv.org/abs/2607.27651
- Wang et al., *SciVer: Evaluating Foundation Models for Multimodal Scientific
  Claim Verification*: https://aclanthology.org/2025.acl-long.420/
