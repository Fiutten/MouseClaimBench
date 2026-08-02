# Semantic risk control v3: technical report

## Scientific thesis

The v3 contribution is a claim-authorization method, not a new neural simulator,
causal-discovery algorithm, or predictive architecture. Its central question is
whether a computational result may authorize a specified scientific claim under
explicit evidence semantics and a population-bounded finite-sample risk target.
The method separates four objects that conventional predictive evaluation can
collapse: score quality, semantic admissibility, risk certification, and the
population to which that certificate applies.

This separation is the proposed novelty. MAPIE, Learn-Then-Test (LTT),
DirectLiNGAM, additive-noise models, clingo, Causal Chambers, CausalBench, and
the International Brain Laboratory (IBL) data are established resources. None
is presented as an original algorithm. The contribution is their integration
into an executable, non-compensatory authorization protocol with independently
checked rule semantics and explicit guarantee scope.

## Formal decision rule

For claim family `c`, let `S_c(X)` be the frozen support score, `G_c(X)` the
semantic admissibility gate, and `t_c` the LTT threshold. Authorization is

`A_c(X) = 1[G_c(X) = 1] 1[S_c(X) >= t_c]`.

The Semantic False Authorization Risk (SFAR) is

`SFAR_c = P(Y_c = 0 | A_c(X) = 1)`.

The target is `SFAR_c <= 0.05` at 95% family-wise confidence. Confidence is
allocated by Bonferroni across the six variable claim families. The complete
case is the independent calibration unit. The gate is applied before risk
calibration, so a threshold cannot compensate for missing required evidence.

The certificate is valid only for its declared population, independent unit,
evidence protocol, and reference protocol. Let `P_cal` and `P_target` denote the
corresponding scope contracts. A certificate can govern a target only when the
contracts match and no relevant distribution shift has been declared. Otherwise
the scoped authorization is forced to abstention. This is a protocol guard, not
a statistical shift detector.

## Assumption-aware direction routing

The router chooses among established estimators using declared assumptions.
Controlled intervention evidence has priority. DirectLiNGAM is eligible only in
linear, non-Gaussian, acyclic settings without unresolved hidden confounding or
selection bias. The additive-noise method is eligible only in continuous,
acyclic, additive-noise settings under the same exclusions. Material measurement
error, missing assumptions, numerical failure, hidden confounding, and selection
bias force abstention.

The frozen v3 run exposed one weakness. The router attempted all 450 independent
heavy-tailed cases because additive-noise compatibility alone did not establish
an association to orient. A post-confirmation repair adds an explicit association
precondition for observational routing. Applied diagnostically to the archived
cases, it reduces routing coverage from 0.5000 to 0.4375, raises attempted
accuracy from 0.8244 to 0.9422, and removes all 450 spurious attempts. This is
post hoc. It requires a new frozen protocol and fresh data before it can be
reported as confirmatory evidence.

## Independent knowledge semantics

The authorizer has two execution paths. The main Python engine evaluates the
versioned evidence profile. A separate Answer Set Programming encoding is
executed by clingo. Equivalence was observed on 2,847 audited cases with zero
mismatches. The audit includes exhaustive assignments for each small claim,
all 625 assignments for the four-block mechanistic claim, deterministic boundary
cases for the ten-block digital-twin claim, and 1,000 fixed randomized digital-
twin assignments.

This result establishes implementation conformance over the audited space. It
does not validate the scientific correctness or consensus status of the
underlying author-proposed evidence taxonomy.

## Evaluation blocks

The frozen synthetic confirmation contains 7,200 complete cases from sixteen
structural-equation regimes. The semantic LTT policy authorizes 24,223 of 43,200
variable claim decisions, giving 0.5607 coverage and an empirical SFAR of
0.00314. It produces zero semantic support violations. A naive 0.5 threshold
produces 4,701 semantic violations and an SFAR of 0.0492. Unconstrained LTT
produces 3,196 semantic violations and an SFAR of 0.0120. These ablations show
that statistical calibration and evidence semantics perform different roles.

Causal Chambers contributes 170 locked physical-system pair cases. Raw transfer
coverage is 0.2422 and SFAR is 0.0526, with zero semantic violations. CausalBench
uses 200 genes, 39,800 directed pairs per domain, K562 as context, and RPE1 as a
locked transport population. RPE1 raw transfer coverage is 0.00970 and SFAR is
0.2474, again with zero semantic violations. The IBL block uses 24 insertions
from 24 mice, split into 17 context and seven locked animals. The locked set
authorizes no claims. It is a safe-abstention boundary, not positive predictive
validation.

The synthetic certificate is out of scope for all three real-data blocks.
Enforcing the scope contract therefore preserves the 24,223 in-scope synthetic
authorizations and sets external scoped authorizations to zero. Raw external
metrics remain diagnostic and are not erased. Their failures are evidence that
the finite-sample guarantee does not transport automatically.

## Sensitivity and ablation

Calibration-size sensitivity uses 20 deterministic subsamples at 250, 500,
1,000, and 2,000 cases plus the complete 3,600-case development set. All 81
policies stay below the 0.05 empirical synthetic SFAR target and produce no
semantic violations. However, mean certified families rise from 2.0 at 250
cases to 6.0 at 3,600 cases. Mean fresh coverage rises from 0.3034 to 0.5607.
Finite-sample certification therefore has a visible data requirement.

This analysis was performed after inspecting v3 outcomes. It is exploratory,
cannot be used to select a preferred calibration size, and cannot replace the
frozen primary confirmation.

## Valid conclusions

The current evidence supports a bounded methodological claim: the proposed
system combines non-compensatory evidence semantics with claim-specific LTT
certificates, independently checks the knowledge semantics, prevents certificate
use outside a declared population, and exposes rather than hides external
transport failure.

The evidence does not establish positive cross-domain generalization, universal
finite-sample risk control, a universally reliable direction method, biological
causality, scientific truth verification, or any form of complete mouse-brain
digital twin. The release audit therefore returns
`methodological_core_validated_external_generalization_not_established`.

## Publication assessment

This is a plausible Q1 methodological candidate only under a narrow framing:
risk-controlled authorization of scientific claims by a knowledge-based system
with typed population scope. It is stronger than the previous version because
it adds a formal statistical certificate, an independent rule backend, fresh
synthetic confirmation, real external stress tests, an executable scope guard,
and explicit negative transport evidence.

It is not immune to criticism. The strongest remaining weakness is the absence
of an external domain with both valid domain-specific calibration and non-zero
authorized coverage on a new untouched population. A stronger empirical claim
requires that evidence plus prospective confirmation of the repaired router.
Without it, the paper must present non-transport and safe abstention as central
findings rather than implying broad external effectiveness.

The current literature also prevents broad priority claims. Scientific
abstention, non-compensatory condition checking, claim-evidence contracts,
neuro-symbolic ASP, and finite-sample risk control all have direct precedents.
Recent work on authorization for adaptive science further combines target-risk
control, source-to-target limitations, locked populations, and non-degenerate
activation. The defensible novelty is therefore the complete typed and
population-scoped composition for neurocomputational claims, not any isolated
component. The dated comparison and required experiments are recorded in
`docs/SEMANTIC_RISK_CONTROL_V3_NOVELTY_AUDIT.md`.

## Reproduction

Create the exact environment with:

```bash
bash scripts/setup_semantic_risk_v3_env.sh .venv-risk-v3
```

Verify the committed package without modifying frozen artifacts:

```bash
ENV_PATH=.venv-risk-v3 bash scripts/reproduce_semantic_risk_v3.sh verify
```

Regenerate the independent semantics audit and synthetic confirmation under
`outputs/` with `core`. Use `external` only after placing the official source
files at the paths recorded in the committed CausalBench and IBL selection
manifests. External mode verifies all hashes before computation.
