# Hybrid selective validation protocol v2

## Motivation

The first prospective experiment rejected the claim that the existing hard
contract is generally superior. Its dominant error source was the isolated
direction block. Version 2 does not alter or reinterpret that result. It asks a
new and narrower question: can a learned selective policy use contextual
evidence while retaining semantic boundaries that a probabilistic model is not
allowed to override?

The machine-readable protocol is
`configs/benchmarks/hybrid_selective_claim_validation_v2.yaml`. The first Git
commit containing that file and this document is the temporal anchor. None of
the ten confirmatory generators may be executed before that commit.

## Directional evidence

The previous polynomial residual heuristic is retained only as an input feature
and historical comparator. The new directional block uses the Additive Noise
Model implementation in `causal-learn` version 0.1.4.8. It compares residual
independence in both orientations. To keep computation fixed and bounded, every
case supplies at most 200 observations selected by a deterministic seed.

A forward margin of at least 0.10 produces `passed`, a reverse margin of at
least 0.10 produces `failed`, and a smaller absolute margin produces
`requires_review`. The margin was selected after inspecting the already-used
Tuebingen cause-effect data. Therefore, Tuebingen is development evidence and
must not be described as independent confirmation in version 2.

ANM direction is identifiable only under its functional-model assumptions. In
particular, hidden confounding, selection, measurement error, and post-nonlinear
mechanisms can invalidate its interpretation. Several confirmatory generators
violate these assumptions deliberately. The correct behavior in such cases is
often abstention rather than forced direction.

## Constrained selective policy

One regularized logistic model is fitted for each claim that has both reference
classes in consumed development data. The inputs include all evidence states,
continuous diagnostic values, ANM outputs, sample size, and known simulation
noise. Isotonic calibration uses a disjoint deterministic calibration split.
The final operating threshold maximizes calibration coverage subject to a
one-sided 95% Clopper--Pearson upper bound of 0.10 on selective error.

The policy emits `supported`, `blocked`, or `abstained`. High calibrated
probability supports a claim, low probability blocks it, and intermediate
probability abstains. A semantic support veto runs before authorization. It
prevents, for example, causal support without a passed intervention block and
digital-twin support without every entity, coverage, intervention,
reproducibility, and operational requirement. No learned probability can
override a veto.

This architecture does not guarantee lower raw error than an unconstrained
classifier. Its claim is instead testable at a declared risk--coverage operating
point: non-trivial coverage, bounded selective error, bounded false
authorization, and zero semantic violations.

## Development and confirmation

The five original SEM regimes and all nine version-1 prospective regimes are
now consumed development data. Their deterministic seeds are divided by modulo
five into model fitting, threshold calibration, and a locked development audit.
No case crosses these roles.

Confirmation uses ten new generators covering mixtures, threshold confounding,
reverse piecewise effects, exponential and piecewise mechanisms,
post-nonlinearity, non-Gaussian linear effects, measurement error, collider
truncation, and intervention. Three sample sizes, three noise levels, and 40
seeds produce 3,600 cases and 36,000 claim decisions per policy.

The seven primary conditions in the YAML file are conjunctive. Failure of one
condition rejects the primary endpoint. Thresholds, features, semantic gates,
generator truth labels, and success criteria cannot be changed after the first
confirmatory artifact is generated. A correction to code that contradicts this
text requires an amendment commit. A scientific redesign requires version 3
and another unused evaluation partition.

## Interpretation boundary

Positive results would validate a constrained computational decision policy
under the declared generators. They would not prove causal direction in mouse
brain data, establish expert consensus, demonstrate improved human decisions,
or validate a complete digital twin. Those claims remain prohibited regardless
of predictive performance.

