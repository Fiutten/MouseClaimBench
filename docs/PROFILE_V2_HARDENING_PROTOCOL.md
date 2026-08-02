# Profile v2 hardening protocol

## Purpose and separation

Profile v1.1 and every result bound to it remain frozen. Profile v2 is a new
operational contract created after the internal adversarial audit. It cannot be
used to reinterpret v1 outcomes as prospective or confirmatory evidence.

The v2 correction has four objectives:

1. replace the ambiguous conclusion `supported` with `profile_authorized`
2. replace broad mechanistic, causal, and generic twin labels with bounded claims
3. reject passed facts whose minimum provenance schema is incomplete
4. preserve every simultaneous deficit instead of selecting one by priority

The profile is author-defined requirements engineering. It is not presented as
a consensus taxonomy and does not require a fabricated human panel. Independent
expert content validity remains absent and is not claimed.

## Formal boundary

Let `Q(c)` be the evidence blocks required by claim `c`. Each source fact has a
declared evidence state and a set of observations. A passed fact is admissible
only when it contains every provenance field required by its block
specification. Missing fields convert an apparent pass into
`requires_review`. The authorization rule is:

```text
profile_authorized(c) iff every effective state in Q(c) is passed
```

Every non-passed effective state is returned in the deficit set. There is no
priority among failed, unknown, not-applicable, and requires-review deficits.
They may coexist without masking one another. An undeclared claim returns
`outside_profile`.

Authorization is a result under one profile identifier, version, hash, context,
and evidence package. It is not a probability or declaration of scientific
truth.

## Claim corrections

The profile removes the generic `mechanistic` claim. Its bounded successor is
`directed_topology_consistent_prediction`, which additionally requires
uncertainty, shift, robustness, competing-alternative, context, and data-quality
blocks. This does not establish mechanism.

The generic `causal` claim becomes `intervention_supported_effect`. It binds the
design, estimand, intervention, outcome, identification assumptions, interval,
robustness, and context.

The generic `digital_twin` claim becomes
`complete_entity_specific_mouse_brain_digital_twin`. Partial models are not
rejected as models. They simply cannot satisfy complete-twin wording. The strict
claim adds biological fidelity, multiscale consistency, data quality, entity
updating, and synchronization requirements.

## Frozen mutation study

`configs/benchmarks/profile_v2_contract_mutation.yaml` fixes the benchmark
before outcome execution. It deterministically enumerates pristine packages,
every single evidence-state defect, every omitted block, every removed required
observation, and the full pairwise Cartesian product of mixed state defects.

Three transparent shortcut policies are evaluated. A fourth comparator mimics
a prioritized single-reason trace. The Python implementation is compared with
an independent Answer Set Programming executor on a deterministic stratified
subset.

The known outcome is contract conformance. It is not biological or scientific
ground truth. Success requires zero false authorizations, authorization of every
pristine package, exact recovery of every deficit set, exact Python-ASP
conformance, and at least one false authorization from a transparent shortcut.

## Change control

Any change to v2 claims, evidence schemas, or mutation endpoints requires a new
profile or protocol version. Frozen v1 artifacts must never be regenerated with
v2 semantics. V2 artifacts must record a clean source revision and the profile
hash.
