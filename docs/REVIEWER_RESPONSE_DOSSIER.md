# Reviewer Response Dossier

## Purpose

This dossier prepares the scientific response strategy for the submitted
MouseBrainBench manuscript. It is not manuscript text. It is an internal control
document that maps each central claim to evidence, limits, likely reviewer
objections, and defensible responses.

The current source of truth for the publication state is:

- `results/publication_freeze/summary.json`;
- `results/microns_q1_package/summary.json`;
- `docs/DATASET_VOLUME_AUDIT.md`;
- `docs/MECHANISTIC_IDENTIFIABILITY_SCORE.md`;
- `docs/MIS2_CALIBRATION_PROTOCOL.md`;
- `results/digital_twin_claim_audit/summary.json`.

Older phase documents are historical unless they are explicitly referenced by
the freeze artifact.

## Submission-Level Position

MouseBrainBench should be defended as a claim-aware validation and benchmarking
framework for partial mouse-brain digital models. It should not be defended as a
new full-brain simulator, a complete digital mouse brain, a causal circuit model,
or a Sensorium state-of-the-art predictor.

The strongest current position is:

> MouseBrainBench separates prediction, reproducibility, topology specificity,
> directed identifiability, local structure-function association, and cost. This
> separation prevents predictive or reproducible models from being
> overinterpreted as mechanistically identifiable digital twins.

The strongest empirical anchor is the bounded MICRONS structure-function package:

- primary endpoint: `all_pairs/readout_location`;
- endpoint status: fixed after discovery and evaluated in two non-overlapping
  hold-outs;
- discovery plus two hold-outs;
- 2991 total units;
- 6575 total synapses;
- 5943 total unique directed connected pairs;
- unit-cluster weighted bootstrap over the directed pair frame;
- positive local observational association under distance- and degree-matched
  controls for the primary endpoint.

## Claim Matrix

| Claim | Status | Evidence | Boundary |
|---|---|---|---|
| MouseBrainBench is a claim-aware benchmark framework. | Supported | Executable gates, artifacts, tests, freeze summary. | It is not a full simulator or new biological engine. |
| Allen VBN is a negative mechanistic-identifiability case. | Supported | Reproducibility passes, topology and directed identifiability fail. | Does not invalidate Allen data or Allen connectivity. |
| Sensorium/Dynamic Sensorium are predictive/interoperability cases. | Supported | Local adapters, model comparators, official stack audit. | No SOTA or mechanistic claim. |
| MICRONS provides local observational structure-function evidence. | Supported within limits | Q1 package, two non-overlapping hold-outs, distance/degree controls, unit-cluster bootstrap. | Not causal, not whole-brain, not independent-animal replication. |
| MIS makes claim evaluation non-compensatory. | Supported as operational framework | Synthetic benchmark and real cases. | Not yet a universal external standard. |
| MouseBrainBench is a mouse-brain digital twin. | Blocked | Digital-twin claim audit blocks it. | Do not defend. |
| The MICRONS effect is a causal mechanism. | Blocked | Observational data and matched controls only. | Do not defend. |
| The official Sensorium baseline is Q1-qualified/SOTA. | Blocked | Official stack runs, but local qualified performance is not established. | Use as bounded internal control only. |

## Likely Reviewer Objections and Response Strategy

### Objection 1: The work is not a new brain simulator.

Response:

This is correct and intentional. The manuscript does not claim to replace BMTK,
SONATA, The Virtual Brain, Allen models, or MICRONS resources. The contribution
is the validation layer that decides which claims a model-target pair can
support. The manuscript should emphasize that new simulation engines are not the
missing component in this setting. The missing component is a reproducible,
non-compensatory audit of prediction, reproducibility, topology, direction,
structure-function association, and cost.

Do not answer by exaggerating the scope. Answer by making the engineering
problem precise.

### Objection 2: The MICRONS analysis is observational and cannot support
causality.

Response:

Agree. The claim is local observational association, not causality. The primary
endpoint is `all_pairs/readout_location`, fixed after discovery and evaluated in
two non-overlapping hold-outs. Distance- and degree-matched controls reduce
obvious spatial and degree confounds, and the unit-cluster bootstrap addresses
stability under shared units. These analyses do not identify an interventional
mechanism. The manuscript explicitly blocks causal, behavioral, whole-brain, and
complete digital-twin claims.

If pressed, the correct future-work answer is interventional validation or
independent cross-resource replication, not stronger wording.

### Objection 3: The MICRONS hold-outs are not independent biological
replications.

Response:

Correct. They are non-overlapping hold-out windows from the same resource. The
manuscript should use `internally reproduced`, not `replicated`, unless
replication is clearly qualified as internal. The evidence supports stability
inside one public MICRONS resource. It does not establish cross-animal,
cross-laboratory, or cross-dataset replication.

### Objection 4: MIS is threshold-dependent.

Response:

Any gate is threshold-dependent, but the relevant design choice is
non-compensation. Prediction cannot compensate for absent topology specificity.
Reproducibility cannot compensate for absent directionality. The sensitivity
audit exists to test whether the main conclusions collapse under reasonable
threshold perturbations. Current sensitivity results support the methodological
claim but do not turn MIS into a universal standard.

The right concession is that MIS requires further calibration across more
datasets before being treated as a general benchmark standard.

### Objection 5: Allen VBN is a negative result. Why include it?

Response:

Because it demonstrates the failure mode the framework is designed to catch. A
target can be reproducible and predictable while still failing topology
specificity and directed identifiability. That negative result is not a weakness
if framed correctly. It is evidence that the framework blocks a tempting but
unsupported mechanistic interpretation.

Do not claim that Allen VBN lacks mechanistic information in general. The claim
is narrower: the specific aggregated target used here does not support the
specific mechanistic interpretation tested here.

### Objection 6: Sensorium baselines are not competitive with the literature.

Response:

That is acknowledged. Sensorium and Dynamic Sensorium are used as predictive and
interoperability cases, not as SOTA claims. The official stack audit shows
technical integration, while local baselines demonstrate how MouseBrainBench
separates predictive success from mechanistic identifiability. If a reviewer
requires a competitive Sensorium claim, the honest response is that this is
future work or outside the current claim.

### Objection 7: The manuscript mixes many datasets and may lack focus.

Response:

The datasets have different roles:

- synthetic cases validate MIS semantics under known truth;
- Allen VBN is a real negative mechanistic-identifiability control;
- Sensorium/Dynamic Sensorium are predictive and interoperability cases;
- MICRONS is the main positive local structure-function case.

The organizing principle is not dataset accumulation. It is claim separation
across evidence types.

### Objection 8: The bootstrap may not fully solve dependence among dyadic
pairs.

Response:

Agree. The bootstrap is presented as a unit-cluster weighted stability analysis,
not as a complete inferential solution for network-dependent dyads. The
manuscript should explicitly state that robust dyadic/network inference remains
future work. The current result is strengthened by two non-overlapping hold-outs
and matched controls, but not elevated to causal or fully independent
replication.

## Required Language Discipline

Use:

- "partial mouse-brain digital models";
- "claim-aware validation";
- "internally reproduced";
- "two non-overlapping hold-out windows";
- "local observational structure-function association";
- "distance- and degree-matched controls";
- "unit-cluster weighted bootstrap";
- "bounded MICRONS structure-function benchmark";
- "predictive/interoperability case".

Avoid:

- "full digital mouse brain";
- "complete mouse-brain twin";
- "causal mechanism";
- "replicated" without qualification;
- "whole-brain MICRONS";
- "Sensorium SOTA";
- "consciousness";
- "biologically complete simulator".

## Minimal Response Package to Keep Ready

For revision, keep these artifacts synchronized:

- `results/publication_freeze/summary.md`;
- `results/publication_freeze/summary.json`;
- `results/microns_q1_package/summary.md`;
- `results/microns_q1_package/summary.json`;
- `results/q1_sensitivity/summary.md`;
- `results/sensorium_official_baseline_audit/summary.md`;
- `results/digital_twin_claim_audit/summary.md`;
- `docs/DATASET_VOLUME_AUDIT.md`;
- `docs/RELEASE_REPRODUCIBILITY_CHECKLIST.md`.

## Decision Under Review Pressure

If reviewers ask for stronger causal or whole-brain claims, do not comply. That
would weaken the paper scientifically. The defensible revision path is to make
the claim boundaries sharper, improve explanation of the gates, and add
robustness checks only if they directly test the submitted claims.
