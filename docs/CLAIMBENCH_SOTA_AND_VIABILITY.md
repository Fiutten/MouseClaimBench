# Claim-aware benchmarking: state of the art and viability

Date: 2026-07-29

## Bottom line

The next MouseBrainBench paper is viable only if the contribution is framed
narrowly and technically. It should not claim to introduce digital-twin
validation in general, mechanistic interpretability in general, or a new
neuroscience benchmark competing with Sensorium, MICrONS, Allen, BMTK, or TVB.

The defensible contribution is a claim-aware validation layer for
neurocomputational digital-model studies. The layer tests whether the evidence
available for a model supports the scientific claim being made. Prediction,
reproducibility, topology specificity, direction, structure-function association,
causality, and digital-twin wording are treated as separate claim levels. Strong
performance in one level cannot compensate a missing gate in another level.

## Relevant state of the art

1. Mechanistic interpretability benchmarks are becoming formal and causal. MIB
   defines standardized tasks and counterfactual-style evaluation for causal
   localization in neural language models. This is close in spirit, but its
   object is model-internal interpretability rather than scientific claim
   governance for neurocomputational models.
   Source: https://openreview.net/forum?id=sSrOwve6vb

2. Causal abstraction provides a formal theory for mechanistic interpretability.
   This raises the standard for any use of the word mechanistic. A predictive
   neural model is not mechanistic unless an explicit abstraction, intervention,
   or directional mechanism is supported.
   Source: https://jmlr.org/beta/papers/v26/23-0058.html

3. VVUQ for digital twins is an active research area. Verification, validation,
   and uncertainty quantification are now expected for credible digital-twin
   claims. MouseBrainBench should align with this literature and should avoid
   implying that a local observational result is a complete twin.
   Source: https://www.nature.com/articles/s41746-025-01447-y

4. Sensorium and Dynamic Sensorium already provide strong predictive benchmarks
   for mouse visual cortex. They are not evidence that a model is mechanistic by
   default. MouseBrainBench can use them as predictive and interoperability
   cases, not as proof of digital-twin status.
   Sources: https://sensorium-competition.net/ and
   https://pmc.ncbi.nlm.nih.gov/articles/PMC10312815/

5. MICrONS gives local, high-resolution structure and function data in mouse
   visual cortex. It is a powerful local structure-function reference, but it is
   not a complete whole-brain mouse connectome.
   Source: https://tutorial.microns-explorer.org/release_manifests/version-1507.html

6. Scientific claim-evidence benchmarks such as CLAIM-BENCH and EvidenceBench
   show that claim verification is itself a live benchmarking topic. Our angle
   must therefore be domain-specific and executable, not merely narrative.
   Sources: https://huggingface.co/papers/2506.08235 and
   https://github.com/EvidenceBench/EvidenceBench

## Novelty assessment

The broad idea "validate claims" is not novel enough. The stronger, narrower
idea is:

- claim-level validation for neurocomputational digital-model studies;
- non-compensatory evidence gates;
- explicit overclaiming risk measurement;
- adversarial synthetic cases where prediction is deliberately decoupled from
  topology, direction, causal support, and structure-function support;
- real-case mapping across Allen, Sensorium, Dynamic Sensorium, and MICrONS;
- a machine-readable claim ledger that links manuscript claims to executable
  artifacts.

This is publishable only if the repository proves that shortcut evaluators
over-authorize unsupported claims while the complete authorization configuration remains conservative, and
if the text clearly states that this is a validation layer rather than a new
brain simulator.

## Main risks

- The method may look simple if presented as a rule checklist. It must be shown
  as a reproducible claim-benchmarking protocol with baselines, attack cases,
  real cases, and measurable risk indices.
- Reviewers may ask why existing VVUQ frameworks are insufficient. The answer is
  that VVUQ is general, while MouseBrainBench operationalizes neurocomputational
  claim types and gives executable tests for overclaiming.
- Reviewers may ask why Sensorium performance is not enough. The answer is that
  Sensorium is a predictive benchmark. It does not by itself certify topology,
  direction, causality, or digital-twin claims.
- Reviewers may ask why MICRONS is not causal. The answer is that the current
  MICRONS component is observational and local. It supports a narrower
  structure-function association only when matched controls and hold-outs pass.

## Viability decision

Proceed, but only under this framing:

MouseBrainBench-ClaimBench is an executable validation and overclaiming-audit
framework for neurocomputational digital-model studies. It separates prediction,
reproducibility, topology specificity, direction, structure-function association,
causality, and digital-twin claims. It does not claim to build or validate a
complete mouse brain digital twin.
