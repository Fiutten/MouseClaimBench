# Next paper technical core

Date: 2026-07-31

## Working title

Executable Claim Auditing for Neurocomputational Digital Model Validation

## Strongest contribution

The strongest contribution is not another mouse-brain benchmark. The defensible
contribution is an executable claim-auditing layer that binds manuscript wording,
model evidence, uncertainty, cost, and claim scope.

## What is already solid

- ClaimBench v2 provides a broad adversarial known-truth suite.
- Shortcut evaluators over-authorize unsupported claims.
- The non-compensatory claim gate blocks overclaiming in the current v2 suite.
- Sensitivity analysis exposes a non-trivial safe region and dangerous threshold
  regions. This is stronger than pretending the thresholds are universally
  robust.
- External causal controls show that the gate is not only a MICRONS/Sensorium
  artifact.
- The claim DSL and manuscript auditor convert claims into executable contracts.
- The LLM-ready claim extraction layer is implemented as a non-authoritative
  candidate extractor with deterministic fallback.
- The package is reproducible through a single runner:
  `mousebrainbench.benchmarks.claimbench_reproduce_package`.
- The reviewer threat model is executable and currently passes all critical
  threats with explicit claim boundaries.

## What is not yet strong enough

- SciFact and Tuebingen cause-effect pairs have executable adapters. They are
  external validation cases, not SOTA claims. This is now an explicit claim
  boundary in the unified report and threat model.
- The uncertainty-aware gate currently uses deterministic local perturbations.
  It is useful as a conservative first layer, but it is not a full Bayesian or
  bootstrap uncertainty model.
- The cost-fidelity frontier uses transparent proxy costs. A Q1 paper should add
  measured wall-clock, memory, data volume, and possibly energy estimates.
- The current LLM-ready layer extracts claim candidates deterministically and can
  ingest optional local LLM outputs. A stronger version should evaluate real LLM
  extractors under a fixed benchmark of annotated manuscript claims.

## Novelty position

The novelty should be stated narrowly:

MouseBrainBench-ClaimAudit introduces an executable evidence-to-claim contract
for neurocomputational digital model studies. It separates prediction,
reproducibility, topology, direction, local structure-function association,
causality, digital-twin wording, uncertainty, and computational cost. The
framework audits whether a manuscript-level claim is supported, blocked, or
uncertain under the available artifacts.

## What not to claim

- Do not claim a complete mouse-brain digital twin.
- Do not claim causal evidence from MICRONS observational results.
- Do not claim Sensorium SOTA unless an official, comparable baseline is fully
  reproduced.
- Do not claim universal validity of thresholds.
- Do not claim automatic peer review.

## Required next additions for a stronger Q1 submission

1. Strengthen the SciFact adapter beyond lexical baselines, or explicitly keep
   it as an external claim-auditing sanity check.
2. Add a stronger causal-direction baseline for Tuebingen before making any
   causal-discovery performance claim.
3. Replace proxy cost with measured runtime/memory/data-volume metrics.
4. Build an annotated claim-extraction benchmark for manuscript sentences.
5. Add bootstrap or Bayesian uncertainty for real MICRONS/Sensorium evidence.
6. Produce a paper-level claim audit table automatically from the final LaTeX.

## Decision

Proceed as a second-paper line if the external adapters show useful diagnostic
value and the paper clearly avoids SOTA claims on SciFact or causal discovery.
For a stronger Q1 submission, add at least one stronger external baseline and
run the manuscript auditor on the actual LaTeX sources.
