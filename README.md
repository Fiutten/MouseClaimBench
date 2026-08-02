# MouseClaimBench

MouseClaimBench is an evidence-constrained knowledge-based system for auditing
scientific claims made from computational mouse-brain studies. It converts
domain evidence into explicit facts, applies a versioned non-compensatory rule
base, and returns both a claim disposition and a machine-readable explanation.
It is not a simulator, a biological truth engine, or a complete digital mouse
brain.

## Knowledge architecture

The executable system separates five elements that must not be collapsed into
a single score:

1. A versioned domain profile declares the claim vocabulary and the evidence
   required by every claim.
2. Evidence facts retain their original values, source, predicate, rationale,
   and status.
3. Prioritized rules implement vetoes, review escalation, uncertainty, scope
   boundaries, and support without compensation across evidence types.
4. Every conclusion includes the fired rule, its witnesses, the evaluated rule
   sequence, the source facts, and the SHA-256 identity of the knowledge profile.
5. A release gate checks clean revisions, required artifacts, inference
   conformance, and explicit scientific claim boundaries.

The evaluated profile is
`mousebrainbench/knowledge/profiles/mouse_brain_claims_v1.yaml`. It defines ten
claims, 22 claim-to-evidence relations, and five inference rules for
computational mouse-brain evidence. It is an author-proposed,
literature-grounded policy, not an externally validated consensus taxonomy. Its
machine-readable curation basis is stored beside the profile and documented in
`docs/KNOWLEDGE_PROFILE_CURATION.md`. The engine can load another structurally
valid profile, but no other scientific domain is empirically evaluated here.
Profile extensibility is therefore an architectural property, not evidence of
cross-domain generality.

## Scientific semantics

Evidence blocks are non-compensatory. Prediction cannot replace topology,
direction, intervention, independent validation, or whole-brain coverage. The
knowledge profile also separates three concepts that are often grouped under
reproducibility:

- computational reproducibility of code and artifacts
- within-resource reproduction across non-overlapping units or cohorts
- external replication in an independent study, resource, or laboratory

Observed values remain in their source scale. Correlations, bootstrap
intervals, and reliability coefficients are not converted into an invented
common score. The five internal disposition identifiers are `supported`,
`blocked`, `uncertain`, `out_of_scope`, and `needs_external_review`. In reports,
the last state is displayed as *referred for review*. Evidence marked
`not_applicable` means *not targeted by the evaluated protocol*.

## Evidence status

Four bounded mouse-brain cases exercise the knowledge profile:

- **Allen VBN:** internally reproduced target structure with failed topology
  specificity and directed identifiability. This is a real negative mechanistic
  case.
- **Static Sensorium:** predictive and topographic evidence reproduced within
  the resource. It does not provide causal evidence or external replication.
- **Dynamic Sensorium:** a temporal model beats a mean-response baseline in two
  stored five-mouse cohorts. Temporal prediction is not treated as biological
  direction.
- **MICRONS:** a fixed local `all_pairs/readout_location` association passes in
  discovery and two non-overlapping hold-outs under distance, degree, FDR, and
  unit-cluster bootstrap controls. All windows come from one resource.

The knowledge-system audit reproduces all 40 author-generated migration decisions and
adds a complete inference trace to all 40. These are conformance results. They
do not independently establish biological truth. A separate structural-
equation oracle benchmark compares the non-compensatory policy with an
equal-weight compensatory policy and a prediction shortcut. It reports
case-cluster uncertainty and per-claim behavior against known data-generating
structures. SciFact tests the separation between
evidence retrieval and support inference. Tuebingen tests abstention from
unsupported causal-direction claims.

The legacy 144-case suite remains a software contract-conformance test because
its labels share the operational semantics of the legacy gate. It is not
presented as independent scientific validation.

### Post-submission research branch

The `hybrid-selective-v2` branch is an experimental evolution kept separate
from the submitted manuscript. It combines a development-trained selective
classifier with semantic support vetoes and an established additive-noise
direction method. Its model was fitted and calibrated on 3740 already-consumed
cases, then frozen before one 3600-case confirmation run over ten new
data-generating regimes.

The frozen primary endpoint did **not** pass. Six substantive conditions passed,
including zero semantic-veto violations, 97.73% aggregate coverage, a 4.72%
selective error, and a 2.00% false-authorization fraction. The directional
condition failed materially: ANM attempted 74.47% of cases but achieved 51.73%
accuracy across all attempted regimes and 65.15% within regimes with a declared
structural direction. The outcome therefore supports only a partial engineering
claim: hard vetoes enforce semantic boundaries and reduce false support, while
the tested ANM component is not a reliable universal direction gate under
confounding, selection, post-nonlinearity, and measurement error.

The immutable outcome is in
`results/hybrid_selective_confirmation/summary.json`. Its descriptive audit is
in `results/hybrid_selective_outcome_audit/audit.md`. These results must not be
used to claim a second strong Q1 contribution, and a tuned rerun cannot be
reported as independent confirmation.

## Scope boundaries

The current artifacts do not support claims of:

- MouseClaimBench does not support a complete, causal, entity-specific, or
  whole-brain mouse digital twin
- cross-domain empirical generality
- external biological replication of the real-data effects
- improved human decision quality or automated peer review
- benchmark-leading Sensorium, SciFact, or causal-discovery performance
- universal scientific truth verification
- language-model authority over claim authorization

A previously prepared expert protocol is retained as optional future work. No
human study has been executed, and a human-effect claim is not part of the
knowledge-system scope or release gate.

## Installation and checks

Use an isolated environment. Exact validated versions are recorded in
`requirements-lock.txt`, while supported ranges remain in `pyproject.toml`.

```bash
python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r requirements-lock.txt
.venv/bin/python -m pip install -e .
.venv/bin/python -m compileall mousebrainbench scripts
.venv/bin/python -m pytest -q
.venv/bin/python -m ruff check .
```

Raw public datasets are excluded from Git. Dataset locations can be supplied to
their adapters. Lightweight MICRONS summaries copied from
`Fiutten/Mouse-brain` are byte-preserved and documented in
`results/EXTERNAL_ARTIFACT_PROVENANCE.md`.

## Main workflows

```bash
.venv/bin/python -m mousebrainbench.benchmarks.oracle_sem_claim_benchmark
.venv/bin/python -m mousebrainbench.benchmarks.real_case_claim_matrix
.venv/bin/python -m mousebrainbench.benchmarks.knowledge_system_audit
.venv/bin/python -m mousebrainbench.benchmarks.knowledge_system_release

# Experimental hybrid-selective branch only
bash scripts/setup_hybrid_validation_env.sh .venv-hybrid
.venv-hybrid/bin/python -m mousebrainbench.benchmarks.hybrid_selective_outcome_audit
```

Artifacts whose `git_revision` ends in `-dirty` are provisional. Submission
artifacts must be regenerated from a clean tree after the relevant code commit.
Numerical JSON files must never be edited manually.

## Manuscript

The canonical source is `main.tex` at the repository root. The `paper/` tree is
a byte-identical mirror for structured local tooling. Synchronize and verify it
with:

```bash
.venv/bin/python scripts/sync_manuscript_mirror.py
.venv/bin/python -m pytest -q tests/test_manuscript_mirror.py
```

Build the flat Elsevier Editorial Manager package with:

```bash
.venv/bin/python scripts/build_elsevier_submission.py
```

The generated `dist/elsevier-submission/` folder includes a SHA-256 manifest.
