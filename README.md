# MouseClaimBench

MouseClaimBench is an executable decision-support framework for auditing the
scientific wording attached to computational mouse-brain studies. It is the
standalone repository for the second manuscript in the MouseBrainBench
workstream. It is not a simulator and it is not a complete digital mouse brain.

## Scientific contribution

The framework represents a candidate statement as a claim-specific evidence
contract. Evidence blocks are non-compensatory. Prediction cannot replace
topology, direction, intervention, independent validation, or whole-brain
coverage. Version 3 also separates three concepts that were previously grouped
under reproducibility:

- computational reproducibility of code and artifacts;
- internal reproduction across non-overlapping units or cohorts in one resource;
- external replication in an independent study, resource, or laboratory.

Every observed value remains in its source scale. The v3 contract does not map
correlations, bootstrap intervals, and reliability coefficients to invented
common scores. Its five workflow dispositions are `supported`, `blocked`,
`uncertain`, `out_of_scope`, and `needs_external_review`.

## Evidence status

The repository currently contains four bounded mouse-brain cases:

- **Allen VBN:** internally reproduced target structure, but failed topology
  specificity and directed identifiability. This is a negative mechanistic case.
- **Static Sensorium:** predictive and topographic evidence reproduced within
  the resource. It is not causal evidence or external replication.
- **Dynamic Sensorium:** a temporal model beats a mean-response baseline in both
  stored five-mouse cohorts. Reliability is not estimable in the stored
  comparator, and temporal prediction is not biological direction.
- **MICRONS:** a fixed local `all_pairs/readout_location` association passes in
  discovery and two non-overlapping hold-outs under distance, degree, FDR, and
  unit-cluster bootstrap controls. The three windows come from one resource and
  therefore do not constitute external replication.

The legacy 144-case suite is retained as a **software contract-conformance
test**. Its labels share the operational semantics of the legacy gate and must
not be presented as independent scientific validation. The oracle structural-
equation benchmark provides a separate data-generating reference and reports
ordinary false-positive and false-negative rates with finite-sample errors.

The SciFact adapter separates retrieval from support classification. Its
train-calibrated support baseline uses the official training split and is
evaluated on the development split. The Tuebingen adapter is an abstention and
directional-overclaim control, not a competitive causal-discovery system.

## Human evaluation

No human study has been executed. The repository includes a preregistrable
expert crossover protocol and an unlabeled item builder under
`configs/human_evaluation_protocol.yaml` and
`docs/HUMAN_EXPERT_EVALUATION_PROTOCOL.md`. These artifacts do not demonstrate
improved author or reviewer decisions. Ethics approval, preregistration,
recruitment, double annotation, adjudication, and analysis are still required.

## Prohibited claims

The present artifacts do not support claims of:

- a complete, causal, entity-specific, or whole-brain mouse digital twin;
- improved human decision quality or automated peer review;
- external biological replication of the reported real-data effects;
- state-of-the-art Sensorium, SciFact, or causal-discovery performance;
- universal scientific truth verification;
- language-model authority over scientific claim authorization.

## Installation and checks

Use an isolated environment. The exact validated package snapshot is recorded
in `requirements-lock.txt`, while supported dependency ranges remain in
`pyproject.toml`.

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r requirements-lock.txt
.venv/bin/python -m pip install -e .
.venv/bin/python -m compileall mousebrainbench scripts
.venv/bin/python -m pytest -q
.venv/bin/python -m ruff check .
```

Raw public datasets are excluded from Git. Dataset locations can be passed to
the corresponding adapters. Lightweight MICRONS summaries copied from
`Fiutten/Mouse-brain` are byte-preserved and documented in
`results/EXTERNAL_ARTIFACT_PROVENANCE.md`.

## Main workflows

```bash
.venv/bin/python -m mousebrainbench.benchmarks.oracle_sem_claim_benchmark
.venv/bin/python -m mousebrainbench.benchmarks.real_case_claim_matrix
.venv/bin/python -m mousebrainbench.benchmarks.human_evaluation_protocol
.venv/bin/python -m mousebrainbench.benchmarks.claimbench_v3_release
```

Artifacts whose `git_revision` ends in `-dirty` are provisional. Submission
artifacts must be regenerated after the relevant code commit from a clean tree.
Numerical JSON files must never be edited manually.

## Manuscript

The canonical Overleaf source is `main.tex` at the repository root. Sections,
tables, and figures are also at root level. The `paper/` tree is a byte-identical
mirror retained for structured local tooling. Synchronize and verify it with:

```bash
.venv/bin/python scripts/sync_manuscript_mirror.py
.venv/bin/python -m pytest -q tests/test_manuscript_mirror.py
```

Elsevier Editorial Manager requires a flat LaTeX upload. Build it with:

```bash
.venv/bin/python scripts/build_elsevier_submission.py
```

The generated `dist/elsevier-submission/` folder includes a SHA-256 manifest.
