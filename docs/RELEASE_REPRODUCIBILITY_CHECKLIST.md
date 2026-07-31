# Release Reproducibility Checklist

## Purpose

This checklist defines what must be true before tagging a public release or
regenerating artifacts for a manuscript revision. It is intentionally stricter
than a normal development checklist because the repository supports scientific
claims.

## Current Release Candidate

| Field | Value |
|---|---|
| Package version | `0.6.0` |
| Manuscript source synchronization commit | `24a7aee` |
| Frozen artifact revision | `7f5293613dc84000719f05c0dc340dc7c9d69c2b` |
| Publication route | `q1_candidate_after_internally_reproduced_microns_signal` |
| Q1 ready according to freeze artifact | `True` |
| MICRONS Q1 package ready | `True` |
| Official Sensorium Q1-qualified | `False` |

The current repository can support the submitted manuscript as a release
candidate, provided that dirty or untracked local files are not folded into the
scientific artifacts without regeneration.

## Required Pre-Release Checks

Run from the repository root:

```bash
git status --short
.venv/bin/python -m compileall mousebrainbench scripts
.venv/bin/python -m pytest -q
.venv/bin/python -m mousebrainbench.benchmarks.digital_twin_claim_audit
.venv/bin/python -m mousebrainbench.benchmarks.sensorium_official_baseline_audit
.venv/bin/python -m mousebrainbench.benchmarks.microns_pilot_gate
.venv/bin/python -m mousebrainbench.benchmarks.publication_freeze
.venv/bin/python -m mousebrainbench.benchmarks.mis2_synthetic_calibration
.venv/bin/python -m mousebrainbench.benchmarks.mis2_threshold_sensitivity
.venv/bin/python -m mousebrainbench.benchmarks.claim_adversarial
.venv/bin/python -m mousebrainbench.benchmarks.claim_attack_suite
.venv/bin/python -m mousebrainbench.benchmarks.microns_primary_robustness
```

Interpretation rules:

- `compileall` must pass.
- `pytest` must pass unless a failure is explicitly caused by missing optional
  external data or optional dependencies. Such failures must be reported
  exactly.
- claim-audit commands must keep blocked claims blocked.
- `publication_freeze` must not downgrade the publication route unexpectedly.
- regenerated artifacts must not contain a `git_revision` ending in `-dirty`.

## Artifact Rules

Do not edit JSON result files manually.

Regenerate artifacts only when all required dependencies and data for that
artifact are available. If an artifact depends on external data that is not
present, keep the previous frozen artifact and document the missing dependency.

Before regenerating manuscript-critical artifacts, use a clean commit:

```bash
git status --short
git rev-parse HEAD
```

Then export the clean revision if the script supports it:

```bash
export MOUSEBRAINBENCH_GIT_REVISION="$(git rev-parse HEAD)"
```

The exported revision must be a hexadecimal Git identifier. Do not export a
revision with `-dirty`.

## Claim-Gate Expectations

| Gate | Expected status | Release interpretation |
|---|---|---|
| Digital-twin claim audit | Passes with full digital-twin claims blocked | Prevents overclaiming. |
| Allen VBN MIS | Negative mechanistic-identifiability case | Reproducible target, no topology/direction claim. |
| Sensorium official audit | Integration available, not Q1-qualified | Predictive/interoperability evidence only. |
| MICRONS micro-pilot gate | Micro-pilot approved, Q1 pilot not approved for small static gate | Historical bounded stress test. |
| MICRONS expanded/Q1 package | Ready | Main positive local observational evidence. |
| Publication freeze | Q1 candidate after internally reproduced MICRONS signal | Current manuscript route. |
| MIS 2.0 synthetic calibration | No false positives in designed non-mechanistic cases | Development evidence for next methodological extension. |
| MIS 2.0 threshold sensitivity | No dangerous or unstable threshold regions | Maps safe versus conservative operating regimes. |
| Claim adversarial benchmark | Claim gate has zero false-positive claims | Demonstrates failure of correlation-only or compensatory overclaiming. |
| Claim attack suite | No high-risk release blockers | Consolidates reviewer-facing claim risk. |
| MICRONS primary robustness | Endpoint survives combined controls or is downgraded | Hardens the local observational MICRONS claim. |

## Dirty-Tree Policy

Untracked manuscript PDFs, temporary folders, local caches, and downloaded data
must not be treated as release artifacts unless they are intentionally tracked or
documented.

Current known local untracked items at the time this checklist was created:

- `MouseBrainBench.pdf`;
- `tmp/`.

These are not part of the release candidate unless explicitly added later.

## GitHub Release Procedure

1. Commit documentation or source changes.
2. Run the pre-release checks again.
3. Push `main` or a dedicated release branch.
4. Create an annotated tag only after checks pass:

```bash
git tag -a v0.6.0 -m "MouseBrainBench v0.6.0 submission release"
git push origin v0.6.0
```

If the tag is created after additional documentation commits, use the actual
release commit, not the earlier manuscript-source commit.

## Zenodo/Archive Procedure

Only archive the release after the GitHub tag is visible remotely. The archive
metadata should state:

- MouseBrainBench is a claim-aware validation framework;
- the release does not provide a full digital mouse brain;
- MICRONS evidence is local, observational, and internally reproduced within one
  resource;
- Sensorium results are predictive/interoperability controls, not SOTA claims;
- Allen VBN is a negative mechanistic-identifiability case.

## Reviewer Revision Procedure

If revision experiments are required:

1. create a new branch named after the revision task;
2. update code and tests first;
3. regenerate only the artifacts affected by the change;
4. run the full pre-release checks;
5. update `docs/REVIEWER_RESPONSE_DOSSIER.md` if claims change;
6. merge only after the claim matrix still matches the regenerated artifacts.
