# MouseClaimBench

MouseClaimBench is an integrity-aware knowledge-based system for authorizing
bounded scientific claims from computational mouse-brain evidence. It evaluates
a versioned claim profile, typed evidence blocks, and their artifact lineage. It
returns one final decision and separate structural, scientific, and integrity
deficit traces.

MouseClaimBench is not a simulator, an automated peer reviewer, a biological
truth engine, or a complete digital mouse brain. An authorization means only
that the supplied package satisfies the named profile version.

## Current research release

The current revision candidate is the `v0.12.2` three-gate release. It
preserves the submitted data analyses while aligning the mathematical model,
software, tests, benchmarks, and manuscript. The knowledge architecture
separates three layers and composes them only at the final gate:

1. **Graph conformance.** RDF/PROV-O represents profiles, artifacts,
   derivations, evidence blocks, and decisions. Independently executed SHACL
   shapes validate structural requirements.
2. **Package integrity.** Thirteen non-compensatory deficit types inspect
   profile substitution, duplicate identifiers, hashes, reference closure,
   cyclic lineage, duplicated evidence, overlapping cohorts, attestation
   consistency, and missing lineage or attestations.
3. **Scientific authorization.** Profile v2 requires every typed evidence block
   and its mandatory observations. Prediction cannot compensate for absent
   topology, direction, intervention, replication, or anatomical coverage.

Final authorization is exactly `structural && domain && integrity`. A dedicated
decision object preserves all three outputs. SHACL validates graph structure
and does not require a scientific block to pass. A well-formed failed block can
therefore be structurally conformant and scientifically unauthorized.
`FinalAuthorizationSystem` is the canonical public API for this paper-level
decision. `DomainIntegrityAuthorizationSystem` intentionally omits SHACL and is
reserved for component ablations. The historical
`IntegrityAwareAuthorizationSystem` name remains only as a deprecated
compatibility import and is not part of the public `__all__` contract.
Executing the canonical three-gate API requires RDF and SHACL support. Install
it with `pip install -e '.[full-authorization]'`. The
`standards-validation` extra remains as a backward-compatible alias.

The SHACL contract in this release applies to RDF graphs emitted by the typed
MouseClaimBench serializer. It is not claimed to be a complete validator for
arbitrary hostile third-party RDF graphs with duplicated or contradictory
singleton values. Profile identity is checked at both the graph boundary and
the manifest boundary as deliberate defense in depth.

Artifact identifiers and block-lineage declarations must be unique. Duplicate
identifiers produce structured integrity refusals at the canonical API rather
than exceptions. Independence and cohort-disjointness pairs must be
irreflexive. Data-generation identity is the composite
`(study_id, data_generation_id)` when both values are non-empty. A generation
identifier is not assumed to be globally unique across studies.

The evaluated profile is
`mousebrainbench/knowledge/profiles/mouse_brain_claims_v2.yaml`. It contains 10
bounded claim types, 22 evidence-block types, and 60 claim-to-evidence
requirements. Its curation basis is stored in
`mouse_brain_claims_v2_basis.yaml`. The profile is author-defined and
literature-grounded. Its acquisition record now contains 60 stable relation
identifiers and claim-specific necessity rationales, 22 predicate contracts,
and explicit source and consensus status. It is not a community consensus
taxonomy. Upstream adapters execute source-specific scientific predicates. The
authorization engine consumes their attested status and checks observation,
rule, rationale, and provenance admissibility before applying the profile.

## Frozen evidence

The release gate verifies the following results from clean source revisions:

| Evidence package | Frozen result | Authorized interpretation |
|---|---|---|
| Contract mutation | 5,677 cases, zero false authorizations or rejections, exact domain deficits | Profile-relative conformance |
| Python/ASP | Exact domain-status and deficit agreement across all 5,677 cases | Independent rule-execution conformance |
| SHACL | Exact structural conformance and structural deficits across the same 5,677 packages | External graph-contract conformance |
| Final gate | All 8 logical combinations reproduce `S && A && I` | Non-compensatory composition only |
| Domain-gate properties | 55,031 deterministic checks over 10,000 packages, zero violations | Six properties and outside closure for core authorization, not arbitrary final-package changes |
| JSON-LD interchange | Five representative package graphs preserve graph isomorphism | Serializer interoperability across pristine, failed, incomplete, rich-manifest, and large-package cases |
| Integrity attacks | 360 attacked packages, zero false authorizations with the complete domain-plus-integrity configuration | Resistance to the declared controlled attacks |
| Integrity ablation | Seven original omissions create 10 false authorizations each; contradiction removal creates none because status mismatch is a second detector | Domain-plus-integrity configurations with the structural gate excluded to isolate manifest controls |
| Scalability | About 27,700--52,800 package decisions per second on one Apple arm64 host | Descriptive local performance only |
| DANDI:001176 | 5 usable subjects against a frozen minimum of 20 | No predictive authorization and no endpoint repair |
| DANDI:000039 | 32 mice, median held-out correlation 0.310, bootstrap lower bound 0.207 | Bounded population-response prediction |
| DANDI stimulus coverage | All 32 mice have complete train support for held-out contrast-direction conditions | Descriptive split transparency with no decision change |
| MICRONS | Discovery fixes the positive direction and two pre-fixed hold-outs pass directed dyadic and node-permutation tests | One local observational structure--function association |
| Knowledge traceability | 60 relation rationales, 22 predicate contracts, 124 observation slots | Complete traceability of the author-defined policy, not external content validity |
| One-edge policy perturbation | 221 profiles and 3,094 fixed profile-case evaluations | Expected monotonic changes under one relation removal or addition |
| Explanation fidelity | 10,000 packages, 51,480 minimality and witness checks | Counterfactual fidelity and information retention, not human utility |
| Compositional integrity | 2,550 attacked packages plus 20 trust-boundary controls | Exact declared-invariant traces and explicit coherent-forgery escapes |
| Extended integrity regression | 100 frozen packages across 9 edge-case families plus 6 direct API regressions | Reference closure, identifier uniqueness, irreflexive relations, and attestation consistency |
| DANDI threshold sensitivity | Six one-at-a-time operational criteria | Post-outcome decision boundaries, not criterion validity or calibration |

The positive DANDI model is intentionally simple Ridge regression. It tests a
prospective authorization workflow rather than state-of-the-art predictive
performance. MICRONS uses one cortical volume, so its non-overlapping windows
are internal reproduction and not independent biological replication.

The submitted release decision is retained unchanged in
`results/standards_prospective_release/summary.json`. The revision decision is
in `results/profile_v2_second_review_release/summary.json`. The new artifacts
do not alter the MICRONS, Sensorium, Allen, IBL, or frozen DANDI numerical
results and do not replace either DANDI threshold profile.

## Defensible contribution

The project does not claim invention of assurance cases, scientific fact
checking, RDF, PROV-O, SHACL, formal rule execution, or artifact hashing. Its
candidate contribution is the evaluated combination of:

- a domain-specific non-compensatory authorization profile
- admissibility checks for the observations behind every passing block
- category-complete multi-deficit traces linked to exact source artifacts
- relational integrity checks over provenance and declared independence
- exact Python and independently implemented ASP domain execution on all 5,677
  cases, plus external SHACL structural validation over the same packages
- executable three-gate composition with layer-resolved deficit traces
- controlled attacks, component ablations, and deterministic property checks
- pre-access frozen positive and negative external applications
- claim-specific knowledge-acquisition traceability and predicate ownership
- one-edge monotonicity probes over every one-relation removal and addition
- counterfactual sufficiency and minimality checks for category-complete deficit traces
- exhaustive compositions of declared attacks with escaping trust-boundary controls
- targeted regression coverage for dangling references and attestation consistency

The targeted review of representative work found no inspected system with this exact evaluated
combination. This supports differentiation and does not establish universal
priority. See `docs/PROFILE_V2_NOVELTY_AUDIT.md` and
`docs/PROFILE_V2_RESULTS_AND_PUBLICATION_STATUS.md`.

## Claim boundaries

The release explicitly blocks claims of:

- a complete, causal, entity-specific, or whole-brain mouse digital twin
- biological truth or automatic peer-review authority
- causal mechanism from the observational MICRONS result
- external biological replication of the current real-data effects
- consensus validity or completeness of the author-defined profile
- improved human decision quality or explanation utility
- benchmark-leading Sensorium, DANDI, or causal-discovery performance
- empirical generality beyond computational mouse-brain evidence

## Reproducibility

Create an isolated environment from the pinned requirements:

```bash
python3 -m venv .venv-risk-v3
.venv-risk-v3/bin/python -m pip install --upgrade pip
.venv-risk-v3/bin/python -m pip install -r requirements-semantic-risk-v3-lock.txt
.venv-risk-v3/bin/python -m pip install -r requirements-standards-lock.txt
.venv-risk-v3/bin/python -m pip install -e .
```

Both lock files are required by the release verification. The standards lock
provides the RDF/SHACL and property-based testing dependencies used by the
profile-v2 standards and formal-property tests.

The lightweight DANDI stimulus-coverage audit uses HTTP range requests and does
not download the selected 7.6 GB corpus:

```bash
.venv/bin/python -m pip install -e '.[dandi-remote]'
.venv/bin/python scripts/analyze_dandi_stimulus_coverage.py
```

The complete historical test suite also covers optional causal-direction
experiments. Install their declared extra before running all tests:

```bash
.venv-risk-v3/bin/python -m pip install -e '.[hybrid-validation]'
.venv-risk-v3/bin/python -m pytest -q
```

Verify the frozen release from a clean checkout:

```bash
ENV_PATH=.venv-risk-v3 bash scripts/reproduce_standards_prospective_v3.sh verify
```

Verify the major-revision response artifacts:

```bash
ENV_PATH=.venv-risk-v3 bash scripts/reproduce_major_revision.sh verify
```

Verify the second-review response and its exact paper-v2 test scope:

```bash
ENV_PATH=.venv-risk-v3 bash scripts/reproduce_second_review.sh verify
```

Verify the paper-code-result consistency revision:

```bash
ENV_PATH=.venv-risk-v3 bash scripts/reproduce_consistency_release.sh verify
```

The current paper-v2 tests and the historical compatibility suite are separated
in `docs/PAPER_V2_TEST_SCOPE.md`. Passing legacy tests is not additional
scientific evidence for profile v2.

Rebuilding numerical artifacts is deliberately separate:

```bash
ENV_PATH=.venv-risk-v3 bash scripts/reproduce_standards_prospective_v3.sh rebuild
```

The revision-only artifacts can be rebuilt without downloading new raw data:

```bash
ENV_PATH=.venv-risk-v3 bash scripts/reproduce_major_revision.sh rebuild
```

Rebuild requires the raw public data at the configured local paths. Public
identifiers, versions, asset sizes, and official hashes are retained in the
frozen DANDI result. Raw provider data are not redistributed.

Artifacts whose `git_revision` ends in `-dirty` are provisional and fail the
release gate. Numerical JSON artifacts must not be edited manually.

## Manuscript

The canonical Knowledge-Based Systems source is `main.tex` at the repository
root. The `paper/` directory is a byte-identical mirror for structured tooling.
Synchronize and verify it with:

```bash
.venv-risk-v3/bin/python scripts/sync_manuscript_mirror.py
.venv-risk-v3/bin/python -m pytest -q tests/test_manuscript_mirror.py
```

The manuscript uses the Elsevier `elsarticle` class, a single abstract below
250 words, six English keywords, numbered references, editable tables, and a
separate `highlights.txt` file containing five highlights of at most 85
characters. Build the flat Editorial Manager package with:

```bash
.venv-risk-v3/bin/python scripts/build_elsevier_submission.py
```

## Research history

Earlier semantic-risk, selective-prediction, causal-direction, and profile-v1
experiments remain in Git and in their named result directories for
traceability. They are not part of the current standards/prospective release
unless referenced by its manifest. Failed frozen experiments remain preserved
as negative evidence and must not be relabelled as independent confirmation
after tuning.
