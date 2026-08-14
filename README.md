# MouseClaimBench

MouseClaimBench is an integrity-aware knowledge-based system for authorizing
bounded scientific claims from computational mouse-brain evidence. It evaluates
a versioned claim profile, typed evidence blocks, and their artifact lineage. It
returns an authorization decision and every scientific or integrity deficit
that prevents authorization.

MouseClaimBench is not a simulator, an automated peer reviewer, a biological
truth engine, or a complete digital mouse brain. An authorization means only
that the supplied package satisfies the named profile version.

## Current research release

The current revision candidate is the `profile-v2-major-revision` release. It
preserves the submitted `standards-prospective-v3` release and adds construct-
validity response artifacts without rewriting its numerical evidence. The
knowledge architecture separates three layers:

1. **Graph conformance.** RDF/PROV-O represents profiles, artifacts,
   derivations, evidence blocks, and decisions. Independently executed SHACL
   shapes validate structural requirements.
2. **Package integrity.** Eight non-compensatory controls inspect profile
   substitution, hashes, broken or cyclic lineage, duplicated evidence,
   overlapping cohorts, contradictory attestations, and missing provenance.
3. **Scientific authorization.** Profile v2 requires every typed evidence block
   and its mandatory observations. Prediction cannot compensate for absent
   topology, direction, intervention, replication, or anatomical coverage.

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
| Contract mutation | 5,497 cases, zero false authorizations or rejections, exact deficits | Profile-relative conformance |
| Python/ASP/SHACL | Exact agreement on their declared comparison sets | Cross-engine implementation equivalence |
| Formalized properties | 55,031 deterministic checks over 10,000 packages, zero violations | Six properties and outside closure relative to profile v2 |
| Integrity attacks | 360 attacked packages, zero false authorizations with the full gate | Resistance to the declared controlled attacks |
| Integrity ablation | Removing any one control creates 10 false authorizations | Necessity under the declared attack construction |
| Scalability | About 63,500--65,500 package decisions per second on one Apple arm64 host | Descriptive local performance only |
| DANDI:001176 | 5 usable subjects against a frozen minimum of 20 | No predictive authorization and no endpoint repair |
| DANDI:000039 | 32 mice, median held-out correlation 0.310, bootstrap lower bound 0.207 | Bounded population-response prediction |
| MICRONS | Positive directed dyadic and node-permutation tests in three internal windows | One local observational structure--function association |
| Knowledge traceability | 60 relation rationales, 22 predicate contracts, 124 observation slots | Complete traceability of the author-defined policy, not external content validity |
| Structural sensitivity | 221 profiles and 3,094 fixed profile-case evaluations | Decision dependence on relation removal or conservative extension |
| Explanation fidelity | 10,000 packages, 51,480 minimality and witness checks | Counterfactual fidelity and information retention, not human utility |
| Compositional integrity | 2,550 attacked packages plus 20 trust-boundary controls | Exact declared-invariant traces and explicit coherent-forgery escapes |

The positive DANDI model is intentionally simple Ridge regression. It tests a
prospective authorization workflow rather than state-of-the-art predictive
performance. MICRONS uses one cortical volume, so its non-overlapping windows
are internal reproduction and not independent biological replication.

The submitted release decision remains immutable in
`results/standards_prospective_release/summary.json`. The revision decision is
in `results/profile_v2_major_revision_release/summary.json`. The new artifacts
do not alter the DANDI, MICRONS, Sensorium, Allen, or IBL numerical results.

## Defensible contribution

The project does not claim invention of assurance cases, scientific fact
checking, RDF, PROV-O, SHACL, formal rule execution, or artifact hashing. Its
candidate contribution is the evaluated combination of:

- a domain-specific non-compensatory authorization profile
- admissibility checks for the observations behind every passing block
- complete multi-deficit traces linked to exact source artifacts
- relational integrity checks over provenance and declared independence
- exact Python and SHACL execution on 5,497 structural cases, with an
  independent ASP path on a deterministic 262-case subset
- controlled attacks, component ablations, and deterministic property checks
- pre-access frozen positive and negative external applications
- claim-specific knowledge-acquisition traceability and predicate ownership
- structural policy sensitivity over every one-relation removal and addition
- counterfactual sufficiency and minimality checks for complete deficit traces
- exhaustive compositions of declared attacks with escaping trust-boundary controls

The scoped novelty audit found no inspected work with this exact evaluated
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
