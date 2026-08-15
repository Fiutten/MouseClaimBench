# MouseClaimBench paper-code traceability

This ledger binds each manuscript claim to one formal object, implementation,
test, and result artifact. It is an internal consistency control. It does not
provide independent scientific content validity.

Revision metadata for this audit:

- manuscript and software version: `0.12.1`
- executable source commit used by regenerated artifacts:
  `396347dc133aad6b0e1418f486395efc7cf84588`
- planned release tag: `v0.12.1`
- canonical manuscript sources: `main.tex`, `sections/`, and `tables/`

Every result row below points to a JSON artifact that repeats the source commit
in its `git_revision` field. Exact equations and tables are identified by their
LaTeX labels so that numbering changes do not break this ledger.

## Formal and executable contract

| Manuscript object | Meaning | Implementation | Positive and negative tests | Reproducible artifact |
|---|---|---|---|---|
| Methods Eq. `eq:profile-v2` | Versioned claim and evidence vocabulary | `mousebrainbench/knowledge/authorization.py` and `mouse_brain_claims_v2.yaml` | `tests/test_knowledge_authorization_v2.py` | `results/profile_v2_traceability/summary.json` |
| Methods Eq. `eq:field-presence` | Non-empty metadata or observation field | `_present()` in `authorization.py`, `authorization_asp.py`, and `standards.py` | source, rule, rationale, and observation mutations in `tests/test_profile_v2_contract_mutation.py` | `results/profile_v2_contract_mutation/summary.json` |
| Methods Eq. `eq:admissibility` | Passed status plus source, rule, rationale, and required observations | `ClaimAuthorizationSystem._evaluate_fact()` | metadata and observation omissions, plus pristine passes | `results/profile_v2_contract_mutation/summary.json` |
| Methods Eq. `eq:scientific-deficits` | Complete domain-deficit set | `ProfileAuthorizationDecision.deficits` | exact-deficit and multi-deficit tests | `results/profile_v2_contract_mutation/summary.json` |
| Methods Eq. `eq:core-authorization` | Non-compensatory domain authorization | `ClaimAuthorizationSystem.infer()` | all generated domain cases and formal properties | `results/profile_v2_contract_mutation/summary.json` |
| Methods Eq. `eq:counterfactual-explanation` | Sufficiency and individual necessity of repairs | `profile_v2_explanation_fidelity.py` | counterfactual repair and restoration tests | `results/profile_v2_explanation_fidelity/summary.json` |
| Methods Eq. `eq:package-rdf` | RDF graph derived from the same claim, blocks, and manifest | `evidence_package_to_rdf()` | JSON-LD isomorphism across five representative package classes and manifest serialization tests | `results/profile_v2_standards/summary.json` |
| Structural gate `S` | External graph-contract conformance | `validate_structure_with_shacl_v2()` | pristine, missing block, missing metadata, missing observation, and `FAILED`-but-well-formed cases | `results/profile_v2_standards/summary.json` |
| Domain gate `A` | Profile-relative scientific authorization | Python and independently derived ASP rules | Python/ASP status and deficit agreement | `results/profile_v2_contract_mutation/summary.json` |
| Integrity set `I` | Relational package-integrity deficits | `validate_evidence_manifest()` | original attack tests and extended reference-attestation regressions | `results/profile_v2_provenance_attacks/summary.json` and `results/profile_v2_integrity_regression/summary.json` |
| Methods Eq. `eq:integrity-authorization` | Final `S and A and I` decision | `FinalAuthorizationSystem` and `compose_final_authorization()` | all eight logical states plus integrated pristine and bounded-refusal cases | `results/profile_v2_final_gate/summary.json` |

## Empirical and performance results

| Manuscript result | Source implementation | Test or audit | Result artifact |
|---|---|---|---|
| 55,031 deterministic property checks | `profile_v2_formal_properties.py` | `tests/test_profile_v2_formal_properties.py` | `results/profile_v2_formal_properties/summary.json` |
| 2,550 original attack compositions | `profile_v2_compositional_integrity_stress.py` | `tests/test_profile_v2_compositional_integrity_stress.py` | `results/profile_v2_compositional_integrity_stress/summary.json` |
| 100 extended integrity regression packages | `profile_v2_integrity_regression.py` | `tests/test_profile_v2_integrity_regression.py` | `results/profile_v2_integrity_regression/summary.json` |
| Table `tab:scalability-results`, 25, 100, 1,000, and 5,000 artifacts | `profile_v2_scalability_ablation.py` | `tests/test_profile_v2_scalability_ablation.py` | `results/profile_v2_scalability_ablation/summary.json` |
| Experiments Eq. `eq:dandi-pooled-mse` and Table `tab:prospective-applications` | `dandi_profile_v2_1.py` | `tests/test_dandi_profile_v2_1.py` | `results/dandi_profile_v2_1/summary.json` |
| Descriptive DANDI train/test stimulus coverage audit | `dandi_stimulus_coverage.py` | `tests/test_dandi_stimulus_coverage.py` | `results/dandi_stimulus_coverage/summary.json` |
| Table `tab:dandi-threshold-sensitivity` | `dandi_threshold_sensitivity.py` | `tests/test_dandi_threshold_sensitivity.py` | `results/dandi_threshold_sensitivity/summary.json` |
| Experiments Eq. `eq:microns-full-model` and Table `tab:prospective-applications` | `microns_network_inference.py` | `tests/test_microns_network_inference.py` | `results/microns_network_inference/summary.json` |

## Release rules

- Every generated JSON artifact records the software version and Git revision.
- A revision ending in `-dirty` is provisional and cannot satisfy the release gate.
- Numerical source-data artifacts are regenerated only when their declared data
  and dependencies are available. No result JSON is manually edited.
- The manuscript mirror under `paper/` is generated from the canonical root
  manuscript by `scripts/sync_manuscript_mirror.py`.
- Tag `v0.12.1` identifies the complete revision snapshot. Each generated
  artifact records the clean source commit from which that artifact was run.
