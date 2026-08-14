# Paper-v2 test scope

## Release suite

The MouseClaimBench paper-v2 release is defined by the profile-v2 modules,
frozen protocols, and result artifacts referenced by the manuscript. The
current review suite is:

```bash
python -m pytest -q \
  tests/test_knowledge_authorization_v2.py \
  tests/test_profile_v2_contract_mutation.py \
  tests/test_profile_v2_standards.py \
  tests/test_profile_v2_formal_properties.py \
  tests/test_profile_v2_provenance_attacks.py \
  tests/test_profile_v2_scalability_ablation.py \
  tests/test_profile_v2_artifact_application.py \
  tests/test_profile_v2_traceability.py \
  tests/test_profile_v2_structural_sensitivity.py \
  tests/test_profile_v2_explanation_fidelity.py \
  tests/test_profile_v2_compositional_integrity_stress.py \
  tests/test_dandi_profile_v2_1.py \
  tests/test_dandi_threshold_sensitivity.py \
  tests/test_manuscript_quality.py
```

These tests verify implementation conformance to the author-defined profile,
artifact integrity, frozen decision rules, and manuscript-result consistency.
They do not validate the scientific completeness of profile v2, human utility,
or biological truth.

## Legacy suite

Tests and modules that use the earlier labels `predictive`, `mechanistic`, or
`conscious` are retained for backward compatibility and historical
reproducibility. They are not used as evidence for the paper-v2 claims and are
not part of the release suite above. A passing full-repository test run confirms
that current changes did not break these historical interfaces. It must not be
reported as additional validation of profile v2.
