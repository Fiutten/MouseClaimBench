# Hybrid selective v2 release audit

- Decision: `hybrid_v2_reproducible_negative_release`
- Provenance chain passed: `True`
- Q1 ready: `False`
- Scientific status: `negative_confirmation_with_partial_engineering_signal`

## Checks

- development_matrix_hash_matches_manifest: `True`
- development_contains_no_v2_cases: `True`
- policy_uses_frozen_protocol: `True`
- policy_uses_frozen_matrix: `True`
- policy_contains_no_v2_cases: `True`
- confirmation_uses_frozen_policy: `True`
- confirmation_case_hash_matches: `True`
- confirmation_scale_is_complete: `True`
- confirmation_was_not_refitted: `True`
- primary_endpoint_is_preserved_as_negative: `True`
- outcome_audit_uses_frozen_confirmation: `True`
- outcome_audit_uses_frozen_cases: `True`
- outcome_audit_did_not_refit: `True`
- outcome_audit_did_not_change_thresholds: `True`
- strong_q1_claim_remains_blocked: `True`
- all_source_revisions_are_clean: `True`

## Boundary

Permitted: method-development evidence and reproducible negative-result analysis.

Blocked: strong Q1 superiority, universal direction, causal identification, or independent-confirmation claims.
