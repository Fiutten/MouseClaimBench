# Profile v2 major-revision release

- Decision: `profile_v2_major_revision_release_complete`

| Release condition | Passed |
|---|---:|
| `parent_submission_release_preserved` | true |
| `author_policy_traceability_complete` | true |
| `structural_profile_sensitivity_complete` | true |
| `counterfactual_explanations_faithful` | true |
| `declared_attack_compositions_complete` | true |
| `trust_boundary_escape_is_reported` | true |
| `new_artifact_revisions_clean` | true |
| `prohibited_claims_remain_false` | true |

## Remaining limits

- profile v2 remains author-defined and not independently content-validated
- counterfactual fidelity does not establish human explanation utility
- structural sensitivity exposes policy dependence but does not calibrate an optimal profile
- coherent metadata or content forgery remains undetectable without an external trust anchor
- mouse-brain applications retain their original local and resource-specific boundaries

The revision adds complete author-policy traceability, policy-structure sensitivity, counterfactual explanation fidelity, and exhaustive composition of the declared integrity invariants with explicit escaping trust-boundary controls. It does not replace independent expert content validation or human evaluation and does not turn profile conformance into biological validation.
