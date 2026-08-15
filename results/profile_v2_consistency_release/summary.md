# Profile v2 paper-code-result consistency release

- Decision: `profile_v2_consistency_release_complete`

| Condition | Passed |
|---|---:|
| `parent_release_preserved` | true |
| `domain_contract_exact` | true |
| `structural_contract_exact` | true |
| `original_integrity_benchmark_preserved` | true |
| `compositional_integrity_exact` | true |
| `extended_integrity_regression_exact` | true |
| `three_gate_composition_exact` | true |
| `scalability_and_ablation_current` | true |
| `dandi_outcomes_unchanged` | true |
| `new_artifact_revisions_clean` | true |
| `prohibited_claims_remain_false` | true |

## Remaining limits

- profile v2 remains author-defined and lacks independent content validation
- SHACL validates graph structure rather than scientific truth
- integrity tests remain conditional on declared invariants
- coherent forgery remains outside the trust model
- no new biological dataset or outcome is introduced by this revision

Passing this release establishes exact alignment among the declared mathematical gates, their software implementation, regression tests, and reported conformance artifacts. It does not independently validate the author-defined scientific profile or any biological claim.
