# Profile v2 second-review release

- Decision: `profile_v2_second_review_release_complete`

| Release condition | Passed |
|---|---:|
| `major_revision_release_preserved` | true |
| `asp_covers_every_contract_case` | true |
| `dandi_threshold_boundaries_complete` | true |
| `artifact_revisions_clean` | true |
| `prohibited_claims_remain_false` | true |

## Remaining limits

- profile v2 has no completed independent expert content validation
- the DANDI thresholds remain author-defined operational criteria
- one-edge relation perturbations are monotonicity probes, not alternative profile validation
- counterfactual fidelity does not establish human explanation utility
- the performance result is a core-engine microbenchmark on one host

The second-review release corrects the formal authorization equations, evaluates the independent ASP path on every generated contract case, and exposes post-outcome DANDI decision boundaries. It does not replace the pending independent expert validation, establish human utility, or convert author-defined operational thresholds into criterion-validated standards.
