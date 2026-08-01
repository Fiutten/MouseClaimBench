# Artifact-Grounded Real-Case Claim Matrix v3

- Decision: `artifact_grounded_case_matrix_complete_with_explicit_limits`
- Cases: `4`
- Status counts: `{'supported': 10, 'blocked': 7, 'uncertain': 7, 'out_of_scope': 16, 'needs_external_review': 0}`

| Case | Supported claims | Sources |
|---|---|---|
| `allen_vbn_identifiability_negative` | `computationally_reproducible`, `internally_reproduced` | `results/allen_vbn_mechanistic_identifiability_score.json` |
| `sensorium_static_predictive_topographic` | `predictive`, `internally_reproduced`, `topology_specific` | `results/sensorium_static_model_comparator/summary.json` |
| `dynamic_sensorium_temporal_prediction` | `predictive`, `internally_reproduced` | `results/dynamic_sensorium_model_comparator/summary.json` |
| `microns_local_structure_function` | `computationally_reproducible`, `internally_reproduced`, `structure_function` | `results/microns_primary_robustness/summary.json`<br>`results/microns_q1_package/summary.json` |

## Explicit Limits

- The case matrix is an artifact-grounded application of a declared policy, not independent ground truth.
- Internal reproduction does not mean replication across animals, laboratories, or resources.
- Unknown and not-applicable evidence are never converted into failed measurements.
- No real case supports a causal, externally replicated, whole-brain digital-twin claim.
