# ClaimBench v2 Contract-Conformance Stress Test

- Decision: `claimbench_v2_blocks_overclaiming_under_broad_attacks`
- Cases: `144`
- Families: `12`

| Evaluator | TP | FP | TN | FN | FPR | FNR |
|---|---:|---:|---:|---:|---:|---:|
| `ablated_claim_gate_no_directed` | `528` | `12` | `612` | `0` | `0.019` | `0.000` |
| `ablated_claim_gate_no_reproducible` | `528` | `0` | `624` | `0` | `0.000` | `0.000` |
| `ablated_claim_gate_no_topology` | `528` | `12` | `612` | `0` | `0.019` | `0.000` |
| `claim_gate` | `528` | `0` | `624` | `0` | `0.000` | `0.000` |
| `compensatory_score` | `453` | `84` | `540` | `75` | `0.135` | `0.142` |
| `correlation_only` | `467` | `325` | `299` | `61` | `0.521` | `0.116` |
| `leaderboard_only` | `461` | `319` | `305` | `67` | `0.511` | `0.127` |
| `reliability_only` | `326` | `178` | `446` | `202` | `0.285` | `0.383` |
| `topology_only` | `192` | `24` | `600` | `336` | `0.038` | `0.636` |

## Families

- `causal_positive_local`
- `digital_twin_decoy_without_causality`
- `digital_twin_decoy_without_independent_validation`
- `digital_twin_positive_control`
- `direction_without_topology`
- `leaderboard_overclaim`
- `matched_structure_function_positive`
- `mechanistic_positive_noncausal`
- `prediction_only_common_drive`
- `reliability_without_structure`
- `spatial_structure_function_confound`
- `topology_without_direction`
