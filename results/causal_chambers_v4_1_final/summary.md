# Causal Chambers v4.1 locked final evaluation

- Decision: `v4_1_bounded_predictive_external_confirmation_failed`
- Selected claims: `predictive, internally_reproduced`
- Final experiments: `33`
- Risk upper bound: `0.1787`
- Coverage lower bound: `0.4784`
- Positive-recovery lower bound: `0.8541`
- Semantic violations: `0`

## Comparator results

| Policy | Risk UCB | Coverage LCB | Recovery LCB | Certified |
|---|---:|---:|---:|---:|
| `abstain_all` | 0.0868 | 0.0000 | 0.0000 | false |
| `fixed_probability_0_5` | 0.4276 | 0.6382 | 0.8541 | false |
| `evidence_contract_only` | 0.1787 | 0.4784 | 0.8541 | false |
| `unconstrained_ltt` | 0.4909 | 0.6382 | 0.8541 | false |
| `semantic_ltt_without_activation_floor` | 0.1787 | 0.4784 | 0.8541 | false |
| `confidence_only_target_calibrated` | 0.3276 | 0.5724 | 0.8541 | false |
| `semantic_ltt_nondegenerate_v4_1` | 0.1787 | 0.4784 | 0.8541 | false |

The result is bounded to predictive and internal-reproduction claims. The final data do not confirm topology, direction, mechanism, or causality.
