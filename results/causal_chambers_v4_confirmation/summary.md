# Causal Chambers v4 target-calibrated confirmation

- Decision: `v4_external_contract_not_supported`
- Risk lock passed: `false`
- Final evaluation opened: `false`
- Calibration experiments: `74`
- Risk-lock experiments: `43`

## Risk-lock comparators

| Policy | Risk UCB | Coverage LCB | Recovery LCB | Certified |
|---|---:|---:|---:|---:|
| `fixed_probability_0_5` | 0.4613 | 0.5874 | 0.8871 | false |
| `evidence_contract_only` | 0.2569 | 0.4678 | 0.8871 | false |
| `unconstrained_ltt` | 0.3625 | 0.5148 | 0.8871 | false |
| `semantic_ltt_without_activation_floor` | 0.2569 | 0.4678 | 0.8871 | false |
| `confidence_only_target_calibrated` | 0.0673 | 0.0000 | 0.0000 | false |
| `semantic_ltt_nondegenerate` | 0.0673 | 0.0000 | 0.0000 | false |

The exact interval is conditional on exchangeability of experiment-level events. Pair rows are not inferential replicates, and within-dataset dependence remains a declared sensitivity requirement.
