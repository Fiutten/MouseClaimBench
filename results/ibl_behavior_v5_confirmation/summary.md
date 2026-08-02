# IBL behavioral v5 external population

- Decision: `external_ibl_behavioral_population_supported`
- Verified tables: `110`
- Inferential unit: `mouse`
- Risk lock passed: `true`
- Risk-lock risk UCB: `0.082032`
- Risk-lock coverage LCB: `0.917968`
- Final opened: `true`

## Risk-lock comparators

| Comparator | Certified | Failures | Risk UCB | Coverage LCB | Recovery LCB |
|---|---:|---:|---:|---:|---:|
| `abstain_all` | false | 0 | 0.0820 | 0.0000 | 0.0000 |
| `fixed_probability_0_5` | true | 0 | 0.0820 | 0.9180 | 0.9180 |
| `evidence_contract_only` | true | 0 | 0.0820 | 0.9180 | 0.9180 |
| `frozen_v5_1_complete_authorizer` | true | 0 | 0.0820 | 0.9180 | 0.9180 |
| `pass_only_sensitivity` | false | 0 | 0.1616 | 0.8384 | 0.8384 |

## Scope

A positive result is external evidence for one behavioral alignment task in one shared IBL experimental protocol. It is not evidence of causal neural mechanism, independent laboratories, whole-brain validity, or digital-twin validity. Mice are biological units, but shared laboratories and task design remain higher-level dependencies and are reported as limitations.
