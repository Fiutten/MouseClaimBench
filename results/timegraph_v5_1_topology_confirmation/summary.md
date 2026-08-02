# TimeGraph v5.1 topology-specific confirmation

- Decision: `v5_1_topology_specific_contract_supported`
- Claim: `topology_specific`
- Fixed threshold: `0.927257772850`
- Risk lock passed: `true`
- Risk-lock failures: `1`
- Risk-lock risk upper bound: `0.046560`
- Risk-lock coverage lower bound: `0.755983`
- Final opened: `true`

| Comparator | Certified | Risk UCB | Coverage LCB | Recovery LCB |
|---|---:|---:|---:|---:|
| `abstain_all` | false | 0.0295 | 0.0000 | 0.0000 |
| `fixed_probability_0_5` | false | 0.1518 | 0.9384 | 0.9699 |
| `evidence_contract_only` | false | 0.1275 | 0.9384 | 0.9699 |
| `frozen_v3_topology_threshold` | false | 0.1275 | 0.9384 | 0.9699 |
| `v5_1_fixed_hierarchical_threshold` | true | 0.0466 | 0.7560 | 0.7741 |

## Boundary

The result concerns only topology-specific authorizations in the frozen TimeGraph mixture. It does not repair the failed global v5 policy.
