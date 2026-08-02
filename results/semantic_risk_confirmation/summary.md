# Semantic risk control v3 synthetic confirmation

- Decision: `v3_synthetic_primary_passed`
- Cases: `7200`
- Score refitted: `False`
- Risk policy recalibrated: `False`

| Policy | Coverage | SFAR | False authorizations | Semantic violations |
|---|---:|---:|---:|---:|
| `naive_probability_threshold` | 0.6827 | 0.0492 | 1451 | 4701 |
| `semantic_gate_without_risk_control` | 0.5739 | 0.0031 | 76 | 0 |
| `unconstrained_MAPIE_risk_control` | 0.6346 | 0.0120 | 328 | 3196 |
| `semantic_MAPIE_risk_control` | 0.5607 | 0.0031 | 76 | 0 |
| `evidence_contract_only` | 0.5863 | 0.0082 | 207 | 0 |

## Frozen primary conditions

- `scale_matches_frozen_protocol`: `True`
- `semantic_support_violations_equal_0`: `True`
- `all_variable_claim_families_ltt_certified`: `True`
- `synthetic_macro_supported_coverage_at_least_0.20`: `True`
- `synthetic_empirical_sfar_at_most_0.05`: `True`
