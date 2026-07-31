# External Causal Claim Validation

- Decision: `external_causal_validation_passed`
- Scenarios: `5`

| Evaluator | TP | FP | TN | FN | ORI | CI |
|---|---:|---:|---:|---:|---:|---:|
| `claim_gate` | `24` | `0` | `16` | `0` | `0.000` | `0.000` |
| `compensatory_score` | `19` | `3` | `13` | `5` | `0.188` | `0.208` |
| `correlation_only` | `20` | `10` | `6` | `4` | `0.625` | `0.167` |
| `leaderboard_only` | `20` | `10` | `6` | `4` | `0.625` | `0.167` |
