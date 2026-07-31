# Q1 Sensitivity Audit

- Decision: `sensitivity_supports_methodological_claim_not_q1_mechanistic_claim`
- Allen stable negative: `True`
- Sensorium static partial positives: `6/9`

## Dynamic Sensorium

| Cohort | Median MLP - mean | Median MLP - SVD |
|---|---:|---:|
| `dynamic_sensorium2023_oracle` | `0.042861` | `0.003856` |
| `dynamic_sensorium_legacy_ood` | `0.036073` | `0.002195` |

## Interpretation

Allen remains negative under threshold perturbation; Sensorium static has robust partial reliability/topography evidence; Dynamic Sensorium has small but useful NN predictive gains. Together this supports a methodological benchmark claim, while still requiring an official baseline or causal/structural extension for a strong Q1 claim.
