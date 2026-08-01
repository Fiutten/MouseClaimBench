# MICRONS network-dependent inference

- Decision: `microns_fixed_endpoint_survives_network_dependent_inference`
- Fixed endpoint: `all_pairs/readout_location`

| Cohort | Units | Connected | Coefficient | Dyadic SE | Dyadic p | FL p | Passed |
|---|---:|---:|---:|---:|---:|---:|---:|
| `discovery` | 1000 | 2095 | 0.0105954 | 0.00199733 | 1.38925e-07 | 0.000999001 | `True` |
| `holdout_offset1000` | 992 | 1926 | 0.0178181 | 0.00209082 | 5.77639e-17 | 0.000999001 | `True` |
| `holdout_offset2000` | 999 | 1922 | 0.0116598 | 0.00205266 | 1.7627e-08 | 0.000999001 | `True` |

## Interpretation

A positive decision supports only a local observational association after the fixed controls. Dyadic covariance assumes pairs without a shared unit are independent. The permutation test additionally assumes exchangeability of reduced-model residual arrays under node relabeling. Neither assumption establishes causality or independent biological replication.
