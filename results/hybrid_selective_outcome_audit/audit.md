# Hybrid selective v2 outcome audit

- Decision: `negative_confirmatory_result_with_partial_engineering_signal`
- Frozen primary passed: `False`
- Strong new Q1 claim supported: `False`
- Confirmation reexecuted: `False`
- Model refitted: `False`

## Variable-claim performance

| Policy | Coverage | Selective error | False authorization fraction |
|---|---:|---:|---:|
| `evidence_contract_v3` | 0.7887 | 0.1141 | 0.0364 |
| `equal_weight_compensatory_75` | 1.0000 | 0.0933 | 0.0649 |
| `prediction_shortcut` | 1.0000 | 0.2620 | 0.2898 |
| `unconstrained_selective_logistic` | 1.0000 | 0.0474 | 0.0478 |
| `constrained_selective_hybrid` | 0.9621 | 0.0799 | 0.0260 |
| `constrained_anm_predictor_ablation` | 0.9755 | 0.0948 | 0.0194 |
| `constrained_uncalibrated_ablation` | 0.9623 | 0.0791 | 0.0226 |

## Direction by regime

| Regime | Truth | Coverage | Attempted accuracy | Forward | Reverse | Review |
|---|---|---:|---:|---:|---:|---:|
| `independent_mixture` | `none` | 0.1667 | 0.0000 | 31 | 29 | 300 |
| `confounded_threshold` | `none` | 0.7417 | 0.0000 | 46 | 221 | 93 |
| `reverse_piecewise` | `y_to_x` | 0.7750 | 0.7240 | 77 | 202 | 81 |
| `direct_exponential` | `x_to_y` | 0.9722 | 0.9971 | 349 | 1 | 10 |
| `direct_piecewise` | `x_to_y` | 0.7944 | 0.7517 | 215 | 71 | 74 |
| `direct_post_nonlinear` | `x_to_y` | 0.9833 | 0.0000 | 0 | 354 | 6 |
| `direct_linear_nongaussian` | `x_to_y` | 0.9000 | 0.9074 | 294 | 30 | 36 |
| `measurement_error_direct` | `x_to_y` | 0.6917 | 0.4940 | 123 | 126 | 111 |
| `collider_truncation` | `none` | 0.6250 | 0.0000 | 119 | 106 | 135 |
| `direct_interventional_piecewise` | `x_to_y` | 0.7972 | 0.7108 | 204 | 83 | 73 |

## Scientific assessment

The frozen primary endpoint failed materially on ANM direction, while aggregate error is diluted by constant-label claims and calibration adds no clear confirmatory advantage.

Defensible claim: A constrained selective layer can enforce semantic support boundaries with non-trivial coverage, but the tested ANM gate is not a reliable general directional-evidence component across assumption violations.

Blocked action: Do not tune on v2 and present a rerun as independent confirmation.
