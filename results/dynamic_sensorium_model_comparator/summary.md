# Dynamic Sensorium Model Comparator

This comparator asks whether progressively stronger transparent models improve held-out prediction and whether that improvement is allowed to count as mechanistic evidence. Current Dynamic Sensorium artifacts remain predictive evidence only because MIS does not pass.

## dynamic_sensorium2023_oracle

- Ratones comparados: `5`
- Evidencia: `predictive_benchmark_requires_reliability_or_causal_constraints`
- MIS passed: `0`
- Reliability estimable: `0`

| Comparación | Victorias modelo derecho | Mediana delta | Media delta |
|---|---:|---:|---:|
| summary_adapter_vs_temporal_filterbank | `4/5` | `0.008472` | `0.009081` |
| mean_response_vs_temporal_filterbank | `5/5` | `0.031135` | `0.028523` |
| mean_response_vs_temporal_svd | `4/5` | `0.038886` | `0.032829` |
| temporal_filterbank_vs_temporal_svd | `3/5` | `0.007751` | `0.004307` |
| temporal_svd_vs_random_feature | `1/5` | `-0.034472` | `-0.032311` |
| temporal_svd_vs_torch_mlp | `3/5` | `0.003856` | `0.003012` |
| torch_mlp_vs_official_sensorium_bounded | `0/5` | `-0.475864` | `-0.447456` |
| temporal_svd_vs_official_sensorium_bounded | `0/5` | `-0.472008` | `-0.444444` |

| Mouse | Mean | Summary | Temporal filterbank | Temporal SVD | Random feature | Torch MLP | Official bounded | Best |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| `dynamic29515` | `0.42377` | `0.422444` | `0.430917` | `0.414528` | `0.417132` | `0.426902` | `0.027006` | `temporal_filterbank` |
| `dynamic29623` | `0.367982` | `0.390218` | `0.403052` | `0.427126` | `0.354533` | `0.417994` | `0.034993` | `temporal_svd` |
| `dynamic29647` | `0.447155` | `0.480146` | `0.474168` | `0.486161` | `0.451688` | `0.490016` | `0.014153` | `torch_mlp` |
| `dynamic29712` | `0.452984` | `0.475949` | `0.48412` | `0.49187` | `0.452112` | `0.490358` | `0.010643` | `temporal_svd` |
| `dynamic29755` | `0.480565` | `0.500909` | `0.522813` | `0.516918` | `0.499584` | `0.526393` | `0.02759` | `torch_mlp` |

## dynamic_sensorium_legacy_ood

- Ratones comparados: `5`
- Evidencia: `positive_ood_prediction_without_mechanistic_identifiability`
- MIS passed: `0`
- Reliability estimable: `0`

| Comparación | Victorias modelo derecho | Mediana delta | Media delta |
|---|---:|---:|---:|
| summary_adapter_vs_temporal_filterbank | `5/5` | `0.011062` | `0.012608` |
| mean_response_vs_temporal_filterbank | `4/5` | `0.025008` | `0.018932` |
| mean_response_vs_temporal_svd | `5/5` | `0.03091` | `0.030865` |
| temporal_filterbank_vs_temporal_svd | `5/5` | `0.005855` | `0.011933` |
| temporal_svd_vs_random_feature | `1/5` | `-0.012473` | `-0.02034` |
| temporal_svd_vs_torch_mlp | `3/5` | `0.002195` | `-0.011646` |
| torch_mlp_vs_official_sensorium_bounded | `0/0` | `None` | `None` |
| temporal_svd_vs_official_sensorium_bounded | `0/0` | `None` | `None` |

| Mouse | Mean | Summary | Temporal filterbank | Temporal SVD | Random feature | Torch MLP | Official bounded | Best |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| `dynamic29156` | `0.364959` | `0.362926` | `0.391491` | `0.398837` | `0.386364` | `0.401032` | `None` | `torch_mlp` |
| `dynamic29228` | `0.412573` | `0.417795` | `0.428857` | `0.434712` | `0.437185` | `0.448674` | `None` | `torch_mlp` |
| `dynamic29234` | `0.329388` | `0.322883` | `0.323004` | `0.360298` | `0.356199` | `0.343338` | `None` | `temporal_svd` |
| `dynamic29513` | `0.388562` | `0.411134` | `0.421781` | `0.426574` | `0.369238` | `0.36199` | `None` | `temporal_svd` |
| `dynamic29514` | `0.403803` | `0.416169` | `0.428811` | `0.433187` | `0.402924` | `0.440346` | `None` | `torch_mlp` |
