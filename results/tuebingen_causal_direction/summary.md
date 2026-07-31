# Tuebingen Causal Direction Adapter

- Decision: `tuebingen_external_direction_benchmark_ready`
- Pairs loaded: `108`
- Direction attempts: `103`
- Direction accuracy: `0.485`
- Weighted accuracy: `0.544`
- Correlation-only direction overclaims: `79`
- Causal performance claim allowed: `False`
- Causal control claim allowed: `True`

## Method Summary

| Method | Attempts | Coverage | Accuracy |
|---|---:|---:|---:|
| `anm` | `69` | `0.639` | `0.594` |
| `igci` | `107` | `0.991` | `0.439` |
| `lingam_proxy` | `93` | `0.861` | `0.516` |

Interpretation: these methods are transparent causal-direction controls. They are used to audit causal wording, not to claim causal-discovery SOTA.
