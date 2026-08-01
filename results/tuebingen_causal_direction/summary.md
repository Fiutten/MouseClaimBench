# Tuebingen Causal Direction Adapter

- Decision: `tuebingen_external_direction_benchmark_ready`
- Pairs loaded: `108`
- Direction attempts: `103`
- Direction attempt rate: `0.954`
- Direction accuracy: `0.485`
- Direction accuracy Wilson 95% CI: `[0.391190583237978, 0.5807304291376137]`
- Weighted accuracy: `0.544`
- Weighted accuracy bootstrap 95% CI: `[0.4146474342125831, 0.6649792084401364]`
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
