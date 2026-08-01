# Oracle Structural-Equation Claim Benchmark

- Decision: `oracle_benchmark_supports_non_compensatory_contract_with_finite_sample_errors`
- Cases: `500`
- Claim decisions per policy: `5000`

| Policy | TP | FP | TN | FN | FPR | FNR |
|---|---:|---:|---:|---:|---:|---:|
| `evidence_contract_v3` | 1954 | 56 | 2844 | 146 | 0.019 | 0.070 |
| `equal_weight_compensatory_75` | 2013 | 146 | 2754 | 87 | 0.050 | 0.041 |
| `prediction_shortcut` | 2085 | 785 | 2115 | 15 | 0.271 | 0.007 |

## Limits

- The benchmark uses low-dimensional structural equations, not biological neural dynamics.
- Thresholds are prespecified operational diagnostics and are not universal scientific constants.
- The benchmark evaluates claim authorization under finite samples; it does not validate mouse-brain mechanisms.
- Case-cluster intervals address dependence among the ten decisions from one case but not uncertainty over the five chosen data-generating regimes.
