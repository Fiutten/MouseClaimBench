# Semantic risk calibration-size sensitivity

This is a post-confirmation exploratory analysis. It is not a new confirmation.

| Calibration cases | Repeats | Certified claims | Coverage (mean) | SFAR (mean) | Repeats SFAR <= target |
|---:|---:|---:|---:|---:|---:|
| 250 | 20 | 2.00 [1, 3] | 0.3034 | 0.0002 | 1.000 |
| 500 | 20 | 3.00 [3, 3] | 0.4220 | 0.0023 | 1.000 |
| 1000 | 20 | 3.10 [3, 5] | 0.4286 | 0.0036 | 1.000 |
| 2000 | 20 | 4.90 [4, 6] | 0.4961 | 0.0036 | 1.000 |
| 3600 | 1 | 6.00 [6, 6] | 0.5607 | 0.0031 | 1.000 |

## Interpretation limits

- The v3 outcomes had already been inspected before this sensitivity analysis.
- No calibration size or threshold may be selected from these results and relabelled confirmatory.
- The analysis measures stability within the synthetic benchmark population only.
