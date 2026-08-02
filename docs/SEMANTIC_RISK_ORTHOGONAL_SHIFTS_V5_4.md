# Orthogonal semantic-risk shifts v5.4

## Design

Version 5.4 replaces the single joint sample-size/noise path with four frozen
TimeGraph shift families. The authorizer and topology threshold remain fixed.
Every evaluable level contains 100 seed bundles and uses exact one-sided bounds
with Bonferroni-adjusted 99.6154% confidence across 13 levels. Only seed bundles
are independent. The 5,200 generated scenarios and their directed pairs are
descriptive lower-level observations.

The planned four-variable condition was removed before outcome inspection. It
cannot satisfy the unchanged three-control specificity contract because only
two variables remain after source and target selection. Relaxing the contract
to retain that condition would have changed the evidence semantics. The
protocol records this compatibility amendment.

## Results

| Family | Level | False bundles | Recovered/eligible | Risk UCB | Shift warning | Certified |
|---|---|---:|---:|---:|---:|---:|
| Tail | Gaussian | 0 | 50/87 | 0.0541 | No | Yes |
| Tail | Student-t 30 | 0 | 46/88 | 0.0541 | No | Yes |
| Tail | Student-t 10 | 1 | 47/87 | 0.0747 | No | Yes |
| Tail | Student-t 5 | 0 | 49/81 | 0.0541 | Yes | Yes |
| Tail | Student-t 3 | 0 | 55/83 | 0.0541 | Yes | Yes |
| Tail | Student-t 2 | 3 | 49/73 | 0.1086 | Yes | No |
| Dimension | 6 | 0 | 52/83 | 0.0541 | No | Yes |
| Dimension | 8 | 3 | 34/82 | 0.1086 | Yes | No |
| Maximum lag | 2 | 1 | 62/84 | 0.0747 | No | Yes |
| Maximum lag | 3 | 1 | 46/78 | 0.0747 | No | Yes |
| Maximum lag | 4 | 1 | 51/86 | 0.0747 | No | Yes |
| Confounding | Absent | 2 | 37/75 | 0.0924 | No | Yes |
| Confounding | Present | 1 | 62/91 | 0.0747 | Yes | Yes |

Heavy tails expose a genuine boundary. The first isolated false bundle appears
at Student-t 10, the shift warning first appears at Student-t 5, and simultaneous
risk certification is lost at Student-t 2. The warning therefore precedes
aggregate certificate loss, but not the first isolated false event.

The eight-variable condition produces three failing bundles and loses the
certificate at the same level where shift is detected. All lag conditions stay
certified and produce no warning. The confounded generator triggers a warning
without certificate loss. Across these four families, the diagnostic therefore
shows early, coincident, absent, and conservative-warning behavior. Four
families are insufficient to estimate a stable sensitivity or specificity, but
they directly refute a universal early-warning interpretation.

Positive recovery is deliberately selective rather than complete. Every level
misses some admissible positives, including certified levels. Certification only
requires the frozen non-zero recovery floor and risk control. The results must
not be paraphrased as comprehensive graph recovery.

## Scientific consequence

The experiment closes the need for more than one prospective degradation path
and identifies two concrete failure regimes. It strengthens boundary knowledge,
not distribution-free robustness. The detector cannot yet be marketed as a
calibrated deployment alarm because warning operating characteristics are based
on four TimeGraph contrasts and include one conservative warning.

Reproduce with:

```bash
.venv-risk-v3/bin/python -m \
  mousebrainbench.benchmarks.semantic_risk_orthogonal_shifts_v5_4
```

