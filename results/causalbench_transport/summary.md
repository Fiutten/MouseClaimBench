# CausalBench interventional transport audit

- Decision: `causalbench_transport_completed_without_semantic_violation`
- Selected genes: `200`
- Directed pairs per domain: `39800`
- Synthetic finite-sample guarantee transported: `false`

| Domain | Role | Authorizations | Coverage | SFAR | Semantic violations |
|---|---|---:|---:|---:|---:|
| `weissmann_k562` | `calibration_context` | 441 | 0.0018 | 0.2789 | 0 |
| `weissmann_rpe1` | `locked_transport_test` | 2316 | 0.0097 | 0.2474 | 0 |

## Limits

- Cross-cell-line results are transport evidence, not a transported LTT guarantee.
- Perturbation effects can reflect indirect paths, off-target effects, or cell-state shifts.
- The adapter evaluates a bounded gene subset and is not a causal-discovery leaderboard.
- Source-gene bootstrap intervals retain dependence through shared targets and controls.
- A failed authorization is abstention and does not establish absence of an effect.
