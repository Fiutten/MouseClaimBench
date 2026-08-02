# Finite-sample guarantee scope audit

- Decision: `out_of_scope_certificates_blocked`

| Case | In scope | Raw authorizations | Raw SFAR | Scoped authorizations |
|---|---:|---:|---:|---:|
| `fresh_synthetic_v3` | `true` | 24223 | 0.0031 | 24223 |
| `causal_chambers_locked` | `false` | 247 | 0.0526 | 0 |
| `causalbench_rpe1_locked` | `false` | 2316 | 0.2474 | 0 |
| `ibl_locked_mice` | `false` | 0 | 0.0000 | 0 |

## Limits

- Population identity is a protocol assertion, not a statistical shift estimate.
- Out-of-scope abstention prevents misuse but provides no positive external coverage.
- A new domain requires its own calibration and a new untouched test population.
