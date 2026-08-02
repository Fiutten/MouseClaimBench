# IBL mouse-level observational transport audit

- Decision: `ibl_observational_transport_completed_without_semantic_violation`
- Insertions / unique mice: `24` / `24`
- Anatomical predictor leakage: `false`
- Synthetic finite-sample guarantee transported: `false`

| Partition | Mice | Authorizations | Coverage | SFAR | Semantic violations |
|---|---:|---:|---:|---:|---:|
| `calibration_context` | 17 | 0 | 0.0000 | 0.0000 | 0 |
| `locked_mouse_test` | 7 | 0 | 0.0000 | 0.0000 | 0 |

## Limits

- The locked partition has seven mice and cannot establish a universal transport guarantee.
- Units within an insertion are dependent despite disjoint unit folds.
- Cosmos anatomy is a coarse target and does not validate behavior or a digital twin.
- Multi-laboratory acquisition is not treated as independent external replication.
- The observational task cannot authorize topology, direction, mechanism, or causality.
