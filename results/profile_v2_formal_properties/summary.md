# Profile v2 formal properties

- Decision: `formal_properties_confirmed`
- Random packages: `10000`
- Property checks: `55031`
- Violations: `0`

| Property | Checks | Violations |
|---|---:|---:|
| `authorization_soundness_relative_to_profile` | 10000 | 0 |
| `authorization_completeness_relative_to_profile` | 10000 | 0 |
| `complete_deficit_identity` | 10000 | 0 |
| `monotonicity_under_evidence_degradation` | 5030 | 0 |
| `invariance_to_input_order` | 10000 | 0 |
| `invariance_to_irrelevant_evidence` | 10000 | 0 |
| `outside_profile_closure` | 1 | 0 |

The properties are theorems of the declared authorization semantics and executable conformance checks of this implementation. They do not prove that the author-defined profile is scientifically complete, biologically true, or externally valid.
