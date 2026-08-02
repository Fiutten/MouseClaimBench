# Profile v2 scalability and ablation

- Decision: `scalability_ablation_confirmed`
- Host: `macOS-26.5.2-arm64-arm-64bit-Mach-O`

| Integrity system | False authorizations | False rejections |
|---|---:|---:|
| `profile_only` | 360 | 0 |
| `hash_only` | 280 | 0 |
| `full_integrity` | 0 | 0 |
| `without_profile_identity_mismatch` | 10 | 0 |
| `without_artifact_hash_mismatch` | 10 | 0 |
| `without_unknown_provenance_reference` | 10 | 0 |
| `without_provenance_cycle` | 10 | 0 |
| `without_duplicate_independent_artifact` | 10 | 0 |
| `without_overlapping_independent_cohorts` | 10 | 0 |
| `without_contradictory_attestation` | 10 | 0 |
| `without_missing_block_lineage` | 10 | 0 |

| Packages | Median s | Packages/s |
|---:|---:|---:|
| 100 | 0.001528 | 65464.7 |
| 1000 | 0.015747 | 63503.8 |
| 10000 | 0.154189 | 64855.6 |

Timings describe this implementation and host only. The controlled ablation isolates the declared attack families, not unknown attacks or real-world prevalence. Neither result establishes scientific content validity.
