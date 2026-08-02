# Profile v2 provenance attacks

- Decision: `provenance_attack_gate_confirmed`
- Cases: `370`
- Attacked cases: `360`
- Full-gate false authorizations: `0`
- Exact attack-trace rate: `1.0000`

| Attack | Cases | Detected |
|---|---:|---:|
| `artifact_hash_tampering` | 80 | 80 |
| `profile_version_substitution` | 80 | 80 |
| `dangling_provenance_reference` | 80 | 80 |
| `circular_provenance` | 80 | 80 |
| `duplicate_independent_artifact` | 80 | 80 |
| `overlapping_independent_cohorts` | 80 | 80 |
| `contradictory_attestation` | 80 | 80 |
| `missing_block_lineage` | 80 | 80 |

The benchmark establishes deterministic detection of the declared controlled attacks. It does not estimate their prevalence, adversarial adaptivity, source trustworthiness, or security against attacks outside this threat model.
