# Orthogonal semantic-risk shifts v5.4

- All 13 evaluable levels certified: `false`
- False-authorization levels: `['tail_t10', 'tail_t2', 'dimension_8', 'lag_2', 'lag_3', 'lag_4', 'confound_absent', 'confound_present']`
- Missed-positive levels: `['tail_gaussian', 'tail_t30', 'tail_t10', 'tail_t5', 'tail_t3', 'tail_t2', 'dimension_6', 'dimension_8', 'lag_2', 'lag_3', 'lag_4', 'confound_absent', 'confound_present']`

| Family | Level | Failures | Misses | Risk UCB | Coverage LCB | Shift warning | Certified |
|---|---|---:|---:|---:|---:|---:|---:|
| `tail_heaviness` | `tail_gaussian` | 0 | 37 | 0.0541 | 0.3646 | false | true |
| `tail_heaviness` | `tail_t30` | 0 | 42 | 0.0541 | 0.3271 | false | true |
| `tail_heaviness` | `tail_t10` | 1 | 40 | 0.0747 | 0.3364 | false | true |
| `tail_heaviness` | `tail_t5` | 0 | 32 | 0.0541 | 0.3552 | true | true |
| `tail_heaviness` | `tail_t3` | 0 | 28 | 0.0541 | 0.4127 | true | true |
| `tail_heaviness` | `tail_t2` | 3 | 24 | 0.1086 | 0.3741 | true | false |
| `dimensionality` | `dimension_6` | 0 | 31 | 0.0541 | 0.3837 | false | true |
| `dimensionality` | `dimension_8` | 3 | 48 | 0.1086 | 0.2458 | true | false |
| `temporal_depth` | `lag_2` | 1 | 22 | 0.0747 | 0.4923 | false | true |
| `temporal_depth` | `lag_3` | 1 | 32 | 0.0747 | 0.3364 | false | true |
| `temporal_depth` | `lag_4` | 1 | 35 | 0.0747 | 0.3741 | false | true |
| `latent_confounding` | `confound_absent` | 2 | 38 | 0.0924 | 0.2546 | false | true |
| `latent_confounding` | `confound_present` | 1 | 29 | 0.0747 | 0.4923 | true | true |

These families estimate failure behavior on exact official TimeGraph model contrasts. They do not calibrate a universal shift detector, establish monotonic degradation outside ordered families, or provide real-domain risk.
