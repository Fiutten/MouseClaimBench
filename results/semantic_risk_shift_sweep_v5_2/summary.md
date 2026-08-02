# Semantic-risk shift degradation v5.2

- Decision: `warning_precedes_or_matches_observed_certificate_loss`
- First warning: `shift_1`
- First certificate failure: `shift_3`
- General-purpose detector validated: `false`

| Level | n | Noise | Failures | Risk UCB | Coverage LCB | Warning | Certified |
|---|---:|---:|---:|---:|---:|---:|---:|
| `shift_0` | 1200 | 0.1 | 2 | 0.0835 | 0.6652 | false | true |
| `shift_1` | 900 | 0.125 | 3 | 0.0992 | 0.5891 | true | true |
| `shift_2` | 700 | 0.16 | 3 | 0.0992 | 0.6213 | true | true |
| `shift_3` | 500 | 0.21 | 6 | 0.1421 | 0.5680 | true | false |
| `shift_4` | 400 | 0.25 | 7 | 0.1556 | 0.6213 | true | false |
| `shift_5` | 250 | 0.35 | 4 | 0.1141 | 0.4755 | true | false |

This sweep characterizes one frozen degradation path. It does not calibrate the warning as a universal deployment decision rule.
