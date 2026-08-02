# CausalRivers v5 real transport

- Decision: `descriptive_real_transport_completed_no_external_certificate`
- Claim: `topology_specific`
- Independent geographical clusters: `3`
- Exact external certificate: `not allowed`

| Block | Usable pairs | Authorizations | False | Coverage |
|---|---:|---:|---:|---:|
| `bavaria_historical` | 80 | 36 | 7 | 0.4500 |
| `east_germany_historical` | 80 | 38 | 9 | 0.4750 |
| `elbe_historical_matched` | 80 | 37 | 17 | 0.4625 |
| `elbe_flood_2024` | 80 | 27 | 10 | 0.3375 |

## Matched flood shift

- Shared pairs: `80`
- Transitions: `{"off_to_off": 33, "off_to_on": 10, "on_to_off": 20, "on_to_on": 17}`
- Shift warning: `true`

CausalRivers tests real-data transport and a matched flood shift. Its three dependent geographical clusters cannot validate the exact TimeGraph risk bound.
