# Profile v2 compositional integrity stress

- Decision: `declared_compositions_confirmed_with_explicit_trust_boundary`
- In-model packages: `2560`
- In-model attacked packages: `2550`
- In-model false authorizations: `0`
- Exact attack traces: `2560`
- Trust-boundary negative controls: `20`
- Expected trust-boundary authorizations: `20`

| Attack order | Packages across ten claims |
|---:|---:|
| 0 | 10 |
| 1 | 80 |
| 2 | 280 |
| 3 | 560 |
| 4 | 700 |
| 5 | 560 |
| 6 | 280 |
| 7 | 80 |
| 8 | 10 |

Exhaustive composition establishes detection and exact trace recovery only for the eight declared invariant families. The coherent-forgery controls are expected to escape because internally consistent content and metadata cannot be authenticated without trusted archives, signatures, or external identity evidence. This benchmark is not an adaptive security guarantee.
