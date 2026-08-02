# Profile v2 results and publication status

## Technical outcome

The hardened release passes every frozen technical condition. All four new
result packages were generated from source revision `9d4d4fa` without a
`-dirty` suffix.

The internal audit contained 20 items that were not retained. Profile v2 maps
all 20 to explicit changes. The generic `mechanistic`, `causal`, and
`digital_twin` identifiers are absent. The strict complete-twin subtype remains
available only as an intentionally demanding boundary. No external content
validity or human consensus is claimed.

The contract-mutation benchmark contains 5,497 cases:

| Case family | Cases |
|---|---:|
| Pristine complete packages | 10 |
| Single status defects | 240 |
| Omitted required blocks | 60 |
| Missing required observations | 339 |
| Pairwise mixed status defects | 4,848 |

Profile v2 has zero false authorizations, zero false rejections of pristine
packages, and exact recovery of all 5,497 deficit sets. Python and the
independent ASP implementation agree on status and deficits in all 262 selected
conformance cases.

The comparison is deliberately structural. A raw all-passed policy that ignores
the provenance schema falsely authorizes all 339 metadata mutations. The
75-percent compensatory policy falsely authorizes 4,674 mutation cases. The
prediction shortcut falsely authorizes 4,242. A prioritized single-reason trace
recovers the complete deficit set in 27.69 percent of cases, compared with 100
percent for v2. These values establish contract behavior, not scientific truth.

## Artifact-grounded application

Five existing evidence cases produce two bounded target authorizations and no
complete-twin authorization:

| Case | Target result | Main interpretation |
|---|---|---|
| Allen VBN | Not authorized | Nine simultaneous deficits include failed topology and direction. |
| Static Sensorium | Profile authorized | Static visual-response prediction only. No SOTA, mechanism, or causal claim. |
| Dynamic Sensorium | Not authorized | Prediction passes, but response-reliability quality requires review. |
| MICRONS | Not authorized | The local association survives stored controls, but dependence-aware network inference remains absent. |
| IBL behavior | Profile authorized | One topology-specific behavioral prediction across two locked 35-mouse splits. |

The IBL result remains bounded to one shared task ecosystem. Simple source
comparators also pass, so it supports transport and contract application rather
than exclusive algorithmic superiority. The artifact application is
retrospective because all source outcomes predate profile v2.

## Novelty judgement

The individual ingredients are not new. Claim-evidence graphs, assurance cases,
formal argumentation, abstention, causal workflows, and risk calibration all
have substantial prior literature. The candidate novelty is their specific
combination in an executable scientific-authorization system:

1. typed provenance schemas over heterogeneous scientific artifacts
2. explicit profile authorization rather than truth verification
3. non-compensatory conjunction with complete multi-deficit traces
4. independent ASP conformance
5. deterministic mutation testing of the evidence contract
6. false-authorization control at the independent experiment hierarchy
7. bounded mouse-brain applications with positive and negative cases

The scoped literature audit found no inspected system with this exact
combination. This supports differentiation but does not prove universal priority.

## Publication assessment

The package is technically ready for a substantial manuscript revision and is a
plausible methodological submission candidate for Knowledge-Based Systems. A Q1
acceptance cannot be guaranteed. The strongest paper must focus on executable
knowledge engineering and semantic risk, not on digital-brain simulation or
automatic scientific truth verification.

The remaining vulnerabilities must stay visible:

- profile v2 is author-defined rather than a consensus taxonomy
- v2 artifact application is retrospective
- real mouse-population risk evidence comes from one IBL task ecosystem
- MICRONS still lacks a complete network-dependence estimator
- no claim is made about human trace utility or decision quality

These limits no longer invalidate the formal methodological contribution because
the paper does not claim consensus validity or human benefit. They do prevent a
broader scientific-validation claim.

## Reproduction

From a clean checkout:

```bash
ENV_PATH=.venv-risk-v3 bash scripts/reproduce_profile_v2.sh
```

The release decision is stored in
`results/profile_v2_release/summary.json`.
