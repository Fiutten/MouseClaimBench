# Profile v2 standards/prospective results and publication status

Frozen release: `results/standards_prospective_release/summary.json`

Major-revision response:
`results/profile_v2_major_revision_release/summary.json`

## Construct-validity response

The original standards/prospective release remains unchanged. Version 0.10.0
adds four bounded response artifacts generated from clean revision `373de37`:

- complete author-policy traceability for 10 claims, 22 predicate contracts,
  60 relation-specific rationales, 124 observation slots, and 20 bibliography IDs
- one-edge structural policy perturbation over 221 profiles and 3,094 fixed profile-case decisions
- counterfactual explanation checks on 10,000 packages, including 9,839 full
  repairs and 51,480 individual necessity and witness checks
- 2,550 attacked packages spanning all 255 non-empty compositions of the eight declared attacks across 10 claims, plus 20
  coherent-forgery controls that intentionally expose the external-trust boundary

Only one relation removal expands authorization: omitting `data_quality` from
bounded prediction changes the Dynamic Sensorium target from refusal to
authorization. Fifty-two one-block extensions produce 71 contractions. These
results quantify dependence on policy structure. They do not select a preferred
profile or establish independent content validity.

All declared attack compositions are blocked with exact traces. All coherent
content/hash and independence-metadata forgeries remain authorized because the
package provides no external truth anchor. This is an explicit negative control,
not a universal security claim.

## Release decision

All 11 frozen release conditions pass. The release is classified as a bounded
Knowledge-Based Systems submission candidate and not as an acceptance
guarantee. Every referenced result records a clean source revision. Broad
causal, whole-brain, digital-twin, consensus, and human-utility claims remain
blocked.

## Executable contract and standards baseline

The contract benchmark contains 5,677 deterministic cases. Python produces zero
false authorizations, zero false rejections, and exact recovery of all expected
deficit sets. The independent ASP implementation matches authorization and
deficits in all 5,677 generated contract cases.

All 5,677 packages are exported to RDF and evaluated by external `pySHACL`
0.40.1. SHACL evaluates structural conformance rather than scientific
authorization. It recovers every expected structural decision and deficit,
including structurally valid packages whose scientific status is non-passing.
The profile graph contains 318 triples. JSON-LD serialization preserves graph
isomorphism. These results establish standards conformance and also show that
structural validation alone is not novel.

## Formal properties

Ten thousand deterministic packages produce 55,031 checks with no violation:

| Property | Passed checks |
|---|---:|
| Soundness relative to profile v2 | 10,000 |
| Completeness relative to profile v2 | 10,000 |
| Exact deficit identity | 10,000 |
| Input-order invariance | 10,000 |
| Irrelevant-evidence invariance | 10,000 |
| Monotonic degradation | 5,030 |
| Outside-profile closure | 1 |

These are properties of the executable profile. They are not proof of biological
truth or profile content validity.

## Integrity attacks and ablation

The benchmark contains 10 pristine packages and 360 attacked packages. It
applies eight single attacks and every pairwise combination to all 10 claims.
The complete integrity gate has zero false authorization and zero false
rejection. It also recovers the exact attack trace in all 370 packages.

The core profile alone falsely authorizes all 360 attacks. A hash-only baseline
falsely authorizes 280. Seven original leave-one-integrity-control-out systems
falsely authorize 10 packages, corresponding to the omitted single attack in
every claim. Removing the contradiction check alone authorizes none because the
block--attestation mismatch invariant is an independent detector. The result
establishes conditional necessity and one defensive redundancy under the
declared deterministic threat model. It is not an adaptive security evaluation.

## Scalability

On one Apple arm64 host with Python 3.12.13, median throughput ranges from 27,653
to 52,816 package decisions per second for batches of 100, 1,000, and 10,000.
Median time is 0.362 s for 10,000 packages. Integrity evaluation of one valid
package takes 0.149 ms with 25 artifacts and 5.328 ms with 5,000. The
reported log-log slope is descriptive and not an asymptotic proof.

## Mouse-brain applications

The profile-v2 artifact mapping authorizes three bounded target claims and no
strict digital-twin claim. Static Sensorium authorizes its declared predictive
comparison. IBL authorizes one topology-specific behavioural prediction under
locked splits. MICRONS authorizes one local observational structure-function
association after dependence-aware inference. Allen directed topology and
Dynamic Sensorium remain unauthorized under their declared deficits.

The MICRONS coefficient is positive in discovery and two non-overlapping
hold-out windows. Directed dyadic standard errors are 0.00200, 0.00209, and
0.00205. No permutation exceeds the observed statistic in 1,000 draws, giving
the finite-permutation corrected value `p = 1/1001`, approximately `0.001`, in every
window. All windows belong to one cortical volume. This is internal reproduction
of a local observational association and not causal or external replication.

## Prospective DANDI applications

DANDI:001176 remains negative because only 5 usable subjects satisfy the frozen
simultaneous-signal schema against a minimum of 20. The model was not run and the
endpoint was not repaired.

DANDI:000039 contains 32 selected and usable mice. The frozen Ridge model gives
a median held-out subject correlation of 0.310, a subject-bootstrap 95% interval
of `[0.207, 0.369]`, 93.75% positive subjects, model squared error 11.579, and
intercept squared error 12.244. All five frozen conditions pass. The authorized
claim is limited to population contrast-response prediction within this
resource. No state-of-the-art model claim is made.

## Publication assessment

The package now supports a substantial knowledge-engineering paper based on six
orthogonal forms of evidence: contract mutation, independent execution,
standards validation, formal properties, integrity attacks and ablations, and
frozen external applications. The novelty claim is the evaluated integration,
not any component in isolation.

The remaining scientific weaknesses are explicit:

- profile v2 is author-defined rather than consensus-validated
- controlled attacks do not represent an adaptive adversary
- SHACL checks structure, not biological truth
- MICRONS contains one volume and internal hold-outs
- one DANDI positive case uses a simple predictor and one case remains negative
- no human interpretability or decision-quality claim is evaluated

These limitations make the claim boundary narrower. They do not erase the
systems contribution. Journal acceptance remains uncertain and must not be
described as guaranteed.

## Reproduction

From a clean checkout:

```bash
ENV_PATH=.venv-risk-v3 bash scripts/reproduce_standards_prospective_v3.sh verify
```

Rebuild only when the required public raw data are locally available:

```bash
ENV_PATH=.venv-risk-v3 bash scripts/reproduce_standards_prospective_v3.sh rebuild
```
