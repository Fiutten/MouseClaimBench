# Profile v2 standards/prospective novelty audit

Checked: 2026-08-02

## Audit boundary

The audit covered scientific claim verification, assurance cases, attributed
argument graphs, semantic graph validation, provenance, model verification and
validation, reproducibility, and computational mouse-brain benchmarking.
Publisher, standards-body, proceedings, DOI, archive, and author pages were
preferred. This is a scoped adversarial search. It is not a systematic review
and cannot prove that no unindexed predecessor exists.

## Closest work and non-novel components

The following components have substantial prior art and are not claimed as new:

- Scientific claim verification retrieves documentary evidence and predicts
  support or contradiction. SciFact, MultiVerS, SciFact-Open, SciTab, SciVer,
  SciTrue, PhyVer, and MEVER cover text, tables, figures, and physical evidence.
- CLAIM-BENCH evaluates LLM extraction and linkage of claim-evidence pairs from
  papers. It does not execute artifact-level scientific authorization profiles:
  https://arxiv.org/abs/2506.08235
- Assurance cases already represent claims, arguments, assumptions, defeaters,
  confidence, and evidence.
- Compliance-by-Construction Argument Graphs combine typed argument graphs,
  deterministic completeness and admissibility constraints, and a W3C
  PROV-aligned ledger in an AI-assisted certification workflow:
  https://doi.org/10.1109/FACCT71761.2026.00009
- Recent attributed-graph work compares the structure and provenance of human-
  and machine-generated assurance cases:
  https://arxiv.org/abs/2604.20577
- Claim-level auditability for research agents already uses persistent semantic
  provenance and protocolized validation:
  https://arxiv.org/abs/2602.13855
- SHACL already validates RDF graphs and returns machine-readable violations:
  https://www.w3.org/TR/shacl/
- PROV-O already provides an interoperable provenance vocabulary:
  https://www.w3.org/TR/prov-o/
- Verification, validation, uncertainty quantification, content hashing, and
  formal rule checking are established engineering methods.
- BMTK, the Virtual Mouse Brain, Allen V1 models, Sensorium, MICRONS, and
  MouseDTB already address simulation, prediction, or biological modelling.
  MouseClaimBench is not a replacement simulator.

## Capability boundary

| Family | Established capability | Profile-v2 distinction evaluated here |
|---|---|---|
| Scientific claim verification | Documentary retrieval, entailment, and multimodal support | Starts from computed artifacts and authorizes only typed domain claims |
| Assurance and argument graphs | Structured claims, evidence, admissibility, and provenance | Executes a mouse-brain profile and returns every scientific and package-integrity deficit |
| SHACL and PROV-O | Structural graph constraints and provenance interchange | Adds non-compensatory domain semantics and relational artifact-dependence controls |
| Model V&V | Verification, validation, uncertainty, credibility, and context of use | Treats each resulting dimension as non-interchangeable evidence |
| Neurocomputational benchmarks | Prediction, simulation, topology, or local biological analysis | Prevents a passed endpoint from promoting mechanism, causality, or twin scope |

## Defensible novelty

The defensible candidate contribution is the evaluated composition of:

1. a versioned domain profile with 10 bounded mouse-brain claim types, 22
   evidence-block types, and mandatory source observations
2. profile-relative, non-compensatory authorization rather than truth scoring
3. complete multi-deficit traces instead of one prioritized rejection reason
4. RDF/PROV-O exchange and third-party SHACL conformance
5. independent Python, ASP, and SHACL execution with exact deficit agreement
6. eight relational integrity controls over artifacts, lineage, cohorts,
   attestations, and profile identity
7. deterministic single and pairwise attacks plus leave-one-control-out ablation
8. formal property checks and measured scalability to 10,000 packages and 5,000
   artifacts
9. two protocols frozen before numerical access to external DANDI assets
10. a dependence-aware, narrowly scoped MICRONS structure-function application

No inspected work reported this exact evaluated combination. The nearest 2026
argument-graph systems overlap at architecture level. MouseClaimBench is
differentiated by domain-operational scientific requirements, complete
cross-engine deficits, controlled package attacks and ablations, and frozen
external neurophysiology applications. This supports a bounded novelty argument
and not a claim of universal priority.

## Results that strengthen the novelty claim

- Python classifies all 5,497 contract cases exactly.
- Independent ASP matches all 262 selected conformance cases.
- External pySHACL matches all 5,497 structural cases and therefore acts as a
  meaningful negative baseline for structural novelty.
- Six formal properties hold in 55,031 frozen checks over 10,000 packages.
- The complete gate blocks all 360 controlled attacked packages. The
  profile-only and hash-only baselines falsely authorize 360 and 280.
- Removing any one of eight integrity controls reintroduces exactly 10 false
  authorizations under the declared attack construction.
- One prospective DANDI protocol remains negative without endpoint repair. A
  second authorizes only held-out population-response prediction in 32 mice.
- MICRONS authorizes one local observational association after directed dyadic
  covariance and node-label permutation. It authorizes no causal or whole-brain
  claim.

## Prohibited novelty claims

The project must not claim invention of assurance cases, argument graphs,
scientific fact checking, knowledge graphs, RDF, PROV-O, SHACL, formal methods,
causal discovery, risk control, or digital-twin validation. It must not claim
that profile v2 is complete, optimal, consensus-validated, or biologically true.
It must not claim a new simulator, state-of-the-art neural prediction, automatic
peer review, human decision improvement, causal MICRONS evidence, external
biological replication, or a complete mouse-brain twin.

## Publication judgement

The current package is a technically substantial and plausibly novel
Knowledge-Based Systems submission candidate when framed as scientific
authorization and artifact-integrity knowledge engineering. It is materially
stronger than a rule-engine demonstration because it contains independent
executors, a standard external baseline, exact attacks and ablations, formal
properties, scaling measurements, and frozen real-data applications.

Acceptance cannot be guaranteed. The principal review risk is content validity:
the profile remains author-defined. The absence of a human panel prevents claims
about consensus and human utility, but it does not invalidate the executable
systems contribution because those claims are explicitly outside scope.
