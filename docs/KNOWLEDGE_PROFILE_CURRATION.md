# Mouse-Brain Claim Profile v2 Curation

## Scientific status

Profile v2 is an author-defined, literature-grounded operational policy. It is
not a consensus taxonomy and has not received independent expert content
validation. The executable audits establish profile coverage, internal
consistency, source traceability, and decision sensitivity. They cannot establish
scientific completeness, biological truth, or improved human review decisions.

## Acquisition procedure

The profile was constructed through eight recorded steps:

1. define bounded claim wording and prohibited promotions
2. identify distinct evidence functions from methodological and domain sources
3. record each block's operational role, scope, exceptions, and rejected substitutes
4. justify every claim-to-evidence relation independently of block definition
5. separate upstream scientific predicates from authorization-engine checks
6. resolve the internal adversarial audit without treating it as external review
7. verify exact executable and bibliographic coverage
8. measure decision sensitivity to relation removal and conservative extension

The machine-readable record is
`mousebrainbench/knowledge/profiles/mouse_brain_claims_v2_basis.yaml`. It contains:

- 22 evidence-block definitions with bibliographic sources
- 22 predicate contracts declaring evaluation ownership and decision-rule scope
- 60 stable claim-to-evidence relation identifiers
- 60 claim-specific necessity rationales
- explicit author-policy and non-consensus status for every relation

The executable profile contains 124 required observation slots across the 22
blocks. These fields make a passed attestation inspectable. They do not impose a
universal numerical threshold. Source adapters execute endpoint-specific
predicates and provide a status, rule, rationale, observations, and provenance.
The authorization engine checks schema admissibility and then applies the
non-compensatory claim contract.

## Source interpretation

Source identifiers resolve to `references.bib`. A source supports the
methodological function of an evidence block. It is not represented as having
prescribed the complete profile or every author-defined relation. The difference
between source grounding and author synthesis is retained in the exported
relation ledger.

## External review status

The historical v1 panel packet remains only for reproduction of the earlier
profile. It does not validate v2. The v2 review protocol targets all current
claims, predicates, and relations. Until independent ratings satisfy that
protocol, `independent_content_validity` and `human_validation` remain false.

## Change control

Any change to claim vocabulary, evidence schemas, predicate ownership, required
relations, or relation rationale requires:

1. a profile or acquisition-record version increment
2. complete executable and bibliographic coverage
3. regeneration of traceability and structural-sensitivity artifacts
4. explicit reporting of every changed authorization and deficit set
5. preservation of the previous profile-bound release
