# Knowledge-profile internal AI audit

- Decision: `internal_ai_audit_revision_required`
- Independent validation: `false`
- Human raters created: `0`
- Items reviewed: `29`
- Decision counts: `{'critical_veto': 1, 'retain': 9, 'revise_major': 11, 'revise_minor': 8}`
- Source traceability passed: `true`
- Exhaustive rule safety passed: `true`

| Item | Decision | Concern |
|---|---|---|
| `relation__predictive__prediction` | `revise_minor` | `predicate_scope` |
| `relation__externally_replicated__external_replication` | `revise_major` | `independence_definition` |
| `relation__topology_specific__topology_specificity` | `revise_minor` | `control_adequacy` |
| `relation__directed__directed_identifiability` | `revise_major` | `assumption_conditional_direction` |
| `relation__structure_function__structure_function_association` | `revise_major` | `network_dependence` |
| `relation__mechanistic__prediction` | `revise_minor` | `necessary_not_sufficient` |
| `relation__mechanistic__internal_reproduction` | `revise_minor` | `necessary_not_sufficient` |
| `relation__mechanistic__topology_specificity` | `revise_minor` | `alternative_mechanisms` |
| `relation__mechanistic__directed_identifiability` | `revise_major` | `mechanistic_identifiability` |
| `relation__causal__causal_intervention` | `revise_major` | `block_name_semantics` |
| `relation__digital_twin__internal_reproduction` | `revise_minor` | `entity_unit_ambiguity` |
| `relation__digital_twin__topology_specificity` | `revise_minor` | `scale_specificity` |
| `relation__digital_twin__directed_identifiability` | `revise_major` | `construct_overrestriction` |
| `relation__digital_twin__causal_intervention` | `revise_major` | `construct_overrestriction` |
| `relation__digital_twin__whole_brain_coverage` | `revise_major` | `construct_label_mismatch` |
| `relation__digital_twin__operational_compute` | `revise_minor` | `context_of_use_budget` |
| `rule__protocol_scope_boundary` | `revise_major` | `status_precedence` |
| `rule__all_requirements_satisfied` | `critical_veto` | `scientific_support_overstatement` |
| `coverage__claim_set` | `revise_major` | `claim_comprehensiveness` |
| `coverage__evidence_block_set` | `revise_major` | `evidence_comprehensiveness` |

Passing this audit would establish internal consistency and documented literature-grounded criticism only. It cannot establish independent content validity, human consensus, or improved human decision quality.
