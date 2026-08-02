# Hybrid selective confirmation v2

- Decision: `hybrid_selective_v2_primary_endpoint_not_passed`
- Cases: `3600`
- Frozen scale: `True`
- Confirmatory refitting: `False`

| Policy | Coverage | Selective error | CP95 upper | False auth. fraction | Veto violations |
|---|---:|---:|---:|---:|---:|
| `evidence_contract_v3` | 0.6733 | 0.0802 | 0.0831 | 0.0280 | 0 |
| `equal_weight_compensatory_75` | 1.0000 | 0.0560 | 0.0580 | 0.0512 | 1286 |
| `prediction_shortcut` | 1.0000 | 0.1572 | 0.1604 | 0.2438 | 6366 |
| `unconstrained_selective_logistic` | 1.0000 | 0.0284 | 0.0299 | 0.0380 | 1968 |
| `constrained_selective_hybrid` | 0.9773 | 0.0472 | 0.0491 | 0.0200 | 0 |
| `constrained_anm_predictor_ablation` | 0.9853 | 0.0563 | 0.0584 | 0.0148 | 0 |
| `constrained_uncalibrated_ablation` | 0.9774 | 0.0467 | 0.0486 | 0.0174 | 0 |

## Primary endpoint

- scale_matches_frozen_protocol: `True`
- semantic_support_veto_violations_equal_0: `True`
- constrained_hybrid_coverage_at_least_0.30: `True`
- constrained_hybrid_selective_error_cp95_upper_at_most_0.12: `True`
- constrained_hybrid_false_authorization_fraction_at_most_0.08: `True`
- constrained_hybrid_selective_error_no_more_than_0.02_above_unconstrained: `True`
- anm_direction_attempted_accuracy_at_least_0.75: `False`
- anm_direction_coverage_at_least_0.20: `True`

## Directional evidence

- attempts: `2681`
- cases: `3600`
- coverage: `0.7447222222222222`
- attempted_accuracy_all_regimes: `0.5173442745244312`
- attempted_accuracy_identifiable_direction_regimes: `0.6514795678722405`
- spurious_attempts_in_no_direction_regimes: `552`
- ambiguous_cases: `919`
- execution_errors: `0`
- status_counts: `{'passed': 1458, 'failed': 1223, 'requires_review': 919}`

## Limits

- general_superiority_of_hard_noncompensatory_contracts
- causal_identification_from_observational_direction_tests
- independent_expert_content_validity
- improved_human_decision_quality
- external_biological_replication
- whole_brain_validity
- complete_mouse_brain_digital_twin
- cross_domain_generality
- Known-truth regimes are low-dimensional stress tests, not neural models.
- ANM direction remains assumption-conditional and is not causal proof.
- The measurement-error label refers to the latent generating direction.
- No expert content-validity or human-utility study was performed.
