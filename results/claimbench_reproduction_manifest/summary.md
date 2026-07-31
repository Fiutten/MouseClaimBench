# ClaimBench v2 Reproduction Manifest

- Decision: `claimbench_reproduction_package_passed`
- Stages passed: `15/15`

| Stage | Passed | Decision | Artifact |
|---|---:|---|---|
| `adversarial_v2` | `True` | `claimbench_v2_blocks_overclaiming_under_broad_attacks` | `results/claim_adversarial_v2/summary.json` |
| `threshold_sensitivity_v2` | `True` | `claim_thresholds_have_nontrivial_safe_region_with_reportable_limits` | `results/claim_threshold_sensitivity_v2/summary.json` |
| `external_causal_synthetic` | `True` | `external_causal_validation_passed` | `results/external_causal_claim_validation/summary.json` |
| `uncertainty_gate_v2` | `True` | `uncertainty_gate_blocks_unsupported_support` | `results/uncertainty_claim_gate_v2/summary.json` |
| `external_benchmark_registry` | `True` | `external_benchmarks_registered_with_pending_data` | `results/external_benchmark_registry/summary.json` |
| `scifact_external_claims` | `True` | `scifact_external_claim_audit_ready` | `results/scifact_claim_verification/summary.json` |
| `tuebingen_causal_direction` | `True` | `tuebingen_external_direction_benchmark_ready` | `results/tuebingen_causal_direction/summary.json` |
| `manuscript_claim_audit` | `True` | `manuscript_claim_audit_passed` | `results/manuscript_claim_audit/summary.json` |
| `llm_claim_extraction_audit` | `True` | `llm_claim_extraction_layer_ready_non_authoritative` | `results/llm_claim_extraction_audit/summary.json` |
| `cost_fidelity_frontier` | `True` | `cost_fidelity_claim_frontier_built` | `results/cost_fidelity_claim_frontier/summary.json` |
| `component_ablation` | `True` | `claimbench_components_have_nonredundant_value` | `results/claimbench_component_ablation/summary.json` |
| `reviewer_attack_v2` | `True` | `reviewer_attack_suite_v2_passed_with_reportable_limits` | `results/reviewer_attack_suite_v2/summary.json` |
| `threat_model` | `True` | `claimbench_threat_model_passed_with_boundaries` | `results/claimbench_threat_model/summary.json` |
| `unified_report` | `True` | `claimbench_v2_methodological_package_ready` | `results/claimbench_unified_report/summary.json` |
| `release_check` | `True` | `claimbench_v2_release_ready` | `results/claimbench_v2_release/summary.json` |
