# ClaimBench v2 Threat Model

- Decision: `claimbench_threat_model_passed_with_boundaries`
- Threats passed: `7/7`
- Critical failed threats: `0`

| Threat | Severity | Passed | Artifact | Boundary |
|---|---|---:|---|---|
| `rules_without_value` | `critical` | `True` | `results/claimbench_component_ablation/summary.json` | Claim non-redundant audit value, not superior prediction. |
| `threshold_arbitrariness` | `high` | `True` | `results/claim_threshold_sensitivity_v2/summary.json` | Report safe and dangerous regions; do not claim universal threshold robustness. |
| `synthetic_overfit` | `high` | `True` | `results/scifact_claim_verification/summary.json` | Use SciFact as external claim-auditing evidence, not SOTA verification. |
| `causal_overclaim` | `critical` | `True` | `results/tuebingen_causal_direction/summary.json` | Use Tuebingen to block causal overclaims, not to claim causal-discovery performance. |
| `wording_drift` | `critical` | `True` | `results/manuscript_claim_audit/summary.json` | Keep manuscript wording tied to executable claim contracts. |
| `llm_authority_drift` | `critical` | `True` | `results/llm_claim_extraction_audit/summary.json` | Use LLMs only for candidate extraction and conservative wording support. |
| `uncertainty_hidden` | `high` | `True` | `results/uncertainty_claim_gate_v2/summary.json` | Uncertain claims remain uncertain; uncertainty cannot become support. |
