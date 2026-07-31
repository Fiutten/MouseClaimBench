# ClaimBench Component Ablation

- Decision: `claimbench_components_have_nonredundant_value`
- Components: `8`
- High/critical components: `7`

| Component | Removed condition | Severity | Evidence | Effect |
|---|---|---|---|---|
| `non_compensatory_gate` | replace with correlation-only evaluation | `critical` | claim_gate_ORI=0.0; correlation_ORI=0.5208333333333334 | Correlation-only evaluation authorizes many unsupported topology, direction, structure-function, and mechanistic claims. |
| `non_compensatory_gate` | replace with compensatory weighted score | `high` | claim_gate_ORI=0.0; compensatory_ORI=0.1346153846153846 | Weighted compensation reintroduces unsupported claim authorization. |
| `topology_block` | allow mechanism without topology specificity | `high` | no_topology_ORI=0.019230769230769232 | Mechanistic claims can pass when topology evidence is missing. |
| `direction_block` | allow mechanism without directed evidence | `high` | no_direction_ORI=0.019230769230769232 | Mechanistic claims can pass when direction is not identified. |
| `uncertainty_status` | force every borderline claim into supported/blocked | `medium` | unsupported_supported=0; supported_uncertain=25 | Uncertainty is needed to report supported claims that become unstable under local evidence perturbations. |
| `scientific_evidence_retrieval` | use lexical citation overlap without retrieval/rationale separation | `high` | shortcut_ORI=0.19886363636363635; bm25_recall_at_5=0.898936170212766; bm25_rationale_ORI=0.25 | SciFact shows that evidence retrieval and support classification must be reported separately from lexical overlap. |
| `causal_abstention` | allow correlation to authorize causal direction | `critical` | correlation_only_direction_overclaims=79; causal_performance_claim_allowed=False | Tuebingen supports causal-overclaiming control, but not a causal-discovery performance claim. |
| `manuscript_audit` | do not audit text against executable claim contracts | `high` | decision=manuscript_claim_audit_passed; active_risk_hits=0; inputs=3 | The current manuscript can be checked directly for unsupported wording. |

## Interpretation

The ablation does not claim that ClaimBench is a better predictor or causal discovery method. It shows that removing claim-gate components reintroduces unsupported scientific wording or removes the ability to audit it.
