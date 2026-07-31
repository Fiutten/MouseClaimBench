# Reviewer Attack Suite v2

- Decision: `reviewer_attack_suite_v2_passed_with_reportable_limits`
- Risks: `1`

| Level | Reviewer attack | Evidence | Response |
|---|---|---|---|
| `medium` | Some threshold cells authorize unsupported claims. | dangerous_cells=135 | Report the dangerous region and keep nominal thresholds fixed. |

## External Controls

- SciFact decision: `scifact_external_claim_audit_ready`
- SciFact shortcut ORI: `0.19886363636363635`
- Tuebingen decision: `tuebingen_external_direction_benchmark_ready`
- Tuebingen correlation-only direction overclaims: `79`
