# Direction-router association precondition audit

- Decision: `precondition_removes_archived_spurious_attempts`
- Removed attempts: `450`

| Router | Coverage | Attempted accuracy | Spurious attempts |
|---|---:|---:|---:|
| Frozen v3 | 0.5000 | 0.8244 | 450 |
| Association precondition | 0.4375 | 0.9422 | 0 |

## Limits

- The no-association regime was identified after the primary v3 result.
- Association status is supplied by synthetic ground truth, not estimated from data.
- The repair requires a new frozen protocol and fresh cases before confirmatory use.
