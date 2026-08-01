# SciFact Claim Verification Adapter

- Decision: `scifact_external_claim_audit_ready`
- Claims: `300`
- Label counts: `{'NOT_ENOUGH_INFO': 112, 'SUPPORT': 124, 'CONTRADICT': 64}`
- Shortcut FPR: `0.199`
- Shortcut FNR: `0.484`
- BM25 evidence recall@5: `0.899`
- BM25/rationale FPR: `0.250`
- BM25/rationale FNR: `0.492`
- Train-calibrated baseline available: `True`
- Train-calibrated baseline FPR: `0.068`
- Train-calibrated baseline FNR: `0.718`
- Abstention rate: `0.363`
- Abstaining FPR: `0.393`
- Runtime seconds: `15.128`

Interpretation: BM25/rationale is a transparent local evidence-retrieval baseline. It is used to separate retrieval, support classification, and overclaiming risk; it is not a SciFact SOTA system.
