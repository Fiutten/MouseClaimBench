# SciFact Claim Verification Adapter

- Decision: `scifact_external_claim_audit_ready`
- Claims: `300`
- Label counts: `{'NOT_ENOUGH_INFO': 112, 'SUPPORT': 124, 'CONTRADICT': 64}`
- Shortcut ORI: `0.199`
- Shortcut CI: `0.484`
- BM25 evidence recall@5: `0.899`
- BM25/rationale ORI: `0.250`
- BM25/rationale CI: `0.492`
- Abstention rate: `0.363`
- Abstaining ORI: `0.393`
- Runtime seconds: `4.219`

Interpretation: BM25/rationale is a transparent local evidence-retrieval baseline. It is used to separate retrieval, support classification, and overclaiming risk; it is not a SciFact SOTA system.
