# LLM Claim Extraction Audit

- Decision: `llm_claim_extraction_layer_ready_non_authoritative`
- Mode: `deterministic_fallback_with_optional_llm_file`
- LLM API called: `False`
- LLM authoritative: `False`
- Candidates: `120`
- Candidate counts by type: `{'mechanistic': 43, 'structure_function': 7, 'digital_twin': 8, 'causal': 10, 'prediction': 41, 'sota': 2, 'reproducibility': 6, 'generalization': 3}`

## Boundary

The LLM layer is limited to candidate extraction and conservative wording support. Claim authorization remains controlled by executable ClaimBench artifacts.
