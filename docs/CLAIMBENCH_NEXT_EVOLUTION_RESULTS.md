# ClaimBench next evolution results

Date: 2026-07-30

## External benchmark expansion

Two public external benchmarks are now operational as diagnostic adapters:

- SciFact dev for scientific claim verification.
- Tuebingen Cause-Effect Pairs for causal direction.

The purpose is not to claim SOTA on either benchmark. The purpose is to test
whether MouseBrainBench claim auditing exposes common shortcuts outside the
mouse-brain evidence stack.

## SciFact result

- Claims evaluated: 300.
- Gold labels: 124 SUPPORT, 64 CONTRADICT, 112 NOT_ENOUGH_INFO.
- Lexical shortcut false positives: 37.
- Lexical shortcut ORI: 0.210.
- Lexical shortcut CI: 0.492.
- Abstention rate under low-similarity threshold: 0.357.

Interpretation: SciFact is useful for the next paper. It shows that traceable
lexical overlap with cited evidence is not sufficient to authorize SUPPORT
claims. This strengthens the argument for executable evidence-to-claim
contracts. It should not be framed as a competitive SciFact verifier.

## Tuebingen result

- Pairs loaded: 108.
- Direction attempts: 97.
- Direction accuracy: 0.536.
- Weighted direction accuracy: 0.609.
- Correlation-only direction overclaims: 79.

Interpretation: Tuebingen is useful as an external direction-overclaiming
control. Correlation alone would authorize directional language in many cases,
but the transparent direction heuristic is only modestly accurate. Therefore,
the benchmark supports the claim-auditing argument, not a claim of strong causal
discovery performance.

## Current novelty position

The strongest line is now:

MouseBrainBench-ClaimAudit provides an executable claim-governance layer for
neurocomputational digital model validation, and it is externally stress-tested
on scientific claim verification and cause-effect direction benchmarks.

## Remaining gap for a stronger Q1 paper

The next technical upgrade should be one of:

1. Add a stronger external SciFact baseline, for example BM25-style retrieval
   plus deterministic evidence/rationale scoring.
2. Add a stronger Tuebingen causal-direction baseline, closer to published ANM
   or IGCI protocols.
3. Add automatic LaTeX/PDF claim extraction with paraphrase matching, so the
   framework audits real manuscript text rather than only declared wording.
4. Add measured runtime, memory, data footprint and claim-frontier plots.

Without one of these, the work is much stronger than before but still vulnerable
to the criticism that the external adapters are diagnostic rather than
competitive.
