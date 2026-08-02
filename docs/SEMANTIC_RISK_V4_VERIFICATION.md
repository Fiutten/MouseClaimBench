# Semantic risk v4 verification record

Verification date: 2026-08-02.

## Executed checks

- Full test suite: 182 tests passed.
- Expected warning: MAPIE reports that no risk-controlling threshold exists in
  the dedicated uncertifiable-policy test. The test passes because abstention is
  the required behavior.
- `python -m compileall mousebrainbench scripts`: passed.
- Ruff on every v4 module, downloader, and v4 test: passed.
- Artifact provenance: every input to the v4 release audit records a clean Git
  revision without the `-dirty` suffix.
- External files: twelve official archives were checked against the MD5 values
  frozen from the Causal Chambers directory before extraction.

## Repository-wide lint boundary

A repository-wide Ruff scan reports 198 findings in legacy modules and tests.
Most are import ordering, old style rules, or pre-existing formatting findings.
They are not introduced by v4. A bulk rewrite was intentionally excluded from
this scientific branch because it would create unrelated code churn and obscure
the provenance of the experimental change. This is declared technical debt, not
a claim that the complete historical repository is lint-clean.

## Scientific release decision

The eight-point implementation is complete. The release audit is deliberately
non-compensatory. It records a positive prospective v4.2 router result but a
negative external authorization result and absent independent profile-content
validation. Therefore `strong_q1_second_paper_ready` is false.

