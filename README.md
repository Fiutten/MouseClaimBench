# MouseClaimBench

MouseClaimBench is the standalone repository for the second manuscript derived
from the MouseBrainBench workstream.

The target contribution is not a new mouse-brain simulator and not a complete
digital mouse brain. The project implements and documents ClaimBench v2, an
executable decision-support framework for scientific AI claim governance.

ClaimBench separates:

- prediction,
- evidence retrieval,
- topology specificity,
- causal direction,
- uncertainty,
- manuscript wording,
- cost and fidelity,
- release reproducibility,
- non-authoritative LLM-assisted claim extraction.

The LLM layer is intentionally non-authoritative. The reproducible mode does not
call an external LLM API. Candidate claims can be extracted deterministically or
loaded from a local JSON file, but final claim authorization remains tied to
executable artifacts.

## Manuscript

The current manuscript draft is located in:

```text
paper/main.tex
```

For Overleaf, use `paper/main.tex` as the main document.

## Repository status

This repository intentionally excludes:

- the first MouseBrainBench paper,
- virtual environments,
- raw datasets,
- historical output folders,
- the legacy cognitive organism prototype.

The included `results/` directory contains lightweight frozen artifacts needed
to support the ClaimBench v2 manuscript draft. Numerical artifacts should not be
edited manually.

## Validation

Recommended local checks:

```bash
python -m compileall mousebrainbench scripts
python -m pytest -q
python -m ruff check .
```

If optional datasets or dependencies are missing, failures should be reported
explicitly rather than hidden.

## Claim boundary

The repository supports a bounded methodological claim:

ClaimBench v2 is a decision-support layer for auditing scientific AI claims by
mapping manuscript statements to explicit evidence contracts and reproducible
artifacts.

It does not support claims of:

- state-of-the-art prediction,
- causal discovery performance,
- complete mouse-brain simulation,
- complete biological digital twin,
- LLM-based scientific validation.

