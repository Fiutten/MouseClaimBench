# MouseClaimBench — KAIS journal adaptation (20 November 2026)

**Target:** *Knowledge and Information Systems* (Springer Nature), research article.

## Canonical single-source manuscript

- Main file: `MouseClaimBench_KAIS_single.tex` (exactly one active TeX manuscript, no `\\input` or `\\include`).
- References: `references.bib`; BibTeX style `unsrtnat` (numbered order of citation).
- Figures: `Fig1.pdf` … `Fig5.pdf` (all five original vector files, renamed in reference order).
- This branch was created for the KAIS adaptation; historical Elsevier sources and frozen experimental results on `main` have not been modified.
- The source uses the standard `article` class for **initial** KAIS review. KAIS explicitly permits a common LaTeX class if its older `kais.cls` is unsuitable. The Springer Nature `sn-jnl` template is recommended but **not used here**; no journal production typesetting is claimed.
- Source SHA-256: `a3c70243f13b401f717ea41c5e5fb452d9ba10cf40cb9f3a8e6e9e6dcbaa2056`.

## Build

From this directory on a standard TeX Live distribution:

```sh
pdflatex -interaction=nonstopmode -halt-on-error MouseClaimBench_KAIS_single.tex
bibtex MouseClaimBench_KAIS_single
pdflatex -interaction=nonstopmode -halt-on-error MouseClaimBench_KAIS_single.tex
pdflatex -interaction=nonstopmode -halt-on-error MouseClaimBench_KAIS_single.tex
```

Local build verified on 20 November 2026: 47 A4 pages, no undefined references, no missing bibliography keys, no overfull boxes, 62 unique LaTeX labels, 5 figures and 12 tables all explicitly cited in order. All fonts embedded. The compiled PDF SHA-256 was `d295eeceda1dac6ccf9b72ef63a3a6dcd44711d2185d14138f9663b8fea90666`.

## Submission status

**Submitted to Knowledge and Information Systems on 20 November 2026 (as reported by the corresponding author).** The KAIS manuscript number and portal acknowledgement have not yet been provided; the exact portal-uploaded files have not been independently compared with this prepared source. Coauthor-approval records have not been independently inspected. Experimental results were not rerun. The profile is author-defined, not consensus-validated, and its domain portability, adaptive-adversary resistance, human explanatory utility and trust-anchor assumptions have not been established. A companion MouseBrainBench manuscript is disclosed in the text, with clear distinction from the authorization architecture.

The submission event is recorded in [editorial history](../../docs/EDITORIAL_SUBMISSIONS.md). The [adaptations and residual risks](../../docs/KAIS_MANUSCRIPT_TRACEABILITY_20261120.md) and cover letter remain available for future editorial responses.

## Publisher requirements

- [KAIS submissions](https://link.springer.com/journal/10115/submission-guidelines)
- [KAIS FAQ — LaTeX and 15,000-word regular-paper ceiling](https://kais.zhonghuapu.com/FAQ.shtml)
- [Springer Nature current LaTeX guidance](https://www.springernature.com/gp/authors/campaigns/latex-author-support)

The 11,568 TeXcount text/header/caption words are a reproducible estimate; any full-page figure/table equivalents must still be included under KAIS editorial counting. The historical KAIS instructions for **accepted papers** differ from the current Springer **initial-submission** instructions; initial submissions here follow numbered references and the current Springer editorial checklist.
