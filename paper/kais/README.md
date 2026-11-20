# MouseClaimBench: KAIS single-source adaptation (20 November 2026)

This directory is reserved for the *Knowledge and Information Systems* (KAIS) adaptation of the complete DKE submission ZIP supplied by the authors on 20 November 2026.

## Verified local deliverable

- Canonical manuscript: `MouseClaimBench_KAIS_single.tex` (one active `.tex`, all 24 nested `\\input` dependencies merged).
- Manuscript SHA-256: `40148d60556f3a7bdc6d28ba9b2c3aea141bf50f637aa9091cc63cc7f8b37f53`.
- Source ZIP SHA-256: `963113243c01512b7a68f05b59492c5e85e806d0baa3a43d2400cde29499596f`.
- Local PDF: 46 pages; completed `pdflatex → bibtex → pdflatex → pdflatex` without missing bibliography or references.
- Abstract: 203 words; six keywords; numerical references; 5 PDF figures; 12 tables.
- No experiment was rerun or altered for journal adaptation.

## Status of GitHub synchronization

**Traceability registered; manuscript upload must be verified separately.** The exact local final source and related figures were supplied as a downloadable KAIS submission package. This directory does not by itself contain the verified single-source manuscript. Do not present it as ready to compile from GitHub until the files are copied here and compared against the SHA-256 above. Do not treat `main` as the KAIS canonical source: the current `paper/main.tex` and `paper/sections/` differ from the later DKE ZIP.

When synchronizing, place here exactly one active `.tex` (`MouseClaimBench_KAIS_single.tex`), `references.bib`, and the five PDF figures `standards_workflow.pdf`, `claim_evidence_matrix_v2.pdf`, `integrity_ablation_v2.pdf`, `scalability_v2.pdf` and `prospective_applications_v2.pdf`. Keep archival Elsevier sources under their original paths.

## Review gates before KAIS submission

The profile is literature-grounded but author-defined, and its content has not been independently validated. Controlled attacks do not establish adaptive-adversary robustness; internally coherent forgery controls show the absence of an external trust anchor. Cross-domain empirical generality is not demonstrated. Coauthors must approve the revised text, authorship, declarations and AI-assisted editorial disclosure before submission.

## Official KAIS sources

- https://link.springer.com/journal/10115/submission-guidelines
- https://kais.zhonghuapu.com/FAQ.shtml
- https://kais.zhonghuapu.com/main.3.shtml

Official regular-paper word limit: 15,000 words; full-page figures/tables count as 500 words each. The initial submission may use a common LaTeX class, although Springer Nature's template is recommended.
