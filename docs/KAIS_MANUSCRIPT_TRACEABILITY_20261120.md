# MouseClaimBench — DKE to KAIS source and editorial provenance

**Audit date:** 2026-11-20

## Trigger

DKE manuscript `DATAK-D-26-02368` was desk-rejected for journal scope; this was not external peer review and should not be represented as a negative scientific assessment.

## Exact provenance

- Author-supplied full LaTeX ZIP SHA-256: `963113243c01512b7a68f05b59492c5e85e806d0baa3a43d2400cde29499596f`.
- Original source entry point: `main.tex` in the ZIP. The older `main_anonymous.tex` was excluded from active source merging, not arbitrarily reconciled.
- 25 active original TeX files recursively consolidated into one `paper/kais/MouseClaimBench_KAIS_single.tex`.
- Final TeX SHA-256: `a3c70243f13b401f717ea41c5e5fb452d9ba10cf40cb9f3a8e6e9e6dcbaa2056`.
- Final PDF SHA-256 (local artifact): `d295eeceda1dac6ccf9b72ef63a3a6dcd44711d2185d14138f9663b8fea90666`.
- Source-manifest anomaly: **17 of the 37 archived file-hash entries** in `manifest.json` do not match the contents of the authors' ZIP. This discrepancy has not been silently normalized or attributed to misconduct. Effective source hashes and the discrepancy table are supplied in the offline QA/handoff archive.

## Editorial modifications, without changing reported experimental results

1. Removed Elsevier `elsarticle` formatting; used a standard LaTeX review class **explicitly permitted for initial KAIS submission**. Do not claim to use the Springer Nature `sn-jnl` typesetting template or historical `kais.cls`.
2. Revised title, six keywords, abstract (194 words), introduction contribution framing, evidence authorization limits and RQs to foreground knowledge-based reasoning.
3. Preserved all formal equations, 10 bounded claim types, 22 predicates, 60 evidence requirements, 124 observation slots, 5,677 contract cases, 55,031 property checks, 2,550 attacked packages, 20 coherent-forgery boundary controls, and DANDI/MICRONS bounded cases.
4. Kept the original `references.bib`; citations are numeric in appearance/order with complete DOI URLs. The unpublished companion is identified transparently in the main text instead of incorrectly presented as a published paper.
5. Reused the exact original five PDF figure objects under the sequential names `Fig1.pdf` through `Fig5.pdf`. Springer-style captions `Fig. 1` etc; removed caption-terminal full stops.
6. Kept all 12 tables. Corrected a table-cell overlap, identified two figures and four tables previously lacking explicit cross-references and repaired them; the first textual citations of all five figures and twelve tables now follow declaration order.
7. Corrected heading hierarchy (no active `\\paragraph` levels) and placed an accurate disclosure of AI-assisted editorial/consolidation work under Methods, where Springer editorial policies request it.
8. Added applicable ethics/consent statement and kept original funding/author roles/competing-interest/data-and-code disclosure. Coauthor approval remains an external pre-submission gate, not a fabricated claim in the paper.
9. Rebuilt PDF with BibTeX and repeated LaTeX passes; inspected representative title, dense text, tables, figures and reference pages.

## Local verification

- 47 A4 pages; TeX source `121,039` characters (subject to confirmed final build).
- 194-word abstract, 6 keywords.
- 5 PDF figures, 12 tables, 62 unique labels; all figures and tables explicitly cited in order.
- 54 used BibTeX entries, 0 missing keys, no undefined cross-references or acronyms.
- Zero overfull/underfull box warnings; PDF has embedded fonts.
- Estimated 11,568 text/header/caption words via `texcount`, excluding the KAIS convention of 500 words per full-page figure/table. Editorial interpretation of the complete 15,000-word budget must be confirmed if needed.
- No test suite or scientific experiment rerun during editorial adaptation.

## Scientific limitations to retain

The profile's scientific content remains author-curated, not independently expert validated or community consensus; Python/ASP equivalence and SHACL validation concern executable policy semantics and structure, not scientific truth. The controlled attacks do not establish adaptive attacker security; 20 self-consistent forgeries pass without an external trust anchor. MICRONS holdouts are internal to one volume, not biological replication; DANDI positive predictive validation is bounded; no human decision-quality or cross-domain portability evaluation is presented.

The paper and its companion **must remain distinguishable**: five frozen MouseBrainBench result packages are retrospective authorization inputs, not new cross-domain external validation.

## Pre-submission checks

Obtain coauthor approval for authorship, funding, declarations, manuscript text and AI-assisted editing disclosure; confirm the exact companion submission status and code/data links; verify current KAIS website word-count and manuscript policies at submission date. **Do not submit solely because the source compiles.**

Official: https://link.springer.com/journal/10115/submission-guidelines
