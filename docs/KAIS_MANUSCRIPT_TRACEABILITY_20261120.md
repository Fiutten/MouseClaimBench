# Editorial provenance — DKE → KAIS, 20 November 2026

**Original manuscript:** `DATAK-D-26-02368`, *MouseClaimBench: A Knowledge-Based Architecture for Integrity-Aware Scientific Claim Authorization*; desk-rejected by *Data & Knowledge Engineering* for journal scope, with no external peer review.

**Provenance:** author-supplied ZIP SHA-256 `963113243c01512b7a68f05b59492c5e85e806d0baa3a43d2400cde29499596f`. Canonical source was its `main.tex`; `main_anonymous.tex` is another Elsevier-era entry point and is not mixed into the canonical manuscript. The 25 actually active TeX files were merged recursively; all `\\input` calls resolved. Exact merged KAIS source SHA-256: `40148d60556f3a7bdc6d28ba9b2c3aea141bf50f637aa9091cc63cc7f8b37f53`.

**Source-manifest anomaly:** **17 of 38 entries** in the archived `manifest.json` do not match the file contents actually present in that same ZIP. A new effective-hash manifest, discrepancy table, and original values are included in the local deliverable. This inconsistency is not hidden or assumed to indicate malicious alteration.

**Journal adaptation (no experiment rerun):**
1. Replaced Elsevier document wrapper with submission-permitted common LaTeX style, numerical bibliography, Springer-compatible statements/declarations.
2. Changed title and keywords toward knowledge-based non-compensatory evidence authorization.
3. Rewrote abstract to 203 words without changing reported experimental numbers.
4. Added explicit knowledge-systems novelty boundary and research questions RQ1–RQ3.
5. Retained 10 claim types, 22 contracts, 60 requirements, 124 observation slots, 5,677 cases, 55,031 property checks, 2,550 attack compositions, 20 coherent-forgery negative controls, DANDI and MICRONS bounded outcomes.
6. Treated the unpublished *MouseBrainBench* companion as an in-text disclosure, not as a published source; retained all five original figures, 12 tables, and the bibliography dataset.
7. Added an accurate AI-assisted editorial preparation disclosure. Three coauthors still need to approve the final manuscript.

**Checks:** locally compiled 46-page PDF with BibTeX and complete cross-references; no subordinate TeX files; 62 unique labels; no missing reference keys. `texcount`: approximately 11,462 text/header/caption words; journal full-page figure/table equivalents remain to be confirmed.

**Current repository:** `main` preserves the prior manuscript and experimental releases. A working branch `work/kais-single-source-20261120` was created for the KAIS work. This branch records the intended changes and provenance; do not claim the exact consolidated file is uploaded until verified.

**Unresolved scientific and editorial risks:** validity of the author-defined profile, distinction from prior assurance-case/SHACL/provenance systems, threat-model limitations, non-generalizability outside mouse-brain evidence, contemporaneous companion-paper status, and required author approval.

See [KAIS submission guidelines](https://link.springer.com/journal/10115/submission-guidelines) and [KAIS FAQ](https://kais.zhonghuapu.com/FAQ.shtml).
