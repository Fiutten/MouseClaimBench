# Knowledge-profile internal AI-assisted audit

## Status and governance

This audit was performed by Codex while working with the project authors. It is
an internal, author-model-assisted adversarial review. It created no participant,
expert identity, rating panel, CVI, agreement statistic, or human-consensus
result. It cannot count toward the frozen external panel.

This distinction is methodological rather than cosmetic. Content validity asks
whether content is relevant, comprehensive, and comprehensible for its construct
and context of use. Current COSMIN guidance treats an independent sample of
professionals as a separate source of content-validity evidence. An AI system
working inside the author team is not such a sample:

- https://pmc.ncbi.nlm.nih.gov/articles/PMC5891557/
- https://www.cosmin.nl/wp-content/uploads/COSMIN-manual-V2_final.pdf

## Method

All 29 frozen review items were assessed for source traceability, construct
alignment, terminology, inference safety, and comprehensiveness. Every declared
source identifier was resolved against the bibliography or the review protocol.
The executable rule structure was proved over a 9,766,290-state analytical
space. The engine was also executed on 11,419 complete or boundary status
combinations. No non-all-passed state produced profile support.

The structural test establishes implementation safety only. It cannot establish
that the required blocks are scientifically sufficient or comprehensive.

## Result

| Decision | Items |
|---|---:|
| Retain | 9 |
| Minor revision | 8 |
| Major revision | 11 |
| Critical veto | 1 |

The decision is `internal_ai_audit_revision_required`.

The critical veto concerns `rule__all_requirements_satisfied`. Its exported
conclusion is `supported`, while the curated basis repeatedly describes
relations as sufficient only for profile authorization or not scientifically
sufficient. The wording can therefore promote internal contract satisfaction
into apparent scientific support. The recommended correction is
`profile_supported`, accompanied by profile identity and context of use in every
exported decision.

Three cross-item weaknesses also require major revision:

1. The mechanistic claim combines prediction, internal reproduction, topology
   specificity, and direction. This does not establish mechanism without
   perturbational discrimination, identifiability, uncertainty, and comparison
   with competing mechanisms. Recent mechanistic-model guidance likewise treats
   identifiability and non-calibration validation as necessary components:
   https://pmc.ncbi.nlm.nih.gov/articles/PMC11442102/
2. The generic `digital_twin` identifier is evaluated as a complete, causal,
   whole-brain, operational, entity-specific twin. This is conservative but
   constructually ambiguous. Digital-twin validation is context-of-use specific,
   and the literature does not supply one universal definition:
   https://doi.org/10.1080/00207543.2024.2357741
3. Mixed `failed`, `requires_review`, `unknown`, and `not_applicable` states are
   resolved through a normative priority order. The software follows it exactly,
   but experts have not validated that precedence.

The review also tightens external replication. A separate resource is not
independent if it republishes the same underlying observations. NASEM defines
replication around a new study with newly collected data addressing the same or
a similar question:
https://www.nationalacademies.org/read/25303/chapter/6

## Consequence

The profile must remain `author-proposed, literature-grounded, internally
audited, and not externally content-validated`. The internal audit is useful
because it reveals concrete revisions before recruiting experts. It does not
close the external-validation requirement and must not be reported as if seven
independent experts had agreed.

Reproduce with:

```bash
.venv-risk-v3/bin/python -m \
  mousebrainbench.benchmarks.knowledge_profile_internal_ai_audit
```

