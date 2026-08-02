# IBL behavioral v5 external validation

## Question and frozen population

This experiment asks whether the fixed topology-specific authorizer can
distinguish the trial-aligned visual stimulus from temporally shifted controls
in a real behavioral population. It does not test a neural causal mechanism.

The selection was frozen before behavioral tables were downloaded. It contains
110 previously unused IBL Brain-Wide Map mice, one insertion session per mouse,
from 12 laboratories. SHA-256 allocation produced 40 calibration-context mice,
35 risk-lock mice, and 35 final mice. Mouse is the inferential unit. Trials and
the four candidate alignments are dependent lower-level observations.

The official IBL convention is used without alteration. Signed contrast is
`100 * (contrastRight - contrastLeft)`, with missing-side contrast set to zero,
and `choice == -1` denotes a right choice. The model compares block-only choice
prediction with block plus candidate contrast in three deterministic held-out
folds. Candidate offsets are 0, 17, 43, and 89 trials. Only offset zero is the
task-aligned candidate.

## Acquisition amendment

The first catalog-only download pass found 54 official default tables with QC
`PASS` and 56 with QC `WARNING`. A PASS-only analysis would leave 17 risk-lock
mice and 21 final mice, below the frozen minimum of 29. Before any parquet table
was opened, protocol v5.3.1 was amended to admit both official default QC
classes, preserve QC as a stratum, and retain PASS-only results as a descriptive
sensitivity. Endpoint, evidence thresholds, candidate offsets, risk target, and
authorizer threshold did not change.

All 110 tables were subsequently resolved through Alyx and verified by dataset
UUID, revision, byte count, and MD5. They occupy 5.6 MB and contain at least 387
usable choices per mouse. No mouse was replaced.

## Results

| Role | Mice | False mouse events | Risk UCB | Coverage LCB | Recovery LCB | Certified |
|---|---:|---:|---:|---:|---:|---:|
| Calibration context | 40 | 0 | 0.0722 | 0.9278 | 0.9278 | Yes |
| Risk lock | 35 | 0 | 0.0820 | 0.9180 | 0.9180 | Yes |
| Final, opened after risk lock | 35 | 0 | 0.0820 | 0.9180 | 0.9180 | Yes |

The complete authorizer selected exactly one candidate for every risk-lock and
final mouse, always offset zero. Median held-out Tjur R-squared was 0.4803 in the
risk lock and 0.4399 in the final split. Median margins over the best shifted
candidate were 0.2759 and 0.2492, respectively.

This is positive external population evidence, but it is not an exclusive
method superiority result. The fixed 0.5 score comparator and the evidence
contract alone also selected only the true candidate and passed the same
endpoint. A correlation-only rule added after the primary result would falsely
select at least one shifted control for all 35 risk-lock and all 35 final mice.
That diagnostic explains why alignment specificity matters, but it is explicitly
post hoc and cannot be reported as a prespecified comparator.

## Valid interpretation

The result closes the previous absence of a real population with at least 29
biological top-level units. It supports transport of one fixed
topology-specific evidence contract to one standardized IBL behavioral task.
It does not establish independent laboratory replication because mice share a
consortium protocol and laboratory-level dependencies remain. The PASS-only
sensitivity has zero observed failures but is not certificate-eligible due to
insufficient mice. No causal-neural, whole-brain, behavioral-twin, or universal
risk claim follows from this experiment.

Reproduce with:

```bash
HOME="$PWD/data/external/ibl/home" \
  .venv-risk-v3/bin/python scripts/fetch_ibl_behavior_v5.py
.venv-risk-v3/bin/python -m \
  mousebrainbench.benchmarks.ibl_behavior_v5_confirmation
```

