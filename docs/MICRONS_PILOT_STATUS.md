# MICrONS Structure-Function Pilot Status

## Current State

The project now has a reproducible MICrONS small-pilot path, but the pilot is
not approved yet.

Downloaded small static files:

- functional EM co-registration:
  `func_unit_em_match_release.csv`;
- proofreading status:
  `proofreading_status_public_release.csv`;
- digital twin v2 functional/anatomical unit properties:
  `dt_anatomy_units.csv`, `dt_performance_units.csv`,
  `dt_ori_dir_tuning_units.csv`.

Generated artifacts:

- `results/microns_pilot_gate/static_pilot_diagnostic.json`;
- `results/microns_pilot_gate/summary.json`;
- local ignored cache under `data/microns/static_small/`.

Rebuild command:

```bash
scripts/download_microns_static_small.sh
.venv/bin/python scripts/build_microns_static_pilot_manifest.py
.venv/bin/python scripts/query_microns_cave_pilot_edges.py
.venv/bin/python scripts/run_microns_structure_function_pilot.py
.venv/bin/python -m mousebrainbench.benchmarks.microns_pilot_gate
```

## Result

The static subset provides:

- 200 co-registered EM/function rows;
- 172 rows with usable functional properties;
- spatial coordinates;
- functional metrics from the digital twin property export;
- root IDs for candidate EM objects.

It does not provide:

- true synaptic edges among the matched units.

The current gate therefore returns:

```text
defer_microns_pilot_manifest_insufficient
```

## Why It Is Blocked

A defensible structure-function pilot requires real structural edges. The small
static files do not include those edges. The full static minnie65 synapse graph
is approximately 47.5 GB, which violates the bounded-pilot rule. The correct
small route is CAVE:

```bash
.venv/bin/python scripts/query_microns_cave_pilot_edges.py
```

That script calls:

```python
client.materialize.synapse_query(pre_ids=roots, post_ids=roots)
```

Current environment result after accepting CAVE terms:

```text
materialization_version=117 returns a real bounded edge set.
```

## Approval Rule

The MICrONS Q1-scale pilot becomes approved only when the manifest has:

- at least 500 neurons;
- spatial coordinates;
- functional responses/properties;
- real structural edges;
- estimated download size no greater than 5 GB.

The project also distinguishes a smaller micro-pilot:

- at least 100 neurons;
- spatial coordinates;
- functional responses/properties;
- real structural edges;
- estimated download size no greater than 1 GB.

The current static subset has enough co-registered units for a micro-pilot, but
requires CAVE queried at `materialization_version=117`. Newer/default CAVE
materializations only returned one edge for the static root IDs, while v117
returned a usable micro-pilot edge set.

## Structure-Function Analysis

When `data/microns/static_small/cave_synaptic_edges.csv` exists, the analysis in
`scripts/run_microns_structure_function_pilot.py` tests whether connected pairs
have higher functional similarity than:

- random unconnected pairs;
- distance-matched unconnected pairs.

The distance-matched null is mandatory. Without it, any positive result could be
explained by local spatial proximity rather than wiring-function coupling.

## Current Result

With CAVE terms accepted and `materialization_version=117`, the current local
micro-pilot has:

- 172 functionally usable co-registered units;
- 82 returned synapses;
- 41 unique directed connected pairs among usable units;
- negative/inconclusive structure-function result after random and
  distance-matched nulls.

This is scientifically useful as a stress test of the MouseBrainBench gate, but
it is not a Q1-enabling positive MICrONS result. The next MICrONS route must
expand the candidate set or use a richer functional/structural subset.
