# Sensorium Official Trained Baseline Status

## What Was Integrated

We now have a bounded local training/evaluation path for the official
Sensorium ecosystem:

- official Sensorium mouse video loader;
- official `make_video_model` factory;
- official Factorized3dCore + factorized readout model family;
- MouseBrainBench comparator integration;
- audit artifact that separates "trained locally" from "Q1-qualified".

The executable artifact is:

```bash
MPLBACKEND=Agg MPLCONFIGDIR=.cache/matplotlib XDG_CACHE_HOME=.cache/xdg \
PYTHONPATH=external/neuralpredictors_43fa:external/sensorium_2023 \
.venv-sensorium-official/bin/python \
  scripts/train_sensorium_official_tiny_baseline.py --device auto
```

On Apple Silicon, `--device auto` selects `mps` when PyTorch can see the Metal
backend. In restricted execution sandboxes, MPS can be hidden even when the same
environment works from a normal terminal. The diagnostic command is:

```bash
.venv-sensorium-official/bin/python scripts/check_apple_mps.py
```

## Current Result

Output:

- `results/sensorium_official_baseline_audit/official_trained_baseline_summary.json`
- `results/dynamic_sensorium_model_comparator/summary.json`
- `results/sensorium_official_baseline_audit/summary.json`

Current run:

- usable mice: 5;
- training budget: 64 train batches, 24 oracle/eval batches per mouse;
- model: official architecture family, bounded local configuration
  (`hidden_channels=[8, 8]`, spatial kernel `5`, temporal kernel `5`);
- device: Apple Metal/MPS on the Mac M5;
- claim: reproducible integration and local control;
- blocked claim: official/SOTA Sensorium baseline.

The bounded official model improves over its matched mean-response baseline in
4/5 Dynamic Sensorium mice, with median delta `0.013391`. However, it remains
far below the tracked transparent temporal/SVD/MLP baselines in the current
comparator. This is useful negative evidence: the official stack is integrated,
but this bounded run is not the differential Q1 piece.

The MPS run confirms that the local Mac GPU can execute the official Sensorium
path. It does not by itself solve the publication-level baseline problem,
because the limiting factor is now model/training scale, not mere device
availability.

## Scientific Interpretation

This artifact proves that MouseBrainBench can execute an official Sensorium
model family end to end. It does not prove that we have matched the official
competition baseline, because the run deliberately uses a bounded local training
budget and a bounded model configuration.

For a Q1 claim, one of these is still required:

1. train the official baseline with the published/accepted budget and
   configuration, then evaluate it through MouseBrainBench; or
2. obtain an external official checkpoint/leaderboard-equivalent prediction set
   and evaluate it through MouseBrainBench.
