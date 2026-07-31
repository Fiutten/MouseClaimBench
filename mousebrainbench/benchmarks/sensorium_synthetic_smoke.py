"""Generate a Sensorium-style smoke benchmark with known predictive structure."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from mousebrainbench.benchmarks.sensorium_predictive_mis import (
    DEFAULT_OUTPUT,
    run_sensorium_benchmark,
)
from mousebrainbench.data.loaders.sensorium import SensoriumTrialTable


def make_synthetic_sensorium_table(
    *,
    seed: int = 31,
    n_train: int = 90,
    n_repeated_stimuli: int = 15,
    repeats: int = 2,
    n_features: int = 7,
    n_neurons: int = 24,
) -> SensoriumTrialTable:
    """Create a compact Sensorium-like table with stimulus-driven responses."""

    rng = np.random.default_rng(seed)
    n_eval = n_repeated_stimuli * repeats
    train_x = rng.normal(size=(n_train, n_features))
    eval_unique_x = rng.normal(size=(n_repeated_stimuli, n_features))
    eval_x = np.repeat(eval_unique_x, repeats, axis=0)
    weights = rng.normal(scale=0.8, size=(n_features, n_neurons))
    neuron_bias = rng.normal(scale=0.2, size=(n_neurons,))
    train_y = train_x @ weights + neuron_bias + rng.normal(scale=0.25, size=(n_train, n_neurons))
    eval_y = eval_x @ weights + neuron_bias + rng.normal(scale=0.25, size=(n_eval, n_neurons))
    tiers = np.asarray(["train"] * n_train + ["validation"] * n_eval)
    stimulus_ids = np.concatenate(
        [np.arange(n_train), n_train + np.repeat(np.arange(n_repeated_stimuli), repeats)]
    )
    return SensoriumTrialTable(
        root=Path("synthetic_sensorium_smoke"),
        modality="dynamic",
        stimulus_features=np.vstack([train_x, eval_x]),
        context_features=np.empty((n_train + n_eval, 0), dtype=float),
        responses=np.vstack([train_y, eval_y]),
        tiers=tiers,
        trial_ids=np.arange(n_train + n_eval),
        stimulus_ids=stimulus_ids,
        neuron_ids=np.arange(n_neurons),
    )


def run(output: str | Path = DEFAULT_OUTPUT, *, git_revision: str | None = None) -> Path:
    """Run the Sensorium predictive/MIS smoke benchmark."""

    return run_sensorium_benchmark(
        make_synthetic_sensorium_table(), output=output, git_revision=git_revision
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--git-revision", default=None)
    args = parser.parse_args()
    print(json.dumps({"output": str(run(args.output, git_revision=args.git_revision).resolve())}))


if __name__ == "__main__":
    main()
