"""Command-line entry point for reproducible MouseBrainBench Phase 1 runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from mousebrainbench.artifacts import save_run
from mousebrainbench.config import (
    build_connectivity,
    build_model,
    build_perturbations,
    build_simulation_config,
    build_stimuli,
    load_config,
)
from mousebrainbench.connectivity.graph_builder import build_directed_graph
from mousebrainbench.simulation.runner import run_simulation
from mousebrainbench.validation.anatomical_metrics import (
    connectivity_density,
    graph_modularity,
    shortest_path_statistics,
)
from mousebrainbench.validation.cost_metrics import cost_metrics
from mousebrainbench.validation.functional_metrics import fc_correlation, functional_connectivity
from mousebrainbench.validation.perturbation_metrics import (
    affected_regions,
    mean_activity_change,
)


def execute(config: dict[str, Any]) -> Path:
    """Build, run, evaluate, and persist one configured experiment."""

    regions, connectivity = build_connectivity(config)
    model = build_model(config)
    simulation = build_simulation_config(config)
    stimuli = build_stimuli(config, regions)
    perturbations = build_perturbations(config, regions)
    baseline = run_simulation(model, connectivity, simulation, stimuli=stimuli)
    perturbed = run_simulation(
        model, connectivity, simulation, stimuli=stimuli, perturbations=perturbations
    )
    graph = build_directed_graph(regions, connectivity)
    baseline_fc = functional_connectivity(baseline.activity)
    perturbed_fc = functional_connectivity(perturbed.activity)
    threshold = float(config.get("analysis", {}).get("affected_region_threshold", 0.01))
    metrics = {
        "interpretation": "engineering_validation_only",
        "anatomical": {
            "density": connectivity_density(connectivity),
            "modularity": graph_modularity(graph),
            **shortest_path_statistics(graph),
        },
        "functional": {"baseline_perturbed_fc_correlation": fc_correlation(baseline_fc, perturbed_fc)},
        "perturbation": {
            "affected_region_ids": affected_regions(baseline, perturbed, threshold=threshold),
            "mean_absolute_activity_change": float(
                np.mean(np.abs(mean_activity_change(baseline, perturbed)))
            ),
        },
        "cost": cost_metrics(perturbed, connectivity),
    }
    run = config.get("run", {})
    run_dir = save_run(
        run.get("output_dir", "outputs"),
        run.get("name", "phase1"),
        config,
        connectivity,
        baseline,
        perturbed,
        metrics,
    )
    if run.get("save_plot", False):
        from mousebrainbench.visualization.plots import plot_activity

        plot_activity(perturbed, run_dir / "activity.png")
    return run_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    args = parser.parse_args()
    run_dir = execute(load_config(args.config))
    print(json.dumps({"run_dir": str(run_dir.resolve())}))


if __name__ == "__main__":
    main()
