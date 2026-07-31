"""Computational-cost metrics recorded beside scientific outputs."""

from __future__ import annotations

import numpy as np

from mousebrainbench.schemas import ConnectivityMatrix, SimulationState


def cost_metrics(state: SimulationState, connectivity: ConnectivityMatrix) -> dict[str, float | int]:
    return {
        "runtime_seconds": float(state.metadata["runtime_seconds"]),
        "n_regions": connectivity.n_regions,
        "n_edges": int(np.count_nonzero(connectivity.dense_weights(copy=False))),
        "n_state_variables": int(
            sum(values.shape[1] if values.ndim > 1 else 1 for values in state.variables.values())
            or connectivity.n_regions
        ),
        "simulated_time_units": float(state.time[-1] - state.time[0]),
        "simulation_time_ratio": float(
            (state.time[-1] - state.time[0]) / max(state.metadata["runtime_seconds"], 1e-12)
        ),
        "activity_memory_mb": float(state.activity.nbytes / 1024**2),
    }
