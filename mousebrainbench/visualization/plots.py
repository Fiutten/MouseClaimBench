"""Small plotting helpers that consume persisted simulation contracts."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt

from mousebrainbench.schemas import SimulationState


def plot_activity(state: SimulationState, path: str | Path, *, max_regions: int = 12) -> Path:
    """Save a compact regional activity overview."""

    destination = Path(path)
    figure, axis = plt.subplots(figsize=(10, 5))
    axis.plot(state.time, state.activity[:, :max_regions])
    axis.set(xlabel="Time", ylabel="Activity", title="Regional activity")
    figure.tight_layout()
    figure.savefig(destination, dpi=150)
    plt.close(figure)
    return destination
