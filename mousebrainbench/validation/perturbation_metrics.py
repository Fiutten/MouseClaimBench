"""Metrics comparing baseline and perturbed simulations."""

from __future__ import annotations

import numpy as np
import networkx as nx

from mousebrainbench.schemas import SimulationState


def mean_activity_change(baseline: SimulationState, perturbed: SimulationState) -> np.ndarray:
    """Difference between whole-trajectory regional means."""

    if baseline.activity.shape != perturbed.activity.shape:
        raise ValueError("simulations must have matching activity shapes")
    return perturbed.activity.mean(axis=0) - baseline.activity.mean(axis=0)


def affected_regions(
    baseline: SimulationState, perturbed: SimulationState, *, threshold: float
) -> tuple[int, ...]:
    changes = np.abs(mean_activity_change(baseline, perturbed))
    return tuple(region_id for region_id, change in zip(baseline.region_ids, changes) if change > threshold)


def trajectory_difference(baseline: SimulationState, perturbed: SimulationState) -> np.ndarray:
    """Absolute activity difference through time for every region."""

    if baseline.activity.shape != perturbed.activity.shape:
        raise ValueError("simulations must have matching activity shapes")
    return np.abs(perturbed.activity - baseline.activity)


def recovery_time(
    baseline: SimulationState,
    perturbed: SimulationState,
    *,
    after: float,
    tolerance: float,
) -> float | None:
    """First time after an intervention where global error remains below tolerance."""

    differences = trajectory_difference(baseline, perturbed).mean(axis=1)
    candidates = np.flatnonzero(baseline.time >= after)
    for index in candidates:
        if np.all(differences[index:] <= tolerance):
            return float(baseline.time[index] - after)
    return None


def propagation_distance(
    graph: nx.DiGraph,
    source_region_ids: tuple[int, ...],
    affected_region_ids: tuple[int, ...],
) -> float:
    """Mean shortest directed hop distance from any perturbed source."""

    distances = []
    for affected in affected_region_ids:
        candidates = [
            nx.shortest_path_length(graph, source, affected)
            for source in source_region_ids
            if nx.has_path(graph, source, affected)
        ]
        if candidates:
            distances.append(min(candidates))
    return float(np.mean(distances)) if distances else 0.0
