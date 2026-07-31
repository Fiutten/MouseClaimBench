"""Validated data contracts shared by MouseBrainBench modules."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy import sparse

Matrix = np.ndarray | sparse.spmatrix


@dataclass(frozen=True)
class BrainRegion:
    """An anatomical region reference, not a neuron or voxel."""

    id: int
    acronym: str
    name: str
    parent_id: int | None = None
    structure_id_path: tuple[int, ...] | None = None
    volume_mm3: float | None = None
    centroid_um: tuple[float, float, float] | None = None
    hemisphere: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.acronym or not self.name:
            raise ValueError("region acronym and name must be non-empty")
        if self.volume_mm3 is not None and self.volume_mm3 < 0:
            raise ValueError("region volume must be non-negative")
        if self.hemisphere not in {None, "left", "right", "bilateral"}:
            raise ValueError("hemisphere must be left, right, bilateral, or None")


@dataclass(frozen=True)
class ConnectivityMatrix:
    """Directed weighted connectivity with explicit source/target orientation."""

    source_region_ids: tuple[int, ...]
    target_region_ids: tuple[int, ...]
    weights: Matrix
    delays: Matrix | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        expected = (len(self.target_region_ids), len(self.source_region_ids))
        if self.weights.shape != expected:
            raise ValueError(
                "weights must have shape (n_targets, n_sources); "
                f"expected {expected}, got {self.weights.shape}"
            )
        if self.delays is not None and self.delays.shape != expected:
            raise ValueError("delays must match weights shape")
        if len(set(self.source_region_ids)) != len(self.source_region_ids):
            raise ValueError("source region IDs must be unique")
        if len(set(self.target_region_ids)) != len(self.target_region_ids):
            raise ValueError("target region IDs must be unique")
        values = self.weights.data if sparse.issparse(self.weights) else np.asarray(self.weights)
        if not np.all(np.isfinite(values)):
            raise ValueError("weights must be finite")
        if self.metadata.get("weights_nonnegative", True) and np.any(values < 0):
            raise ValueError("weights must be non-negative unless metadata permits signed weights")

    @property
    def is_square(self) -> bool:
        return self.source_region_ids == self.target_region_ids

    @property
    def n_regions(self) -> int:
        if not self.is_square:
            raise ValueError("n_regions is defined only for square aligned connectivity")
        return len(self.source_region_ids)

    def dense_weights(self, *, copy: bool = True) -> np.ndarray:
        dense = self.weights.toarray() if sparse.issparse(self.weights) else np.asarray(self.weights)
        return dense.copy() if copy else dense


@dataclass(frozen=True)
class SimulationState:
    """Time-aligned simulation output; activity shape is time by region."""

    time: np.ndarray
    activity: np.ndarray
    region_ids: tuple[int, ...]
    variables: dict[str, np.ndarray] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.time.ndim != 1:
            raise ValueError("time must be one-dimensional")
        if self.activity.shape != (len(self.time), len(self.region_ids)):
            raise ValueError("activity must have shape (n_time_steps, n_regions)")
        if not np.all(np.isfinite(self.activity)):
            raise ValueError("activity contains non-finite values")
        for name, values in self.variables.items():
            if len(values) != len(self.time):
                raise ValueError(f"variable {name!r} is not aligned to time")
