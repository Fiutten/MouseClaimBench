"""Explicit connectivity normalization methods."""

from __future__ import annotations

from enum import StrEnum

import numpy as np

from mousebrainbench.schemas import ConnectivityMatrix


class Normalization(StrEnum):
    NONE = "none"
    ROW_SUM = "row_sum"
    COLUMN_SUM = "column_sum"
    SPECTRAL_RADIUS = "spectral_radius"


def normalize_connectivity(
    connectivity: ConnectivityMatrix, method: Normalization | str
) -> ConnectivityMatrix:
    """Return normalized connectivity without mutating the input."""

    method = Normalization(method)
    weights = connectivity.dense_weights()
    if method is Normalization.ROW_SUM:
        denominator = weights.sum(axis=1, keepdims=True)
        weights = np.divide(weights, denominator, out=np.zeros_like(weights), where=denominator > 0)
    elif method is Normalization.COLUMN_SUM:
        denominator = weights.sum(axis=0, keepdims=True)
        weights = np.divide(weights, denominator, out=np.zeros_like(weights), where=denominator > 0)
    elif method is Normalization.SPECTRAL_RADIUS:
        radius = float(np.max(np.abs(np.linalg.eigvals(weights))))
        if radius > 0:
            weights = weights / radius
    metadata = {**connectivity.metadata, "normalization": method.value}
    return ConnectivityMatrix(
        connectivity.source_region_ids,
        connectivity.target_region_ids,
        weights,
        connectivity.delays,
        metadata,
    )
