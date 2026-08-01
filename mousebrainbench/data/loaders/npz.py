"""Versioned native interchange for connectivity fixtures and derived matrices."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from mousebrainbench.schemas import ConnectivityMatrix

FORMAT_VERSION = "1"


def save_connectivity_npz(connectivity: ConnectivityMatrix, path: str | Path) -> Path:
    """Save a connectivity matrix while preserving orientation and provenance."""

    destination = Path(path)
    np.savez_compressed(
        destination,
        format_version=FORMAT_VERSION,
        source_region_ids=np.asarray(connectivity.source_region_ids),
        target_region_ids=np.asarray(connectivity.target_region_ids),
        weights=connectivity.dense_weights(copy=False),
        metadata=json.dumps(connectivity.metadata),
    )
    return destination


def load_connectivity_npz(path: str | Path) -> ConnectivityMatrix:
    """Load the native format and reject unknown versions."""

    with np.load(path, allow_pickle=False) as data:
        version = str(data["format_version"])
        if version != FORMAT_VERSION:
            raise ValueError(f"unsupported connectivity format version: {version}")
        return ConnectivityMatrix(
            source_region_ids=tuple(int(value) for value in data["source_region_ids"]),
            target_region_ids=tuple(int(value) for value in data["target_region_ids"]),
            weights=data["weights"],
            metadata=json.loads(str(data["metadata"])),
        )
