"""Load a provenance-rich regional matrix derived from Allen tracer experiments."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from mousebrainbench.schemas import BrainRegion, ConnectivityMatrix


def load_allen_visual_connectivity(
    path: str | Path,
    region_acronyms: tuple[str, ...],
) -> tuple[list[BrainRegion], ConnectivityMatrix, dict[str, Any]]:
    """Load and reorder a derived Allen regional matrix as weights[target, source]."""

    source = Path(path)
    payload = json.loads(source.read_text())
    stored = tuple(item["acronym"] for item in payload["regions"])
    missing = set(region_acronyms) - set(stored)
    if missing:
        raise ValueError(f"connectivity is missing requested regions: {sorted(missing)}")
    positions = [stored.index(acronym) for acronym in region_acronyms]
    weights = np.asarray(payload["weights_target_by_source"], dtype=float)
    weights = weights[np.ix_(positions, positions)]
    if not np.allclose(np.diag(weights), 0.0):
        raise ValueError("Allen visual connectivity diagonal must be zero")
    selected = [payload["regions"][index] for index in positions]
    regions = [
        BrainRegion(
            id=int(item["id"]),
            acronym=str(item["acronym"]),
            name=str(item["name"]),
            hemisphere="bilateral",
            metadata={"source": "Allen Mouse Brain Connectivity Atlas"},
        )
        for item in selected
    ]
    ids = tuple(region.id for region in regions)
    metadata = {
        "source": "Allen Mouse Brain Connectivity Atlas tracer experiments",
        "biological": True,
        "scale": "regional_mesoscopic",
        "orientation": "weights[target, source]",
        "aggregation": payload["aggregation"],
        "diagonal": payload["diagonal"],
        "source_path": str(source.resolve()),
        "experiment_counts_by_source": {
            acronym: payload["experiment_counts_by_source"][acronym]
            for acronym in region_acronyms
        },
    }
    return regions, ConnectivityMatrix(ids, ids, weights, metadata=metadata), payload
