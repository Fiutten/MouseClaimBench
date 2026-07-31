"""Synthetic engineering fixtures that must never be described as biological."""

from __future__ import annotations

import numpy as np

from mousebrainbench.schemas import BrainRegion, ConnectivityMatrix


def generate_synthetic_connectivity(
    n_regions: int,
    *,
    density: float = 0.1,
    seed: int = 0,
    include_named_regions: bool = True,
) -> tuple[list[BrainRegion], ConnectivityMatrix]:
    """Generate a reproducible directed fixture for software validation."""

    if n_regions < 2:
        raise ValueError("n_regions must be at least 2")
    if not 0 < density <= 1:
        raise ValueError("density must fall in (0, 1]")
    rng = np.random.default_rng(seed)
    weights = rng.lognormal(mean=-2.0, sigma=0.8, size=(n_regions, n_regions))
    weights *= rng.random((n_regions, n_regions)) < density
    np.fill_diagonal(weights, 0.0)
    acronyms = [f"R{i:03d}" for i in range(n_regions)]
    if include_named_regions:
        acronyms[0] = "VISp"
        acronyms[1] = "TH"
    regions = [
        BrainRegion(
            id=index + 1,
            acronym=acronym,
            name=f"Synthetic region {acronym}",
            parent_id=None,
            metadata={"source": "synthetic_engineering_fixture", "biological": False},
        )
        for index, acronym in enumerate(acronyms)
    ]
    ids = tuple(region.id for region in regions)
    connectivity = ConnectivityMatrix(
        source_region_ids=ids,
        target_region_ids=ids,
        weights=weights,
        metadata={
            "source": "synthetic_engineering_fixture",
            "biological": False,
            "seed": seed,
            "density_requested": density,
            "orientation": "weights[target, source]",
        },
    )
    return regions, connectivity
