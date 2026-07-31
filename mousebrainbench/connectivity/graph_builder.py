"""Conversion between validated connectivity and NetworkX graphs."""

from __future__ import annotations

import networkx as nx
import numpy as np

from mousebrainbench.schemas import BrainRegion, ConnectivityMatrix


def build_directed_graph(
    regions: list[BrainRegion],
    connectivity: ConnectivityMatrix,
    *,
    threshold: float = 0.0,
) -> nx.DiGraph:
    """Build a directed graph respecting weights[target, source] orientation."""

    by_id = {region.id: region for region in regions}
    expected_ids = set(connectivity.source_region_ids) | set(connectivity.target_region_ids)
    if expected_ids != set(by_id):
        raise ValueError("regions and connectivity IDs do not match")
    graph = nx.DiGraph()
    for region in regions:
        graph.add_node(region.id, acronym=region.acronym, name=region.name)
    weights = connectivity.dense_weights(copy=False)
    for target_index, source_index in np.argwhere(weights > threshold):
        graph.add_edge(
            connectivity.source_region_ids[source_index],
            connectivity.target_region_ids[target_index],
            weight=float(weights[target_index, source_index]),
        )
    graph.graph.update(connectivity.metadata)
    return graph
