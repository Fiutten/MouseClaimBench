"""Structural graph metrics with explicit definitions."""

from __future__ import annotations

import networkx as nx
import numpy as np

from mousebrainbench.schemas import ConnectivityMatrix


def connectivity_density(connectivity: ConnectivityMatrix) -> float:
    weights = connectivity.dense_weights(copy=False)
    possible = weights.size - min(weights.shape)
    return float(np.count_nonzero(weights) / possible) if possible else 0.0


def in_out_degree_distribution(connectivity: ConnectivityMatrix) -> dict[str, np.ndarray]:
    weights = connectivity.dense_weights(copy=False)
    return {
        "in_degree": np.count_nonzero(weights, axis=1),
        "out_degree": np.count_nonzero(weights, axis=0),
    }


def region_strength(connectivity: ConnectivityMatrix) -> dict[str, np.ndarray]:
    weights = connectivity.dense_weights(copy=False)
    return {"in_strength": weights.sum(axis=1), "out_strength": weights.sum(axis=0)}


def weight_distribution(connectivity: ConnectivityMatrix) -> np.ndarray:
    weights = connectivity.dense_weights(copy=False)
    return weights[weights != 0]


def shortest_path_statistics(graph: nx.DiGraph) -> dict[str, float]:
    """Compute finite unweighted path statistics; disconnected pairs are omitted."""

    lengths = [
        distance
        for source, targets in nx.all_pairs_shortest_path_length(graph)
        for target, distance in targets.items()
        if source != target
    ]
    return {
        "mean_finite_path_length": float(np.mean(lengths)) if lengths else 0.0,
        "reachable_pairs": float(len(lengths)),
    }


def graph_modularity(graph: nx.DiGraph) -> float:
    """Weighted modularity of the undirected projection used as a descriptive metric."""

    undirected = graph.to_undirected()
    if undirected.number_of_edges() == 0:
        return 0.0
    communities = nx.community.greedy_modularity_communities(undirected, weight="weight")
    return float(nx.community.modularity(undirected, communities, weight="weight"))


def hemispheric_symmetry(
    connectivity: ConnectivityMatrix,
    homologous_pairs: tuple[tuple[int, int], ...],
) -> float:
    """Correlate homologous left/right connectivity submatrices.

    Pair order is explicit because acronym-based hemisphere inference is unsafe.
    """

    if len(homologous_pairs) < 2:
        raise ValueError("at least two explicit homologous pairs are required")
    index = {region_id: position for position, region_id in enumerate(connectivity.source_region_ids)}
    left = [index[pair[0]] for pair in homologous_pairs]
    right = [index[pair[1]] for pair in homologous_pairs]
    weights = connectivity.dense_weights(copy=False)
    correlation = np.corrcoef(weights[np.ix_(left, left)].ravel(), weights[np.ix_(right, right)].ravel())[0, 1]
    return float(np.nan_to_num(correlation, nan=0.0))
