import networkx as nx
import numpy as np

from mousebrainbench.benchmarks.causalrivers_v5_transport import (
    BlockData,
    matched_transition,
    select_graph_pairs,
)


def test_graph_pair_selection_is_balanced_deterministic_and_truth_blind() -> None:
    graph = nx.DiGraph()
    graph.add_nodes_from(range(8))
    graph.add_edges_from((index, index + 1) for index in range(7))
    columns = tuple(str(index) for index in range(8))
    first = select_graph_pairs(
        graph, columns, namespace="frozen", direct_count=4, control_count=4
    )
    second = select_graph_pairs(
        graph, columns, namespace="frozen", direct_count=4, control_count=4
    )
    assert first == second
    assert sum(pair[2] for pair in first) == 4
    assert len(first) == 8


def _block(block_id: str, pair_ids: tuple[str, ...]) -> BlockData:
    records = tuple({"pair_id": pair_id} for pair_id in pair_ids)
    count = len(records)
    return BlockData(
        block_id=block_id,
        role="test",
        records=records,
        scores=np.zeros(count),
        labels=np.ones(count, dtype=bool),
        admissible=np.ones(count, dtype=bool),
        features=np.zeros((count, 2)),
        selected_pairs=(),
        sampled_rows=10,
        source_rows=10,
        loaded_columns=2,
        exclusions={},
    )


def test_matched_transition_uses_only_identical_pairs() -> None:
    historical = _block("historical", ("a->b", "b->c", "x->y"))
    shifted = _block("shifted", ("a->b", "b->c", "z->w"))
    result = matched_transition(
        historical,
        shifted,
        np.asarray([False, True, True]),
        np.asarray([True, True, False]),
    )
    assert result["shared_pairs"] == 2
    assert result["historical_only_pairs"] == 1
    assert result["shift_only_pairs"] == 1
    assert result["transitions"]["off_to_on"] == 1
    assert result["transitions"]["on_to_on"] == 1
