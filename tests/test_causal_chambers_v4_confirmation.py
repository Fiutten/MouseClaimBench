import numpy as np

from mousebrainbench.benchmarks.causal_chambers_v4_confirmation import (
    RoleData,
    _failure_taxonomy,
    _select_pairs,
)


def test_pair_selection_is_bounded_and_deterministic() -> None:
    pairs = tuple((f"x{index}", f"y{index}", index < 5) for index in range(10))
    first = _select_pairs(
        pairs, namespace="frozen", direct_count=1, control_count=1
    )
    second = _select_pairs(
        tuple(reversed(pairs)), namespace="frozen", direct_count=1, control_count=1
    )
    assert first == second
    assert len(first) == 2
    assert sum(row[2] for row in first) == 1


def test_failure_taxonomy_is_exhaustive_and_nonoverlapping() -> None:
    data = RoleData(
        role="risk_lock",
        records=(),
        scores=np.asarray([[0.9, 0.2], [0.8, 0.7]]),
        labels=np.ones((2, 2), dtype=bool),
        admissible=np.asarray([[True, True], [False, True]]),
        features=np.zeros((2, 1)),
        experiment_ids=np.asarray(["a", "b"]),
        datasets=np.asarray(["d", "d"]),
        file_hashes={},
        source_bytes=0,
        exclusions={},
    )
    result = _failure_taxonomy(
        data, threshold=0.5, calibration_failure=None, shift_warning=True
    )
    counts = result["mutually_exclusive_counts"]
    assert sum(counts.values()) == 4
    assert counts["authorized"] == 2
    assert counts["semantic_inadmissibility"] == 1
    assert counts["score_below_threshold"] == 1
    assert result["orthogonal_context_flags"]["detected_shift_warning"]

