import numpy as np

from mousebrainbench.benchmarks.causal_chambers_v4_1_final import (
    _combine,
    _evaluate_decisions,
    _subset,
)
from mousebrainbench.benchmarks.causal_chambers_v4_confirmation import RoleData


def _data(role: str, offset: int) -> RoleData:
    return RoleData(
        role=role,
        records=(),
        scores=np.asarray([[0.2, 0.8], [0.6, 1.0]]) + offset,
        labels=np.ones((2, 2), dtype=bool),
        admissible=np.ones((2, 2), dtype=bool),
        features=np.ones((2, 3)) * offset,
        experiment_ids=np.asarray([f"{role}-a", f"{role}-b"]),
        datasets=np.asarray([role, role]),
        file_hashes={role: str(offset)},
        source_bytes=offset + 1,
        exclusions={"invalid_pair_record": offset},
    )


def test_combine_preserves_distinct_experimental_units() -> None:
    combined = _combine(_data("calibration", 0), _data("risk", 1))
    assert combined.scores.shape == (4, 2)
    assert len(np.unique(combined.experiment_ids)) == 4
    assert combined.source_bytes == 3


def test_subset_changes_claim_axis_only() -> None:
    data = _data("final", 0)
    selected = _subset(data, [1])
    assert selected.scores.shape == (2, 1)
    assert selected.features.shape == data.features.shape
    assert np.array_equal(selected.experiment_ids, data.experiment_ids)


def test_comparator_threshold_metadata_is_preserved() -> None:
    data = _subset(_data("final", 0), [1])
    decisions = {"policy": np.ones_like(data.admissible)}
    limits = {
        "target_risk": 0.10,
        "minimum_coverage": 0.10,
        "minimum_positive_recovery": 0.05,
        "confidence": 0.95,
    }
    result = _evaluate_decisions(decisions, data, limits, {"policy": 1.0})
    assert result["policy"]["threshold"] == 1.0
