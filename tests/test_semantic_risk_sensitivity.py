import numpy as np
import pytest

from mousebrainbench.benchmarks.semantic_risk_sensitivity import (
    _aggregate_rows,
    deterministic_subsample_indices,
)


def test_deterministic_subsample_is_reproducible_unique_and_bounded() -> None:
    first = deterministic_subsample_indices(100, 25, repeat=3)
    second = deterministic_subsample_indices(100, 25, repeat=3)

    assert np.array_equal(first, second)
    assert len(np.unique(first)) == 25
    assert first.min() >= 0
    assert first.max() < 100


def test_full_sample_has_canonical_order_independent_of_repeat() -> None:
    assert np.array_equal(
        deterministic_subsample_indices(8, 8, repeat=9),
        np.arange(8),
    )


def test_invalid_subsample_size_is_rejected() -> None:
    with pytest.raises(ValueError, match="sample size"):
        deterministic_subsample_indices(10, 11, repeat=0)


def test_repeat_summary_does_not_select_best_repeat() -> None:
    summary = _aggregate_rows(
        [
            {
                "supported_coverage": 0.2,
                "semantic_false_authorization_risk": 0.04,
                "certified_claims": 4,
                "sfar_at_or_below_target": True,
            },
            {
                "supported_coverage": 0.4,
                "semantic_false_authorization_risk": 0.08,
                "certified_claims": 6,
                "sfar_at_or_below_target": False,
            },
        ]
    )

    assert summary["fresh_supported_coverage_mean"] == pytest.approx(0.3)
    assert summary["fresh_sfar_mean"] == pytest.approx(0.06)
    assert summary["certified_claims_min"] == 4
    assert summary["certified_claims_max"] == 6
    assert summary["fraction_repeats_at_or_below_target"] == 0.5
