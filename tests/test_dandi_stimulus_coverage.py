import numpy as np
import pytest

from mousebrainbench.benchmarks.dandi_stimulus_coverage import (
    aggregate_subject_coverage,
    analyze_stimulus_arrays,
)


def test_coverage_respects_frozen_chronological_boundaries() -> None:
    contrast = np.asarray([0.1, 0.2, 0.1, 0.2, 0.1, 0.2, 0.1, 0.3, 0.2, 0.3])
    direction = np.asarray([0, 90, 0, 90, 0, 90, 0, 180, 90, 180])
    row = analyze_stimulus_arrays(contrast, direction, expected_trials=10)

    assert row["train_trials"] == 6
    assert row["reserved_trials"] == 2
    assert row["test_trials"] == 2
    assert row["unique_test_condition_coverage"] == 0.5
    assert row["test_trial_condition_coverage"] == 0.5
    assert row["test_only_conditions"] == [[0.3, 180.0]]


def test_coverage_rejects_mismatch_with_frozen_endpoint_trials() -> None:
    with pytest.raises(ValueError, match="do not match"):
        analyze_stimulus_arrays(
            np.asarray([0.1, 0.2, np.nan]),
            np.asarray([0.0, 90.0, 180.0]),
            expected_trials=3,
        )


def test_aggregate_is_descriptive_and_non_decision_changing() -> None:
    row = analyze_stimulus_arrays(
        np.tile(np.asarray([0.1, 0.2]), 5),
        np.tile(np.asarray([0.0, 90.0]), 5),
    )
    aggregate = aggregate_subject_coverage([row, row])

    assert aggregate["subjects"] == 2
    assert aggregate["authorization_rule_changed"] is False
    assert aggregate["subjects_with_test_only_condition"] == 0
