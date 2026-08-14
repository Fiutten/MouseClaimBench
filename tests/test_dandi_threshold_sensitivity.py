from pathlib import Path

import pytest
import yaml

from mousebrainbench.benchmarks.dandi_threshold_sensitivity import evaluate


def _result():
    protocol = yaml.safe_load(
        Path("configs/benchmarks/dandi_threshold_sensitivity.yaml").read_text()
    )
    return evaluate(protocol)


def test_threshold_sensitivity_reproduces_frozen_decisions() -> None:
    result = _result()

    assert result["completed"] is True
    assert all(result["conditions"].values())
    assert result["scope"]["criterion_validity_claimed"] is False
    assert result["scope"]["threshold_calibration_claimed"] is False


def test_threshold_sensitivity_reports_observed_decision_boundaries() -> None:
    result = _result()
    boundaries = result["decision_boundaries"]

    assert boundaries["contrast_maximum_subject_requirement_that_passes"] == 32
    assert boundaries["availability_maximum_subject_requirement_that_passes"] == 5
    assert boundaries[
        "contrast_maximum_median_correlation_threshold_that_passes"
    ] == pytest.approx(0.3100103839)
    assert boundaries[
        "contrast_maximum_relative_mse_improvement_that_passes"
    ] == pytest.approx(0.0543127301)


def test_threshold_grids_change_decisions_at_stricter_values() -> None:
    results = _result()["one_at_a_time_results"]

    assert results["contrast_minimum_subjects"][-1][
        "authorized_with_other_frozen_conditions_held"
    ] is False
    assert results["median_subject_correlation"][-1][
        "authorized_with_other_frozen_conditions_held"
    ] is False
    assert results["availability_minimum_subjects"][0][
        "authorized_with_other_frozen_conditions_held"
    ] is True
    assert results["availability_minimum_subjects"][1][
        "authorized_with_other_frozen_conditions_held"
    ] is False
