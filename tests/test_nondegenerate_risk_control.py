import numpy as np
import pytest

from mousebrainbench.validation.nondegenerate_risk_control import (
    calibrate_nondegenerate_policy,
    evaluate_nondegenerate_policy,
    one_sided_binomial_bound,
)


def test_exact_bounds_have_expected_boundary_behavior() -> None:
    assert one_sided_binomial_bound(0, 29, confidence=0.95, side="upper") <= 0.10
    assert one_sided_binomial_bound(0, 28, confidence=0.95, side="upper") > 0.10
    assert one_sided_binomial_bound(29, 29, confidence=0.95, side="lower") >= 0.89
    assert one_sided_binomial_bound(0, 0, confidence=0.95, side="upper") == 1.0


def test_universal_abstention_cannot_be_certified() -> None:
    scores = np.ones((40, 1))
    labels = np.ones((40, 1), dtype=bool)
    certificate = evaluate_nondegenerate_policy(
        scores,
        labels,
        np.zeros_like(labels),
        np.asarray([f"experiment-{index}" for index in range(40)]),
        threshold=0.5,
        target_risk=0.10,
        minimum_coverage=0.10,
        minimum_positive_recovery=0.05,
    )
    assert certificate.failing_experiments == 0
    assert certificate.coverage_lower_bound == 0.0
    assert not certificate.certified


def test_dependent_rows_are_collapsed_to_experiments() -> None:
    scores = np.full((120, 1), 0.9)
    labels = np.ones((120, 1), dtype=bool)
    labels[0] = False
    experiments = np.repeat([f"experiment-{index}" for index in range(40)], 3)
    certificate = evaluate_nondegenerate_policy(
        scores,
        labels,
        np.ones_like(labels),
        experiments,
        threshold=0.5,
        target_risk=0.20,
        minimum_coverage=0.10,
        minimum_positive_recovery=0.05,
    )
    assert certificate.experiments == 40
    assert certificate.failing_experiments == 1
    assert certificate.authorizations == 120


def test_calibration_selects_active_certified_threshold() -> None:
    experiments = np.asarray([f"experiment-{index}" for index in range(60)])
    scores = np.linspace(0.01, 1.0, 60)[:, None]
    labels = (scores >= 0.3)
    result = calibrate_nondegenerate_policy(
        scores,
        labels,
        np.ones_like(labels),
        experiments,
        target_risk=0.10,
        minimum_coverage=0.10,
        minimum_positive_recovery=0.05,
    )
    assert result is not None
    assert result.certified
    assert result.authorized_experiments > 0
    assert result.risk_upper_bound <= 0.10
    assert result.coverage_lower_bound >= 0.10


def test_invalid_counts_are_rejected() -> None:
    with pytest.raises(ValueError):
        one_sided_binomial_bound(2, 1, confidence=0.95, side="upper")
