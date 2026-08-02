import numpy as np
import pytest

from mousebrainbench.validation.hierarchical_risk_control import (
    calibrate_hierarchical_policy,
    evaluate_hierarchical_policy,
    validate_nested_hierarchy,
)


def _hierarchy(bundles: int, scenarios: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    top = np.repeat([f"seed-{index}" for index in range(bundles)], scenarios)
    subgroup = np.asarray(
        [f"seed-{index}/scenario-{scenario}" for index in range(bundles) for scenario in range(scenarios)]
    )
    strata = np.tile([f"family-{index}" for index in range(scenarios)], bundles)
    return top, subgroup, strata


def test_false_claim_rows_collapse_to_one_seed_bundle_failure() -> None:
    top, subgroup, strata = _hierarchy(40, 4)
    scores = np.full((160, 1), 0.9)
    labels = np.ones_like(scores, dtype=bool)
    labels[:4] = False
    result = evaluate_hierarchical_policy(
        scores,
        labels,
        np.ones_like(labels),
        top,
        subgroup,
        strata,
        threshold=0.5,
        target_risk=0.20,
        minimum_coverage=0.10,
        minimum_positive_recovery=0.05,
        minimum_independent_units=29,
    )
    assert result.certificate.experiments == 40
    assert result.certificate.failing_experiments == 1
    assert result.lower_level_summary["subgroups"]["failing_units"] == 4
    assert result.lower_level_summary["claim_rows"]["false_authorizations"] == 4


def test_minimum_cluster_count_blocks_an_apparently_clean_result() -> None:
    top, subgroup, strata = _hierarchy(28, 2)
    labels = np.ones((56, 1), dtype=bool)
    result = evaluate_hierarchical_policy(
        np.ones_like(labels, dtype=float),
        labels,
        labels,
        top,
        subgroup,
        strata,
        threshold=0.5,
        target_risk=0.10,
        minimum_coverage=0.10,
        minimum_positive_recovery=0.05,
        minimum_independent_units=29,
    )
    assert not result.sufficient_independent_units
    assert not result.certified


def test_cross_parent_subgroup_is_rejected() -> None:
    with pytest.raises(ValueError, match="two-parents"):
        validate_nested_hierarchy(
            np.asarray(["seed-1", "seed-2"]),
            np.asarray(["two-parents", "two-parents"]),
        )


def test_calibration_cannot_select_universal_abstention() -> None:
    top, subgroup, strata = _hierarchy(40, 2)
    labels = np.ones((80, 1), dtype=bool)
    result = calibrate_hierarchical_policy(
        np.ones_like(labels, dtype=float),
        labels,
        np.zeros_like(labels),
        top,
        subgroup,
        strata,
        target_risk=0.10,
        minimum_coverage=0.10,
        minimum_positive_recovery=0.05,
        minimum_independent_units=29,
    )
    assert result is None


def test_calibration_selects_cluster_valid_active_threshold() -> None:
    top, subgroup, strata = _hierarchy(50, 2)
    bundle_score = np.linspace(0.01, 1.0, 50)
    scores = np.repeat(bundle_score, 2)[:, None]
    labels = scores >= 0.35
    result = calibrate_hierarchical_policy(
        scores,
        labels,
        np.ones_like(labels),
        top,
        subgroup,
        strata,
        target_risk=0.10,
        minimum_coverage=0.10,
        minimum_positive_recovery=0.05,
        minimum_independent_units=29,
    )
    assert result is not None
    assert result.certified
    assert result.certificate.risk_upper_bound <= 0.10
    assert result.certificate.authorized_experiments > 0
