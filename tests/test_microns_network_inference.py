import numpy as np

from mousebrainbench.benchmarks.microns_network_inference import (
    _confirmation_status,
    _dyadic_meat,
    _freedman_lane_node_permutation,
    _linear_inference,
)


def _complete_pairs(n_units: int) -> tuple[np.ndarray, np.ndarray]:
    return np.where(~np.eye(n_units, dtype=bool))


def test_dyadic_meat_equals_explicit_shared_unit_sum() -> None:
    pre, post = _complete_pairs(5)
    rng = np.random.default_rng(4)
    scores = rng.normal(size=(len(pre), 3))
    observed = _dyadic_meat(scores, pre, post, 5)
    expected = np.zeros((3, 3))
    for left in range(len(pre)):
        for right in range(len(pre)):
            if {pre[left], post[left]} & {pre[right], post[right]}:
                expected += np.outer(scores[left], scores[right])

    assert np.allclose(observed, expected)


def test_dyadic_inference_detects_positive_coefficient_with_shared_unit_noise() -> None:
    rng = np.random.default_rng(8)
    n_units = 45
    pre, post = _complete_pairs(n_units)
    focal = rng.binomial(1, 0.12, size=len(pre)).astype(float)
    control = rng.normal(size=len(pre))
    design = np.column_stack((np.ones(len(pre)), focal, control))
    unit_effect = rng.normal(scale=0.7, size=n_units)
    outcome = 0.45 * focal + 0.2 * control + unit_effect[pre] + unit_effect[post]
    outcome += rng.normal(scale=0.5, size=len(pre))

    result = _linear_inference(design, outcome, pre, post, n_units)

    assert result["connected_coefficient"] > 0.3
    assert result["dyadic_cluster_standard_error"] > 0
    assert 0 <= result["dyadic_cluster_two_sided_p_value"] <= 1


def test_freedman_lane_node_permutation_is_deterministic() -> None:
    rng = np.random.default_rng(3)
    n_units = 16
    pre, post = _complete_pairs(n_units)
    focal = rng.binomial(1, 0.15, size=len(pre)).astype(float)
    control = rng.normal(size=(len(pre), 1))
    design = np.column_stack((np.ones(len(pre)), focal, control))
    outcome = 0.8 * focal + control[:, 0] + rng.normal(scale=0.4, size=len(pre))

    first = _freedman_lane_node_permutation(
        design=design,
        controls=control,
        outcome=outcome,
        pre=pre,
        post=post,
        n_units=n_units,
        n_permutations=99,
        rng=np.random.default_rng(91),
    )
    second = _freedman_lane_node_permutation(
        design=design,
        controls=control,
        outcome=outcome,
        pre=pre,
        post=post,
        n_units=n_units,
        n_permutations=99,
        rng=np.random.default_rng(91),
    )

    assert first == second
    assert first["observed_coefficient"] > 0
    assert 0 <= first["one_sided_p_value"] <= 1


def test_confirmation_uses_direction_from_discovery_and_both_holdouts() -> None:
    cohorts = [
        {"connected_coefficient": 0.2, "network_inference_passed": False},
        {"connected_coefficient": 0.1, "network_inference_passed": True},
        {"connected_coefficient": 0.1, "network_inference_passed": True},
    ]

    status = _confirmation_status(cohorts)

    assert status["discovery_direction_positive"] is True
    assert status["holdout_results"] == (True, True)
    assert status["confirmation_passed"] is True


def test_confirmation_rejects_negative_discovery_direction_or_failed_holdout() -> None:
    negative_discovery = [
        {"connected_coefficient": -0.2, "network_inference_passed": True},
        {"connected_coefficient": 0.1, "network_inference_passed": True},
        {"connected_coefficient": 0.1, "network_inference_passed": True},
    ]
    failed_holdout = [
        {"connected_coefficient": 0.2, "network_inference_passed": True},
        {"connected_coefficient": 0.1, "network_inference_passed": False},
        {"connected_coefficient": 0.1, "network_inference_passed": True},
    ]

    assert _confirmation_status(negative_discovery)["confirmation_passed"] is False
    assert _confirmation_status(failed_holdout)["confirmation_passed"] is False
