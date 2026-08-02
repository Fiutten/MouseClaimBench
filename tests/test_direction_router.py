import numpy as np

from mousebrainbench.validation.direction_router import (
    DirectionAssumptions,
    route_direction,
)


def _forward_method(x, y, *, seed):
    del x, y, seed
    return {"predicted_direction": "forward", "status": "passed"}


def test_router_abstains_when_hidden_confounding_is_not_excluded() -> None:
    values = np.arange(30, dtype=float)
    result = route_direction(
        values,
        values,
        DirectionAssumptions(
            additive_noise=True,
            acyclic=True,
            hidden_confounding_excluded=False,
            selection_bias_excluded=True,
            provenance="test declaration",
        ),
        seed=1,
        anm_function=_forward_method,
    )

    assert result["attempted"] is False
    assert result["status"] == "requires_review"
    assert "hidden_confounding_not_excluded" in result["blockers"]


def test_router_uses_lingam_only_under_declared_linear_nongaussian_assumptions() -> None:
    values = np.arange(30, dtype=float)
    result = route_direction(
        values,
        values + 1.0,
        DirectionAssumptions(
            linear=True,
            non_gaussian=True,
            acyclic=True,
            hidden_confounding_excluded=True,
            selection_bias_excluded=True,
            provenance="known synthetic SEM",
        ),
        seed=2,
        lingam_function=_forward_method,
    )

    assert result["attempted"] is True
    assert result["method"] == "causal-learn_DirectLiNGAM"
    assert result["direction_support_allowed"] is True
    assert result["causal_support_allowed"] is False


def test_randomized_intervention_has_precedence_and_can_support_causality() -> None:
    rng = np.random.default_rng(3)
    result = route_direction(
        rng.normal(size=50),
        rng.normal(size=50),
        DirectionAssumptions(
            randomized_intervention=True,
            hidden_confounding_excluded=False,
            selection_bias_excluded=False,
            provenance="randomized physical intervention",
        ),
        seed=3,
        intervention_control=rng.normal(0.0, 1.0, size=100),
        intervention_treated=rng.normal(1.2, 1.0, size=100),
    )

    assert result["method"] == "controlled_intervention_contrast"
    assert result["predicted_direction"] == "forward"
    assert result["causal_support_allowed"] is True


def test_unknown_assumptions_force_abstention() -> None:
    values = np.arange(30, dtype=float)
    result = route_direction(
        values,
        values,
        DirectionAssumptions(
            hidden_confounding_excluded=True,
            selection_bias_excluded=True,
        ),
        seed=4,
    )

    assert result["attempted"] is False
    assert result["blockers"] == ["no_declared_method_matches_assumptions"]
