"""Assumption-aware routing for directional evidence.

The router reuses established estimators and refuses to choose a direction when
their assumptions are absent. Only controlled intervention evidence may mark a
direction as causal support; observational methods remain conditional evidence.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy import stats

from mousebrainbench.benchmarks.causal_direction_anm import anm_direction_evidence


@dataclass(frozen=True)
class DirectionAssumptions:
    """Auditable facts used to select, or reject, a direction estimator."""

    randomized_intervention: bool = False
    linear: bool = False
    non_gaussian: bool = False
    additive_noise: bool = False
    continuous: bool = True
    acyclic: bool = False
    hidden_confounding_excluded: bool = False
    selection_bias_excluded: bool = False
    material_measurement_error: bool = False
    association_established: bool | None = None
    provenance: str = "undeclared"


def association_precondition(
    x: np.ndarray,
    y: np.ndarray,
    *,
    familywise_alpha: float = 0.01,
    minimum_absolute_association: float = 0.10,
    minimum_passing_tests: int = 1,
) -> dict[str, Any]:
    """Screen for association before spending directional evidence.

    Pearson and Spearman tests cover linear and monotone association.  The
    Bonferroni two-test correction is frozen prospectively in v4.  Passing this
    screen does not identify direction or causality; it only prevents a direction
    estimator from orienting pairs for which association was not established.
    """

    x_values = np.asarray(x, dtype=float).reshape(-1)
    y_values = np.asarray(y, dtype=float).reshape(-1)
    if len(x_values) != len(y_values):
        raise ValueError("association variables must contain equal observations")
    finite = np.isfinite(x_values) & np.isfinite(y_values)
    x_values = x_values[finite]
    y_values = y_values[finite]
    if len(x_values) < 20 or min(np.std(x_values), np.std(y_values)) == 0.0:
        return {
            "established": False,
            "blocker": "insufficient_or_constant_association_data",
            "finite_samples": len(x_values),
        }
    pearson = stats.pearsonr(x_values, y_values)
    spearman = stats.spearmanr(x_values, y_values)
    tests = (
        ("pearson", float(pearson.statistic), float(pearson.pvalue)),
        ("spearman", float(spearman.statistic), float(spearman.pvalue)),
    )
    if minimum_passing_tests not in {1, 2}:
        raise ValueError("minimum_passing_tests must be one or two")
    passing = [
        name
        for name, statistic, p_value in tests
        if abs(statistic) >= minimum_absolute_association
        and min(1.0, 2.0 * p_value) < familywise_alpha
    ]
    return {
        "established": len(passing) >= minimum_passing_tests,
        "passing_tests": passing,
        "tests": [
            {
                "name": name,
                "statistic": statistic,
                "raw_p_value": p_value,
                "bonferroni_p_value": min(1.0, 2.0 * p_value),
            }
            for name, statistic, p_value in tests
        ],
        "familywise_alpha": familywise_alpha,
        "minimum_absolute_association": minimum_absolute_association,
        "minimum_passing_tests": minimum_passing_tests,
        "finite_samples": len(x_values),
        "causal_or_directional_evidence": False,
    }


def _direct_lingam(x: np.ndarray, y: np.ndarray, *, seed: int) -> dict[str, Any]:
    try:
        from causallearn.search.FCMBased import lingam
    except ImportError as exc:
        raise RuntimeError(
            "DirectLiNGAM routing requires the `hybrid-validation` dependencies"
        ) from exc
    values = np.column_stack((np.asarray(x, dtype=float), np.asarray(y, dtype=float)))
    finite = np.all(np.isfinite(values), axis=1)
    values = values[finite]
    if len(values) < 20 or np.any(values.std(axis=0) == 0.0):
        raise ValueError("DirectLiNGAM requires 20 finite non-constant paired samples")
    values = (values - values.mean(axis=0)) / values.std(axis=0)
    model = lingam.DirectLiNGAM(random_state=seed)
    model.fit(values)
    adjacency = np.asarray(model.adjacency_matrix_, dtype=float)
    forward = abs(float(adjacency[1, 0]))
    reverse = abs(float(adjacency[0, 1]))
    margin = forward - reverse
    direction = "forward" if margin > 0.05 else "reverse" if margin < -0.05 else "uncertain"
    return {
        "predicted_direction": direction,
        "status": "passed" if direction == "forward" else "failed" if direction == "reverse" else "requires_review",
        "forward_weight": forward,
        "reverse_weight": reverse,
        "signed_margin": margin,
        "causal_order": [int(value) for value in model.causal_order_],
    }


def _intervention_contrast(
    control: np.ndarray,
    treated: np.ndarray,
    *,
    alpha: float = 0.01,
) -> dict[str, Any]:
    control_values = np.asarray(control, dtype=float)
    treated_values = np.asarray(treated, dtype=float)
    control_values = control_values[np.isfinite(control_values)]
    treated_values = treated_values[np.isfinite(treated_values)]
    if min(len(control_values), len(treated_values)) < 10:
        raise ValueError("intervention contrast requires ten finite observations per arm")
    test = stats.ttest_ind(treated_values, control_values, equal_var=False)
    effect = float(treated_values.mean() - control_values.mean())
    direction = (
        "forward" if test.pvalue < alpha and effect > 0.0
        else "reverse" if test.pvalue < alpha and effect < 0.0
        else "uncertain"
    )
    return {
        "predicted_direction": direction,
        "status": "passed" if direction == "forward" else "failed" if direction == "reverse" else "requires_review",
        "mean_effect": effect,
        "p_value": float(test.pvalue),
        "alpha": alpha,
        "control_n": len(control_values),
        "treated_n": len(treated_values),
    }


def route_direction(
    x: np.ndarray,
    y: np.ndarray,
    assumptions: DirectionAssumptions,
    *,
    seed: int,
    intervention_control: np.ndarray | None = None,
    intervention_treated: np.ndarray | None = None,
    anm_function: Callable[..., dict[str, Any]] = anm_direction_evidence,
    lingam_function: Callable[..., dict[str, Any]] = _direct_lingam,
    require_association_precondition: bool = False,
) -> dict[str, Any]:
    """Select a valid established estimator or return a reasoned abstention.

    ``require_association_precondition`` is opt-in to preserve the frozen v3
    router. New protocols should enable it so an observational direction method
    cannot orient variables whose association has not first been established.
    Randomized intervention evidence is exempt because its contrast directly
    tests an intervention effect.
    """

    common_blockers = []
    if not assumptions.hidden_confounding_excluded:
        common_blockers.append("hidden_confounding_not_excluded")
    if not assumptions.selection_bias_excluded:
        common_blockers.append("selection_bias_not_excluded")
    if assumptions.material_measurement_error:
        common_blockers.append("material_measurement_error")
    if (
        require_association_precondition
        and not assumptions.randomized_intervention
        and assumptions.association_established is not True
    ):
        common_blockers.append("association_not_established")

    method = "abstain"
    evidence: dict[str, Any] = {
        "predicted_direction": "uncertain",
        "status": "requires_review",
    }
    blockers = list(common_blockers)
    try:
        if assumptions.randomized_intervention:
            if intervention_control is None or intervention_treated is None:
                blockers.append("intervention_arms_missing")
            else:
                method = "controlled_intervention_contrast"
                blockers = []
                evidence = _intervention_contrast(intervention_control, intervention_treated)
        elif not blockers and all(
            (assumptions.linear, assumptions.non_gaussian, assumptions.acyclic)
        ):
            method = "causal-learn_DirectLiNGAM"
            evidence = lingam_function(x, y, seed=seed)
        elif not blockers and all(
            (assumptions.additive_noise, assumptions.continuous, assumptions.acyclic)
        ):
            method = "causal-learn_ANM"
            evidence = anm_function(x, y, seed=seed)
        elif not blockers:
            blockers.append("no_declared_method_matches_assumptions")
    except (ValueError, FloatingPointError, np.linalg.LinAlgError) as exc:
        method = "abstain"
        blockers.append(f"numerical_failure:{type(exc).__name__}")
        evidence = {
            "predicted_direction": "uncertain",
            "status": "requires_review",
            "execution_error": str(exc),
        }

    attempted = method != "abstain" and not blockers
    predicted_direction = evidence.get("predicted_direction", "uncertain")
    direction_identified = attempted and predicted_direction in {"forward", "reverse"}
    return {
        "method": method if attempted else "abstain",
        "attempted": attempted,
        "predicted_direction": predicted_direction,
        "status": "passed" if direction_identified else "requires_review",
        "causal_support_allowed": (
            direction_identified and method == "controlled_intervention_contrast"
        ),
        "direction_support_allowed": direction_identified,
        "assumption_provenance": assumptions.provenance,
        "blockers": blockers,
        "evidence": evidence,
    }
