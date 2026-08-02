"""Exact experiment-level risk control with non-trivial activation constraints.

Pair-level scientific claims can share data and variables.  This module avoids
pretending that those rows are independent by reducing decisions to one bounded
event per declared experimental unit.  A policy is complete only when it controls
false authorization *and* has positive lower confidence bounds on use and useful
recovery.  Thus, universal abstention cannot be certified as a successful policy.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.stats import beta


def one_sided_binomial_bound(
    successes: int,
    trials: int,
    *,
    confidence: float,
    side: str,
) -> float:
    """Return an exact one-sided Clopper-Pearson bound.

    ``successes`` names the event whose probability is bounded.  For risk this
    is a failed experiment and the upper bound is required.  For activation and
    recovery it is a successful event and the lower bound is required.
    """

    if trials < 0 or successes < 0 or successes > trials:
        raise ValueError("successes and trials must satisfy 0 <= successes <= trials")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must lie strictly between zero and one")
    if side not in {"lower", "upper"}:
        raise ValueError("side must be 'lower' or 'upper'")
    if trials == 0:
        return 0.0 if side == "lower" else 1.0
    alpha = 1.0 - confidence
    if side == "lower":
        return 0.0 if successes == 0 else float(beta.ppf(alpha, successes, trials - successes + 1))
    return 1.0 if successes == trials else float(beta.ppf(1.0 - alpha, successes + 1, trials - successes))


@dataclass(frozen=True)
class NonDegenerateCertificate:
    """Joint risk, activation, recovery, and semantic-integrity certificate."""

    threshold: float
    experiments: int
    failing_experiments: int
    authorized_experiments: int
    eligible_positive_experiments: int
    recovered_positive_experiments: int
    authorizations: int
    false_authorizations: int
    semantic_violations: int
    risk_upper_bound: float
    coverage_lower_bound: float
    positive_recovery_lower_bound: float
    target_risk: float
    minimum_coverage: float
    minimum_positive_recovery: float
    confidence: float
    certified: bool

    def as_dict(self) -> dict[str, Any]:
        return {
            "threshold": self.threshold,
            "experiments": self.experiments,
            "failing_experiments": self.failing_experiments,
            "authorized_experiments": self.authorized_experiments,
            "eligible_positive_experiments": self.eligible_positive_experiments,
            "recovered_positive_experiments": self.recovered_positive_experiments,
            "authorizations": self.authorizations,
            "false_authorizations": self.false_authorizations,
            "semantic_violations": self.semantic_violations,
            "empirical_experiment_failure_rate": (
                self.failing_experiments / self.experiments if self.experiments else 0.0
            ),
            "empirical_authorized_experiment_coverage": (
                self.authorized_experiments / self.experiments if self.experiments else 0.0
            ),
            "empirical_positive_recovery": (
                self.recovered_positive_experiments / self.eligible_positive_experiments
                if self.eligible_positive_experiments
                else 0.0
            ),
            "empirical_pair_sfar": (
                self.false_authorizations / self.authorizations if self.authorizations else 0.0
            ),
            "risk_upper_bound": self.risk_upper_bound,
            "coverage_lower_bound": self.coverage_lower_bound,
            "positive_recovery_lower_bound": self.positive_recovery_lower_bound,
            "target_risk": self.target_risk,
            "minimum_coverage": self.minimum_coverage,
            "minimum_positive_recovery": self.minimum_positive_recovery,
            "confidence": self.confidence,
            "certified": self.certified,
            "inferential_unit": "experiment",
        }


def evaluate_nondegenerate_policy(
    scores: np.ndarray,
    labels: np.ndarray,
    admissible: np.ndarray,
    experiment_ids: np.ndarray,
    *,
    threshold: float,
    target_risk: float,
    minimum_coverage: float,
    minimum_positive_recovery: float,
    confidence: float = 0.95,
) -> NonDegenerateCertificate:
    """Evaluate one frozen threshold after collapsing dependent rows by experiment."""

    values = np.asarray(scores, dtype=float)
    truth = np.asarray(labels, dtype=bool)
    gate = np.asarray(admissible, dtype=bool)
    units = np.asarray(experiment_ids).astype(str)
    if values.shape != truth.shape or values.shape != gate.shape:
        raise ValueError("scores, labels, and admissibility must have identical shape")
    if values.ndim != 2 or len(units) != values.shape[0]:
        raise ValueError("inputs must contain rows by claims and one experiment id per row")
    if not np.all(np.isfinite(values)):
        raise ValueError("scores must be finite")
    for value, name in (
        (target_risk, "target_risk"),
        (minimum_coverage, "minimum_coverage"),
        (minimum_positive_recovery, "minimum_positive_recovery"),
    ):
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"{name} must lie between zero and one")

    decisions = gate & (values >= threshold)
    return evaluate_authorization_decisions(
        decisions,
        truth,
        gate,
        units,
        threshold=threshold,
        target_risk=target_risk,
        minimum_coverage=minimum_coverage,
        minimum_positive_recovery=minimum_positive_recovery,
        confidence=confidence,
    )


def evaluate_authorization_decisions(
    decisions: np.ndarray,
    labels: np.ndarray,
    admissible: np.ndarray,
    experiment_ids: np.ndarray,
    *,
    threshold: float,
    target_risk: float,
    minimum_coverage: float,
    minimum_positive_recovery: float,
    confidence: float = 0.95,
) -> NonDegenerateCertificate:
    """Evaluate arbitrary baseline decisions under the same experiment contract."""

    decisions = np.asarray(decisions, dtype=bool)
    truth = np.asarray(labels, dtype=bool)
    gate = np.asarray(admissible, dtype=bool)
    units = np.asarray(experiment_ids).astype(str)
    if decisions.shape != truth.shape or decisions.shape != gate.shape:
        raise ValueError("decisions, labels, and admissibility must have identical shape")
    if decisions.ndim != 2 or len(units) != decisions.shape[0]:
        raise ValueError("decision inputs require rows by claims and one unit id per row")
    unique_units = np.unique(units)
    failed = authorized = eligible_positive = recovered = 0
    for unit in unique_units:
        selected = units == unit
        unit_decisions = decisions[selected]
        unit_truth = truth[selected]
        unit_gate = gate[selected]
        failed += int(np.any(unit_decisions & ~unit_truth))
        authorized += int(np.any(unit_decisions))
        positive = bool(np.any(unit_truth & unit_gate))
        eligible_positive += int(positive)
        recovered += int(positive and np.any(unit_decisions & unit_truth))

    n_units = len(unique_units)
    risk_upper = one_sided_binomial_bound(
        failed, n_units, confidence=confidence, side="upper"
    )
    coverage_lower = one_sided_binomial_bound(
        authorized, n_units, confidence=confidence, side="lower"
    )
    recovery_lower = one_sided_binomial_bound(
        recovered, eligible_positive, confidence=confidence, side="lower"
    )
    violations = int((decisions & ~gate).sum())
    return NonDegenerateCertificate(
        threshold=float(threshold),
        experiments=n_units,
        failing_experiments=failed,
        authorized_experiments=authorized,
        eligible_positive_experiments=eligible_positive,
        recovered_positive_experiments=recovered,
        authorizations=int(decisions.sum()),
        false_authorizations=int((decisions & ~truth).sum()),
        semantic_violations=violations,
        risk_upper_bound=risk_upper,
        coverage_lower_bound=coverage_lower,
        positive_recovery_lower_bound=recovery_lower,
        target_risk=target_risk,
        minimum_coverage=minimum_coverage,
        minimum_positive_recovery=minimum_positive_recovery,
        confidence=confidence,
        certified=bool(
            risk_upper <= target_risk
            and coverage_lower >= minimum_coverage
            and recovery_lower >= minimum_positive_recovery
            and violations == 0
        ),
    )


def calibrate_nondegenerate_policy(
    scores: np.ndarray,
    labels: np.ndarray,
    admissible: np.ndarray,
    experiment_ids: np.ndarray,
    *,
    target_risk: float,
    minimum_coverage: float,
    minimum_positive_recovery: float,
    confidence: float = 0.95,
) -> NonDegenerateCertificate | None:
    """Select the most active certified threshold on target calibration units."""

    values = np.asarray(scores, dtype=float)
    candidates = np.unique(np.concatenate(([0.0, 1.0 + np.finfo(float).eps], values.ravel())))
    certificates = [
        evaluate_nondegenerate_policy(
            values,
            labels,
            admissible,
            experiment_ids,
            threshold=float(threshold),
            target_risk=target_risk,
            minimum_coverage=minimum_coverage,
            minimum_positive_recovery=minimum_positive_recovery,
            confidence=confidence,
        )
        for threshold in candidates
    ]
    valid = [certificate for certificate in certificates if certificate.certified]
    if not valid:
        return None
    return max(
        valid,
        key=lambda item: (
            item.authorized_experiments,
            item.recovered_positive_experiments,
            -item.risk_upper_bound,
            item.threshold,
        ),
    )
