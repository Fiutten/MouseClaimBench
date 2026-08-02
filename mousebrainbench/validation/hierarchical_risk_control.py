"""Cluster-valid risk control for hierarchically nested scientific claims.

Scientific claim rows commonly reuse observations, variables, or simulation
states. Treating those rows as independent produces confidence intervals that
are much too narrow. This module validates a declared hierarchy and collapses
all lower-level decisions to one worst-case event per independent top-level
unit. Exact bounds are then computed only across those top-level units.

Lower-level summaries remain useful for diagnosis. They are deliberately
labelled descriptive and never participate in certification.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from mousebrainbench.validation.nondegenerate_risk_control import (
    NonDegenerateCertificate,
    evaluate_authorization_decisions,
)


@dataclass(frozen=True)
class HierarchicalCertificate:
    """A top-level certificate plus non-inferential hierarchy diagnostics."""

    certificate: NonDegenerateCertificate
    minimum_independent_units: int
    hierarchy_valid: bool
    sufficient_independent_units: bool
    lower_level_summary: dict[str, Any]
    stratum_summary: dict[str, dict[str, Any]]
    certified: bool

    def as_dict(self) -> dict[str, Any]:
        payload = self.certificate.as_dict()
        payload.update(
            {
                "inferential_unit": "top_level_cluster",
                "minimum_independent_units": self.minimum_independent_units,
                "hierarchy_valid": self.hierarchy_valid,
                "sufficient_independent_units": self.sufficient_independent_units,
                "lower_level_summary": self.lower_level_summary,
                "stratum_summary": self.stratum_summary,
                "certified": self.certified,
                "lower_level_inference_allowed": False,
            }
        )
        return payload


def _as_rows(values: np.ndarray, *, name: str, rows: int) -> np.ndarray:
    output = np.asarray(values).astype(str)
    if output.ndim != 1 or len(output) != rows:
        raise ValueError(f"{name} must contain exactly one identifier per row")
    if np.any(output == ""):
        raise ValueError(f"{name} cannot contain empty identifiers")
    return output


def validate_nested_hierarchy(
    top_level_ids: np.ndarray,
    subgroup_ids: np.ndarray,
) -> None:
    """Require every lower-level subgroup to belong to one top-level unit."""

    top = np.asarray(top_level_ids).astype(str)
    subgroup = np.asarray(subgroup_ids).astype(str)
    if top.ndim != 1 or subgroup.ndim != 1 or top.shape != subgroup.shape:
        raise ValueError("top_level_ids and subgroup_ids must be aligned vectors")
    for name in np.unique(subgroup):
        parents = np.unique(top[subgroup == name])
        if len(parents) != 1:
            raise ValueError(
                f"subgroup {name!r} is assigned to {len(parents)} top-level units"
            )


def _descriptive_level(
    decisions: np.ndarray,
    labels: np.ndarray,
    admissible: np.ndarray,
    ids: np.ndarray,
) -> dict[str, Any]:
    failed = authorized = eligible = recovered = 0
    unique = np.unique(ids)
    for unit in unique:
        selected = ids == unit
        unit_decisions = decisions[selected]
        unit_truth = labels[selected]
        unit_gate = admissible[selected]
        failed += int(np.any(unit_decisions & ~unit_truth))
        authorized += int(np.any(unit_decisions))
        has_positive = bool(np.any(unit_truth & unit_gate))
        eligible += int(has_positive)
        recovered += int(has_positive and np.any(unit_decisions & unit_truth))
    return {
        "units": len(unique),
        "failing_units": failed,
        "authorized_units": authorized,
        "eligible_positive_units": eligible,
        "recovered_positive_units": recovered,
        "empirical_failure_rate": failed / len(unique) if len(unique) else 0.0,
        "empirical_coverage": authorized / len(unique) if len(unique) else 0.0,
        "empirical_positive_recovery": recovered / eligible if eligible else 0.0,
        "inferential_status": "descriptive_only",
    }


def evaluate_hierarchical_decisions(
    decisions: np.ndarray,
    labels: np.ndarray,
    admissible: np.ndarray,
    top_level_ids: np.ndarray,
    subgroup_ids: np.ndarray,
    strata: np.ndarray,
    *,
    threshold: float,
    target_risk: float,
    minimum_coverage: float,
    minimum_positive_recovery: float,
    minimum_independent_units: int,
    confidence: float = 0.95,
) -> HierarchicalCertificate:
    """Evaluate decisions with inference restricted to independent clusters."""

    chosen = np.asarray(decisions, dtype=bool)
    truth = np.asarray(labels, dtype=bool)
    gate = np.asarray(admissible, dtype=bool)
    if chosen.ndim != 2 or chosen.shape != truth.shape or chosen.shape != gate.shape:
        raise ValueError("decision, label, and admissibility matrices must align")
    rows = chosen.shape[0]
    top = _as_rows(top_level_ids, name="top_level_ids", rows=rows)
    subgroup = _as_rows(subgroup_ids, name="subgroup_ids", rows=rows)
    stratum = _as_rows(strata, name="strata", rows=rows)
    if minimum_independent_units <= 0:
        raise ValueError("minimum_independent_units must be positive")
    validate_nested_hierarchy(top, subgroup)

    base = evaluate_authorization_decisions(
        chosen,
        truth,
        gate,
        top,
        threshold=threshold,
        target_risk=target_risk,
        minimum_coverage=minimum_coverage,
        minimum_positive_recovery=minimum_positive_recovery,
        confidence=confidence,
    )
    enough = base.experiments >= minimum_independent_units
    by_stratum = {
        name: _descriptive_level(
            chosen[stratum == name],
            truth[stratum == name],
            gate[stratum == name],
            subgroup[stratum == name],
        )
        for name in sorted(np.unique(stratum))
    }
    lower = {
        "subgroups": _descriptive_level(chosen, truth, gate, subgroup),
        "claim_rows": {
            "rows": rows,
            "claim_candidates": int(chosen.size),
            "authorizations": int(chosen.sum()),
            "false_authorizations": int((chosen & ~truth).sum()),
            "semantic_violations": int((chosen & ~gate).sum()),
            "inferential_status": "descriptive_only",
        },
    }
    return HierarchicalCertificate(
        certificate=base,
        minimum_independent_units=minimum_independent_units,
        hierarchy_valid=True,
        sufficient_independent_units=enough,
        lower_level_summary=lower,
        stratum_summary=by_stratum,
        certified=bool(base.certified and enough),
    )


def evaluate_hierarchical_policy(
    scores: np.ndarray,
    labels: np.ndarray,
    admissible: np.ndarray,
    top_level_ids: np.ndarray,
    subgroup_ids: np.ndarray,
    strata: np.ndarray,
    *,
    threshold: float,
    target_risk: float,
    minimum_coverage: float,
    minimum_positive_recovery: float,
    minimum_independent_units: int,
    confidence: float = 0.95,
) -> HierarchicalCertificate:
    """Threshold scores and evaluate them under the declared hierarchy."""

    values = np.asarray(scores, dtype=float)
    if not np.all(np.isfinite(values)):
        raise ValueError("scores must be finite")
    decisions = np.asarray(admissible, dtype=bool) & (values >= threshold)
    return evaluate_hierarchical_decisions(
        decisions,
        labels,
        admissible,
        top_level_ids,
        subgroup_ids,
        strata,
        threshold=threshold,
        target_risk=target_risk,
        minimum_coverage=minimum_coverage,
        minimum_positive_recovery=minimum_positive_recovery,
        minimum_independent_units=minimum_independent_units,
        confidence=confidence,
    )


def calibrate_hierarchical_policy(
    scores: np.ndarray,
    labels: np.ndarray,
    admissible: np.ndarray,
    top_level_ids: np.ndarray,
    subgroup_ids: np.ndarray,
    strata: np.ndarray,
    *,
    target_risk: float,
    minimum_coverage: float,
    minimum_positive_recovery: float,
    minimum_independent_units: int,
    confidence: float = 0.95,
) -> HierarchicalCertificate | None:
    """Choose the most active top-level-valid threshold on calibration data."""

    values = np.asarray(scores, dtype=float)
    candidates = np.unique(
        np.concatenate(([0.0, 1.0 + np.finfo(float).eps], values.ravel()))
    )
    results = [
        evaluate_hierarchical_policy(
            values,
            labels,
            admissible,
            top_level_ids,
            subgroup_ids,
            strata,
            threshold=float(threshold),
            target_risk=target_risk,
            minimum_coverage=minimum_coverage,
            minimum_positive_recovery=minimum_positive_recovery,
            minimum_independent_units=minimum_independent_units,
            confidence=confidence,
        )
        for threshold in candidates
    ]
    valid = [result for result in results if result.certified]
    if not valid:
        return None
    return max(
        valid,
        key=lambda result: (
            result.certificate.authorized_experiments,
            result.certificate.recovered_positive_experiments,
            -result.certificate.risk_upper_bound,
            result.certificate.threshold,
        ),
    )
