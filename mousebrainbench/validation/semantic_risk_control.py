"""Finite-sample control of false scientific claim authorizations.

The complete semantic gate is embedded in the score function before MAPIE's
Learn-Then-Test calibration. Consequently, the learned threshold cannot make an
inadmissible claim supportable. Risk is calibrated claim by claim and confidence
is allocated with a Bonferroni family-wise correction.
"""

from __future__ import annotations

import importlib.metadata
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class ClaimRiskCertificate:
    """Serializable LTT certificate for one variable claim family."""

    claim: str
    threshold: float | None
    target_sfar: float
    confidence_level: float
    calibration_cases: int
    calibration_authorizations: int
    calibration_false_authorizations: int
    valid_threshold_count: int
    certified: bool
    implementation: str

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> ClaimRiskCertificate:
        """Restore a certificate without requiring a fitted MAPIE object."""

        return cls(
            claim=str(payload["claim"]),
            threshold=(
                float(payload["threshold"]) if payload.get("threshold") is not None else None
            ),
            target_sfar=float(payload["target_sfar"]),
            confidence_level=float(payload["confidence_level"]),
            calibration_cases=int(payload["calibration_cases"]),
            calibration_authorizations=int(payload["calibration_authorizations"]),
            calibration_false_authorizations=int(
                payload["calibration_false_authorizations"]
            ),
            valid_threshold_count=int(payload["valid_threshold_count"]),
            certified=bool(payload["certified"]),
            implementation=str(payload["implementation"]),
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "claim": self.claim,
            "threshold": self.threshold,
            "target_sfar": self.target_sfar,
            "confidence_level": self.confidence_level,
            "calibration_cases": self.calibration_cases,
            "calibration_authorizations": self.calibration_authorizations,
            "calibration_false_authorizations": self.calibration_false_authorizations,
            "valid_threshold_count": self.valid_threshold_count,
            "certified": self.certified,
            "implementation": self.implementation,
        }


@dataclass(frozen=True)
class SemanticRiskPolicy:
    """A family of simultaneous claim-specific support certificates."""

    claims: tuple[str, ...]
    certificates: tuple[ClaimRiskCertificate, ...]
    target_sfar: float
    familywise_confidence: float
    multiplicity_method: str

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> SemanticRiskPolicy:
        """Restore a frozen policy from its deterministic JSON representation."""

        return cls(
            claims=tuple(str(value) for value in payload["claims"]),
            certificates=tuple(
                ClaimRiskCertificate.from_dict(item) for item in payload["certificates"]
            ),
            target_sfar=float(payload["target_sfar"]),
            familywise_confidence=float(payload["familywise_confidence"]),
            multiplicity_method=str(payload["multiplicity_method"]),
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "claims": list(self.claims),
            "certificates": [item.as_dict() for item in self.certificates],
            "target_sfar": self.target_sfar,
            "familywise_confidence": self.familywise_confidence,
            "multiplicity_method": self.multiplicity_method,
            "simultaneously_certified": all(item.certified for item in self.certificates),
        }


def _mapie_controller():
    try:
        from mapie.risk_control import BinaryClassificationController
    except ImportError as exc:
        raise RuntimeError(
            "semantic risk control requires the `semantic-risk-v3` dependencies"
        ) from exc
    return BinaryClassificationController


def _semantic_probability(values: np.ndarray) -> np.ndarray:
    """Expose scores to MAPIE only when the immutable semantic gate passes."""

    matrix = np.asarray(values, dtype=float)
    if matrix.ndim != 2 or matrix.shape[1] != 2:
        raise ValueError("semantic score input must have [score, admissible] columns")
    score = np.clip(matrix[:, 0], 0.0, 1.0)
    admissible = matrix[:, 1] > 0.5
    positive = np.where(admissible, score, 0.0)
    return np.column_stack((1.0 - positive, positive))


def calibrate_semantic_risk(
    probabilities: np.ndarray,
    labels: np.ndarray,
    admissible: np.ndarray,
    *,
    claim_names: Sequence[str],
    target_sfar: float = 0.05,
    familywise_confidence: float = 0.95,
) -> SemanticRiskPolicy:
    """Calibrate one LTT precision controller per claim with family-wise control."""

    scores = np.asarray(probabilities, dtype=float)
    truths = np.asarray(labels, dtype=bool)
    gates = np.asarray(admissible, dtype=bool)
    expected = (scores.shape[0], len(claim_names))
    if scores.shape != expected or truths.shape != expected or gates.shape != expected:
        raise ValueError("probabilities, labels, and admissibility must share case/claim shape")
    if not 0.0 < target_sfar < 1.0:
        raise ValueError("target_sfar must lie strictly between zero and one")
    if not 0.0 < familywise_confidence < 1.0:
        raise ValueError("familywise_confidence must lie strictly between zero and one")
    if not np.all(np.isfinite(scores)):
        raise ValueError("claim probabilities must be finite")

    Controller = _mapie_controller()
    family_delta = 1.0 - familywise_confidence
    per_claim_confidence = 1.0 - family_delta / len(claim_names)
    certificates: list[ClaimRiskCertificate] = []
    for claim_index, claim in enumerate(claim_names):
        features = np.column_stack((scores[:, claim_index], gates[:, claim_index]))
        controller = Controller(
            _semantic_probability,
            "precision",
            target_level=1.0 - target_sfar,
            confidence_level=per_claim_confidence,
        )
        controller.calibrate(features, truths[:, claim_index].astype(int))
        valid = np.asarray(controller.valid_predict_params, dtype=float).reshape(-1)
        threshold = float(controller.best_predict_param) if valid.size else None
        selected = (
            gates[:, claim_index] & (scores[:, claim_index] >= threshold)
            if threshold is not None
            else np.zeros(len(scores), dtype=bool)
        )
        certificates.append(
            ClaimRiskCertificate(
                claim=claim,
                threshold=threshold,
                target_sfar=target_sfar,
                confidence_level=per_claim_confidence,
                calibration_cases=len(scores),
                calibration_authorizations=int(selected.sum()),
                calibration_false_authorizations=int((selected & ~truths[:, claim_index]).sum()),
                valid_threshold_count=int(valid.size),
                certified=threshold is not None,
                implementation=f"MAPIE-{importlib.metadata.version('mapie')}:LTT-precision",
            )
        )
    return SemanticRiskPolicy(
        claims=tuple(claim_names),
        certificates=tuple(certificates),
        target_sfar=target_sfar,
        familywise_confidence=familywise_confidence,
        multiplicity_method="Bonferroni across declared claim families",
    )


def authorize_with_policy(
    policy: SemanticRiskPolicy,
    probabilities: np.ndarray,
    admissible: np.ndarray,
) -> np.ndarray:
    """Apply frozen thresholds, returning support or abstention only."""

    scores = np.asarray(probabilities, dtype=float)
    gates = np.asarray(admissible, dtype=bool)
    expected = (scores.shape[0], len(policy.claims))
    if scores.shape != expected or gates.shape != expected:
        raise ValueError("policy input has incompatible case/claim shape")
    decisions = np.zeros(expected, dtype=np.int8)
    for claim_index, certificate in enumerate(policy.certificates):
        if certificate.threshold is None:
            continue
        decisions[:, claim_index] = (
            gates[:, claim_index]
            & (scores[:, claim_index] >= certificate.threshold)
        ).astype(np.int8)
    return decisions


def semantic_false_authorization_metrics(
    decisions: np.ndarray,
    labels: np.ndarray,
) -> dict[str, float | int]:
    """Measure authorization coverage and SFAR without pair-level pseudoreplication."""

    support = np.asarray(decisions) == 1
    truths = np.asarray(labels, dtype=bool)
    if support.shape != truths.shape:
        raise ValueError("decisions and labels must have identical shape")
    false = support & ~truths
    return {
        "cases": int(support.shape[0]),
        "claim_families": int(support.shape[1]),
        "authorizations": int(support.sum()),
        "false_authorizations": int(false.sum()),
        "supported_coverage": float(support.mean()),
        "semantic_false_authorization_risk": (
            float(false.sum() / support.sum()) if support.any() else 0.0
        ),
    }
