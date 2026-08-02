"""Population-scope contracts for finite-sample claim-risk guarantees.

Semantic admissibility and statistical certification answer different
questions. A claim can satisfy its evidence contract while a risk certificate
is invalid for the target population. This module makes that second boundary
executable and prevents an out-of-scope certificate from authorizing support.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class PopulationScope:
    """Immutable population and protocol identity covered by a certificate."""

    scope_id: str
    population_family: str
    independent_unit: str
    evidence_protocol: str
    reference_protocol: str

    @property
    def fingerprint(self) -> str:
        payload = json.dumps(self.as_dict(include_fingerprint=False), sort_keys=True)
        return f"sha256:{hashlib.sha256(payload.encode()).hexdigest()}"

    def as_dict(self, *, include_fingerprint: bool = True) -> dict[str, str]:
        output = {
            "scope_id": self.scope_id,
            "population_family": self.population_family,
            "independent_unit": self.independent_unit,
            "evidence_protocol": self.evidence_protocol,
            "reference_protocol": self.reference_protocol,
        }
        if include_fingerprint:
            output["fingerprint"] = self.fingerprint
        return output


@dataclass(frozen=True)
class GuaranteeScopeAssessment:
    """Decision about whether one certificate may govern a target population."""

    valid: bool
    calibration_scope: PopulationScope
    target_scope: PopulationScope
    detected_shift: bool | None
    blockers: tuple[str, ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "valid": self.valid,
            "calibration_scope": self.calibration_scope.as_dict(),
            "target_scope": self.target_scope.as_dict(),
            "detected_shift": self.detected_shift,
            "blockers": list(self.blockers),
        }


def assess_guarantee_scope(
    calibration: PopulationScope,
    target: PopulationScope,
    *,
    detected_shift: bool | None = None,
) -> GuaranteeScopeAssessment:
    """Require exact population/protocol scope and no detected distribution shift."""

    blockers = []
    for field in (
        "population_family",
        "independent_unit",
        "evidence_protocol",
        "reference_protocol",
    ):
        if getattr(calibration, field) != getattr(target, field):
            blockers.append(f"scope_mismatch:{field}")
    if detected_shift is True:
        blockers.append("distribution_shift_detected")
    return GuaranteeScopeAssessment(
        valid=not blockers,
        calibration_scope=calibration,
        target_scope=target,
        detected_shift=detected_shift,
        blockers=tuple(blockers),
    )


def enforce_guarantee_scope(
    authorization_decisions: np.ndarray,
    assessment: GuaranteeScopeAssessment,
) -> np.ndarray:
    """Retain supports only when their finite-sample certificate is in scope."""

    decisions = np.asarray(authorization_decisions, dtype=np.int8)
    if np.any((decisions != 0) & (decisions != 1)):
        raise ValueError("guarantee-scoped decisions must be support-or-abstain")
    return decisions.copy() if assessment.valid else np.zeros_like(decisions)
