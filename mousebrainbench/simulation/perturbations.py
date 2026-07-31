"""Reversible perturbations applied during simulation without mutating inputs."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

import numpy as np


class PerturbationType(StrEnum):
    LESION = "lesion"
    INHIBITION = "inhibition"
    STIMULATION = "stimulation"
    NOISE_INCREASE = "noise_increase"
    CONNECTIVITY_SCALING = "connectivity_scaling"


@dataclass(frozen=True)
class Perturbation:
    """Scheduled perturbation targeting region indices."""

    kind: PerturbationType
    target_indices: tuple[int, ...]
    onset: float
    duration: float | None = None
    magnitude: float = 1.0

    def __post_init__(self) -> None:
        if self.onset < 0:
            raise ValueError("perturbation onset must be non-negative")
        if self.duration is not None and self.duration <= 0:
            raise ValueError("perturbation duration must be positive")
        if self.kind in {PerturbationType.INHIBITION, PerturbationType.CONNECTIVITY_SCALING}:
            if not 0 <= self.magnitude <= 1:
                raise ValueError("inhibition and connectivity scaling magnitude must be in [0, 1]")

    def active(self, time: float) -> bool:
        return time >= self.onset and (self.duration is None or time < self.onset + self.duration)

    def as_dict(self) -> dict[str, object]:
        """Return JSON-safe provenance for persisted simulation metadata."""

        return {
            "kind": self.kind.value,
            "target_indices": list(self.target_indices),
            "onset": self.onset,
            "duration": self.duration,
            "magnitude": self.magnitude,
        }


def perturb_inputs_and_connectivity(
    *,
    time: float,
    state: np.ndarray,
    weights: np.ndarray,
    external_input: np.ndarray,
    noise_scale: float,
    perturbations: tuple[Perturbation, ...],
    n_regions: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Return perturbed copies for one derivative evaluation."""

    current_state = state.copy()
    current_weights = weights.copy()
    current_input = external_input.copy()
    current_noise = noise_scale
    for perturbation in perturbations:
        if not perturbation.active(time):
            continue
        targets = list(perturbation.target_indices)
        if perturbation.kind is PerturbationType.LESION:
            current_state[targets] = 0.0
            if len(current_state) >= 2 * n_regions:
                current_state[[index + n_regions for index in targets]] = 0.0
            current_weights[targets, :] = 0.0
            current_weights[:, targets] = 0.0
        elif perturbation.kind is PerturbationType.INHIBITION:
            current_input[targets] -= perturbation.magnitude
        elif perturbation.kind is PerturbationType.STIMULATION:
            current_input[targets] += perturbation.magnitude
        elif perturbation.kind is PerturbationType.NOISE_INCREASE:
            current_noise += perturbation.magnitude
        elif perturbation.kind is PerturbationType.CONNECTIVITY_SCALING:
            current_weights[:, targets] *= perturbation.magnitude
    return current_state, current_weights, current_input, current_noise


def enforce_state_constraints(
    state: np.ndarray,
    *,
    time: float,
    perturbations: tuple[Perturbation, ...],
    n_regions: int,
) -> np.ndarray:
    """Clamp state variables required by active persistent perturbations."""

    constrained = state.copy()
    for perturbation in perturbations:
        if perturbation.kind is not PerturbationType.LESION or not perturbation.active(time):
            continue
        targets = list(perturbation.target_indices)
        constrained[targets] = 0.0
        if len(constrained) >= 2 * n_regions:
            constrained[[index + n_regions for index in targets]] = 0.0
    return constrained
