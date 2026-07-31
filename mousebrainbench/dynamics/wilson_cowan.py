"""Region-wise Wilson-Cowan excitatory/inhibitory neural-mass model."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from mousebrainbench.dynamics.base import DynamicalModel


@dataclass(frozen=True)
class WilsonCowanModel(DynamicalModel):
    """Two-population neural mass per region with structural E coupling."""

    tau_e: float = 10.0
    tau_i: float = 20.0
    w_ee: float = 1.5
    w_ei: float = 1.0
    w_ie: float = 1.0
    w_ii: float = 0.5
    coupling: float = 0.5
    slope: float = 4.0
    threshold: float = 0.5
    initial_scale: float = 0.02

    def __post_init__(self) -> None:
        if self.tau_e <= 0 or self.tau_i <= 0:
            raise ValueError("time constants must be positive")

    def initial_state(self, n_regions: int, rng: np.random.Generator) -> np.ndarray:
        return np.clip(rng.normal(0.1, self.initial_scale, 2 * n_regions), 0.0, 1.0)

    def derivative(
        self, state: np.ndarray, weights: np.ndarray, external_input: np.ndarray
    ) -> np.ndarray:
        n_regions = weights.shape[0]
        excitatory, inhibitory = state[:n_regions], state[n_regions:]
        network_input = self.coupling * (weights @ excitatory)
        drive_e = self.w_ee * excitatory - self.w_ei * inhibitory + network_input + external_input
        drive_i = self.w_ie * excitatory - self.w_ii * inhibitory
        derivative_e = (-excitatory + self._sigmoid(drive_e)) / self.tau_e
        derivative_i = (-inhibitory + self._sigmoid(drive_i)) / self.tau_i
        return np.concatenate((derivative_e, derivative_i))

    def activity(self, state: np.ndarray, n_regions: int) -> np.ndarray:
        return state[:n_regions]

    def state_variables(self, state_history: np.ndarray, n_regions: int) -> dict[str, np.ndarray]:
        return {
            "excitatory": state_history[:, :n_regions],
            "inhibitory": state_history[:, n_regions:],
        }

    def _sigmoid(self, values: np.ndarray) -> np.ndarray:
        exponent = np.clip(-self.slope * (values - self.threshold), -700.0, 700.0)
        return 1.0 / (1.0 + np.exp(exponent))
