"""Linear-rate baseline for transparent network dynamics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from mousebrainbench.dynamics.base import DynamicalModel


@dataclass(frozen=True)
class LinearRateModel(DynamicalModel):
    """dx/dt = -x/tau + coupling * W f(x) + input."""

    tau: float = 10.0
    coupling: float = 1.0
    activation: Literal["identity", "tanh", "relu"] = "tanh"
    initial_scale: float = 0.01

    def __post_init__(self) -> None:
        if self.tau <= 0:
            raise ValueError("tau must be positive")

    def initial_state(self, n_regions: int, rng: np.random.Generator) -> np.ndarray:
        return rng.normal(0.0, self.initial_scale, n_regions)

    def derivative(
        self, state: np.ndarray, weights: np.ndarray, external_input: np.ndarray
    ) -> np.ndarray:
        if self.activation == "identity":
            activated = state
        elif self.activation == "relu":
            activated = np.maximum(state, 0.0)
        else:
            activated = np.tanh(state)
        return -state / self.tau + self.coupling * (weights @ activated) + external_input

    def activity(self, state: np.ndarray, n_regions: int) -> np.ndarray:
        return state[:n_regions]
