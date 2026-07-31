"""Shared dynamical model contract."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class DynamicalModel(ABC):
    """Derivative-based model integrated by the simulation runner."""

    @abstractmethod
    def initial_state(self, n_regions: int, rng: np.random.Generator) -> np.ndarray:
        """Create the internal state vector."""

    @abstractmethod
    def derivative(
        self, state: np.ndarray, weights: np.ndarray, external_input: np.ndarray
    ) -> np.ndarray:
        """Compute state derivative."""

    @abstractmethod
    def activity(self, state: np.ndarray, n_regions: int) -> np.ndarray:
        """Extract one regional activity value per region."""

    def state_variables(self, state_history: np.ndarray, n_regions: int) -> dict[str, np.ndarray]:
        """Return optional named state histories."""

        del state_history, n_regions
        return {}
