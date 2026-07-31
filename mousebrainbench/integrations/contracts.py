"""Minimal interface for future detailed modules without pretending they exist."""

from __future__ import annotations

from typing import Protocol

import numpy as np


class DetailedModule(Protocol):
    """Boundary for coupling an external detailed simulator to regional dynamics."""

    def run(self, regional_drive: np.ndarray, time: np.ndarray, seed: int) -> np.ndarray:
        """Return a documented regional summary aligned to the supplied time axis."""
