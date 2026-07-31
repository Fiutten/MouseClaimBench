"""External regional stimulus definitions."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class RegionalStimulus:
    """Constant-amplitude regional stimulus over a half-open time interval."""

    target_indices: tuple[int, ...]
    onset: float
    duration: float
    amplitude: float

    def __post_init__(self) -> None:
        if self.onset < 0 or self.duration <= 0:
            raise ValueError("stimulus onset must be non-negative and duration positive")

    def value(self, time: float, n_regions: int) -> np.ndarray:
        values = np.zeros(n_regions, dtype=float)
        if self.onset <= time < self.onset + self.duration:
            values[list(self.target_indices)] = self.amplitude
        return values


def combined_stimulus(
    stimuli: tuple[RegionalStimulus, ...], time: float, n_regions: int
) -> np.ndarray:
    """Sum all active stimuli at one time point."""

    if not stimuli:
        return np.zeros(n_regions, dtype=float)
    return np.sum([stimulus.value(time, n_regions) for stimulus in stimuli], axis=0)
