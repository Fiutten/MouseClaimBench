"""Validated regional activity extracted from an empirical recording."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass(frozen=True)
class RegionalActivity:
    """Binned population firing rate aligned across named brain regions."""

    time: np.ndarray
    activity_hz: np.ndarray
    region_acronyms: tuple[str, ...]
    unit_counts: tuple[int, ...]
    session_id: int
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        expected = (len(self.time), len(self.region_acronyms))
        if self.activity_hz.shape != expected:
            raise ValueError(f"activity_hz must have shape {expected}")
        if len(self.unit_counts) != len(self.region_acronyms):
            raise ValueError("unit_counts must align with region_acronyms")
        if len(set(self.region_acronyms)) != len(self.region_acronyms):
            raise ValueError("region acronyms must be unique")
        if np.any(self.activity_hz < 0) or not np.all(np.isfinite(self.activity_hz)):
            raise ValueError("firing rates must be finite and non-negative")


@dataclass(frozen=True)
class EvokedRegionalProfile:
    """Baseline-corrected regional response profile split for reliability checks."""

    response_hz: np.ndarray
    odd_event_response_hz: np.ndarray
    even_event_response_hz: np.ndarray
    region_acronyms: tuple[str, ...]
    unit_counts: tuple[int, ...]
    session_id: int
    event_count: int
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        expected = (len(self.region_acronyms),)
        for name, values in (
            ("response_hz", self.response_hz),
            ("odd_event_response_hz", self.odd_event_response_hz),
            ("even_event_response_hz", self.even_event_response_hz),
        ):
            if values.shape != expected or not np.all(np.isfinite(values)):
                raise ValueError(f"{name} must be a finite regional vector")
        if self.event_count < 2:
            raise ValueError("at least two events are required")


@dataclass(frozen=True)
class EvokedRegionalTimecourse:
    """Event-aligned regional population response with odd/even event splits."""

    time: np.ndarray
    activity_hz: np.ndarray
    odd_event_activity_hz: np.ndarray
    even_event_activity_hz: np.ndarray
    region_acronyms: tuple[str, ...]
    unit_counts: tuple[int, ...]
    session_id: int
    event_count: int
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        expected = (len(self.time), len(self.region_acronyms))
        for name, values in (
            ("activity_hz", self.activity_hz),
            ("odd_event_activity_hz", self.odd_event_activity_hz),
            ("even_event_activity_hz", self.even_event_activity_hz),
        ):
            if values.shape != expected or not np.all(np.isfinite(values)):
                raise ValueError(f"{name} must have finite shape {expected}")
        if self.event_count < 2:
            raise ValueError("at least two events are required")
