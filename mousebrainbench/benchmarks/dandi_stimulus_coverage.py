"""Descriptive stimulus coverage for the frozen DANDI chronological split.

This module never changes the predictive endpoint or its authorization rule. It
only reports whether contrast and direction conditions in the held-out segment
were represented in the training segment.
"""

from __future__ import annotations

from typing import Any

import numpy as np


def _stable_level(value: float) -> float:
    return float(np.round(value, 12))


def analyze_stimulus_arrays(
    contrast: np.ndarray,
    direction: np.ndarray,
    *,
    expected_trials: int | None = None,
) -> dict[str, Any]:
    """Summarize train/test condition overlap under the frozen 60/20/20 split."""

    contrast = np.asarray(contrast, dtype=float)
    direction = np.asarray(direction, dtype=float)
    if contrast.shape != direction.shape:
        raise ValueError("contrast and direction must have the same shape")
    finite = np.isfinite(contrast) & np.isfinite(direction)
    contrast = contrast[finite]
    direction = direction[finite]
    trials = len(contrast)
    if expected_trials is not None and trials != expected_trials:
        raise ValueError(
            f"finite stimulus trials ({trials}) do not match the frozen endpoint "
            f"trials ({expected_trials})"
        )
    if trials < 2:
        raise ValueError("at least two finite stimulus trials are required")

    train_stop = int(0.60 * trials)
    test_start = int(0.80 * trials)
    train_pairs = {
        (_stable_level(c), _stable_level(d))
        for c, d in zip(contrast[:train_stop], direction[:train_stop], strict=True)
    }
    test_sequence = [
        (_stable_level(c), _stable_level(d))
        for c, d in zip(contrast[test_start:], direction[test_start:], strict=True)
    ]
    test_pairs = set(test_sequence)
    train_contrasts = {_stable_level(value) for value in contrast[:train_stop]}
    test_contrasts = {_stable_level(value) for value in contrast[test_start:]}
    train_directions = {_stable_level(value) for value in direction[:train_stop]}
    test_directions = {_stable_level(value) for value in direction[test_start:]}
    shared_pairs = train_pairs & test_pairs
    trial_pair_coverage = float(
        np.mean([pair in train_pairs for pair in test_sequence])
    )
    return {
        "finite_trials": trials,
        "train_trials": train_stop,
        "reserved_trials": test_start - train_stop,
        "test_trials": trials - test_start,
        "train_contrasts": sorted(train_contrasts),
        "test_contrasts": sorted(test_contrasts),
        "train_directions": sorted(train_directions),
        "test_directions": sorted(test_directions),
        "train_condition_count": len(train_pairs),
        "test_condition_count": len(test_pairs),
        "shared_test_condition_count": len(shared_pairs),
        "unique_test_condition_coverage": len(shared_pairs) / len(test_pairs),
        "test_trial_condition_coverage": trial_pair_coverage,
        "test_only_contrasts": sorted(test_contrasts - train_contrasts),
        "test_only_directions": sorted(test_directions - train_directions),
        "test_only_conditions": [list(pair) for pair in sorted(test_pairs - train_pairs)],
    }


def aggregate_subject_coverage(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate descriptive coverage without producing an authorization decision."""

    if not rows:
        raise ValueError("at least one subject coverage row is required")
    unique_coverage = np.asarray(
        [row["unique_test_condition_coverage"] for row in rows], dtype=float
    )
    trial_coverage = np.asarray(
        [row["test_trial_condition_coverage"] for row in rows], dtype=float
    )
    return {
        "subjects": len(rows),
        "subjects_with_test_only_contrast": sum(
            bool(row["test_only_contrasts"]) for row in rows
        ),
        "subjects_with_test_only_direction": sum(
            bool(row["test_only_directions"]) for row in rows
        ),
        "subjects_with_test_only_condition": sum(
            bool(row["test_only_conditions"]) for row in rows
        ),
        "unique_test_condition_coverage": {
            "minimum": float(np.min(unique_coverage)),
            "median": float(np.median(unique_coverage)),
            "mean": float(np.mean(unique_coverage)),
        },
        "test_trial_condition_coverage": {
            "minimum": float(np.min(trial_coverage)),
            "median": float(np.median(trial_coverage)),
            "mean": float(np.mean(trial_coverage)),
        },
        "authorization_rule_changed": False,
    }
