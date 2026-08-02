"""Content-validity statistics for a completed independent expert panel."""

from __future__ import annotations

import math
from typing import Any

import numpy as np


def item_cvi(ratings: np.ndarray, *, valid_at: int = 3) -> float:
    """Return the proportion of experts assigning a content-valid rating."""

    values = np.asarray(ratings, dtype=int)
    if values.ndim != 1 or len(values) == 0:
        raise ValueError("item CVI requires a non-empty rating vector")
    if np.any((values < 1) | (values > 4)):
        raise ValueError("content-validity ratings must lie between one and four")
    return float(np.mean(values >= valid_at))


def modified_kappa(ratings: np.ndarray, *, valid_at: int = 3) -> float:
    """Return chance-corrected agreement for one binary-validity item."""

    values = np.asarray(ratings, dtype=int)
    cvi = item_cvi(values, valid_at=valid_at)
    experts = len(values)
    agreeing = int(np.sum(values >= valid_at))
    chance = math.comb(experts, agreeing) * (0.5**experts)
    return float((cvi - chance) / (1.0 - chance)) if chance < 1.0 else 0.0


def binary_fleiss_kappa(matrix: np.ndarray, *, valid_at: int = 3) -> float:
    """Measure agreement across items after the declared valid/invalid split."""

    values = np.asarray(matrix, dtype=int)
    if values.ndim != 2 or min(values.shape) == 0:
        raise ValueError("Fleiss kappa requires items by raters")
    if np.any((values < 1) | (values > 4)):
        raise ValueError("content-validity ratings must lie between one and four")
    raters = values.shape[1]
    if raters < 2:
        raise ValueError("Fleiss kappa requires at least two raters")
    valid = values >= valid_at
    yes = valid.sum(axis=1)
    no = raters - yes
    observed = np.mean((yes * (yes - 1) + no * (no - 1)) / (raters * (raters - 1)))
    prevalence = float(valid.mean())
    expected = prevalence**2 + (1.0 - prevalence) ** 2
    return float((observed - expected) / (1.0 - expected)) if expected < 1.0 else 1.0


def dimension_summary(matrix: np.ndarray, item_ids: tuple[str, ...]) -> dict[str, Any]:
    """Summarize item CVI, modified kappa, and scale-average CVI."""

    values = np.asarray(matrix, dtype=int)
    if values.shape[0] != len(item_ids):
        raise ValueError("item identifiers do not match the rating matrix")
    rows = {
        item_id: {
            "item_cvi": item_cvi(values[index]),
            "modified_kappa": modified_kappa(values[index]),
        }
        for index, item_id in enumerate(item_ids)
    }
    return {
        "items": rows,
        "scale_average_cvi": float(np.mean([row["item_cvi"] for row in rows.values()])),
        "binary_fleiss_kappa": binary_fleiss_kappa(values),
    }
