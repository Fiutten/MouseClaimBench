"""Warning-only distribution-shift diagnostics for claim authorization."""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy import stats
from scipy.spatial.distance import cdist


def _holm_rejections(p_values: np.ndarray, alpha: float) -> np.ndarray:
    order = np.argsort(p_values)
    rejected = np.zeros(len(p_values), dtype=bool)
    for rank, index in enumerate(order):
        if p_values[index] <= alpha / (len(p_values) - rank):
            rejected[index] = True
        else:
            break
    return rejected


def _energy_statistic(first: np.ndarray, second: np.ndarray) -> float:
    between = cdist(first, second).mean()
    within_first = cdist(first, first).mean()
    within_second = cdist(second, second).mean()
    return float(2.0 * between - within_first - within_second)


def diagnose_shift(
    calibration: np.ndarray,
    target: np.ndarray,
    *,
    feature_names: tuple[str, ...] | None = None,
    alpha: float = 0.01,
    permutations: int = 199,
    seed: int = 20260804,
) -> dict[str, Any]:
    """Test marginal and multivariate shift without changing certificate validity."""

    first = np.asarray(calibration, dtype=float)
    second = np.asarray(target, dtype=float)
    if first.ndim != 2 or second.ndim != 2 or first.shape[1] != second.shape[1]:
        raise ValueError("calibration and target must be 2D with equal feature count")
    if min(len(first), len(second)) < 2 or not np.all(np.isfinite(first)) or not np.all(np.isfinite(second)):
        raise ValueError("shift diagnostics require at least two finite rows per population")
    names = feature_names or tuple(f"feature_{index}" for index in range(first.shape[1]))
    if len(names) != first.shape[1]:
        raise ValueError("feature_names has incompatible length")

    p_values = np.asarray(
        [stats.ks_2samp(first[:, index], second[:, index]).pvalue for index in range(first.shape[1])]
    )
    rejected = _holm_rejections(p_values, alpha)
    center = first.mean(axis=0)
    scale = first.std(axis=0)
    scale[scale < 1e-12] = 1.0
    standardized_first = (first - center) / scale
    standardized_second = (second - center) / scale
    observed = _energy_statistic(standardized_first, standardized_second)
    combined = np.vstack((standardized_first, standardized_second))
    rng = np.random.default_rng(seed)
    exceedances = 0
    for _ in range(permutations):
        order = rng.permutation(len(combined))
        permuted_first = combined[order[: len(first)]]
        permuted_second = combined[order[len(first) :]]
        exceedances += int(_energy_statistic(permuted_first, permuted_second) >= observed)
    energy_p = (exceedances + 1) / (permutations + 1)
    warning = bool(np.any(rejected) or energy_p <= alpha)
    return {
        "warning": warning,
        "role": "warning_only",
        "may_authorize_or_restore_certificate": False,
        "alpha": alpha,
        "calibration_rows": len(first),
        "target_rows": len(second),
        "marginal_ks": [
            {"feature": name, "p_value": float(p_value), "holm_rejected": bool(reject)}
            for name, p_value, reject in zip(names, p_values, rejected, strict=True)
        ],
        "energy_distance": observed,
        "energy_permutation_p_value": float(energy_p),
        "permutations": permutations,
    }

