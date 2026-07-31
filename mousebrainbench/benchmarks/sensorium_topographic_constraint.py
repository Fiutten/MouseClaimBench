"""Topographic structure test for Sensorium response targets.

This module adds a concrete mechanistic constraint that can be evaluated with
the public Sensorium metadata: neurons that are closer in recorded cortical
coordinates should have more similar stimulus-response tuning than expected
after shuffling coordinates. Passing this test is not a full causal mechanism,
but it is a verifiable structural constraint and stronger evidence than
prediction alone.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import median
from typing import Any

import numpy as np
from scipy.stats import spearmanr


def _numeric_key(path: Path) -> tuple[int, str]:
    try:
        return int(path.stem), path.name
    except ValueError:
        return 10**12, path.name


def _trial_file_map(directory: Path) -> dict[int, Path]:
    files: dict[int, Path] = {}
    for path in directory.glob("*.npy"):
        try:
            files[int(path.stem)] = path
        except ValueError:
            continue
    if not files:
        raise FileNotFoundError(f"no numeric .npy trial files found in {directory}")
    return files


def _response_vector(response: np.ndarray) -> np.ndarray:
    values = np.asarray(response, dtype=float)
    if values.ndim == 1:
        return np.nan_to_num(values, nan=0.0)
    if values.ndim == 2:
        return np.nan_to_num(np.nanmean(values, axis=1), nan=0.0)
    collapsed = np.nanmean(values.reshape(values.shape[0], -1), axis=1)
    return np.nan_to_num(collapsed, nan=0.0)


def _load_tuning_matrix(root: Path, *, eval_tier: str) -> tuple[np.ndarray, np.ndarray, int]:
    """Load neuron x stimulus tuning for one Sensorium directory."""

    response_files = _trial_file_map(root / "data" / "responses")
    trial_keys = np.asarray(sorted(response_files), dtype=int)
    tiers_all = np.load(root / "meta" / "trials" / "tiers.npy", allow_pickle=True).astype(str)
    stimulus_ids_all = np.load(
        root / "meta" / "trials" / "frame_image_id.npy", allow_pickle=True
    )
    valid = trial_keys[trial_keys < len(tiers_all)]
    valid = valid[tiers_all[valid] == eval_tier]
    if len(valid) == 0:
        raise ValueError(f"no trials found for eval tier {eval_tier!r} in {root}")

    responses = np.stack([_response_vector(np.load(response_files[int(key)])) for key in valid])
    stimulus_ids = stimulus_ids_all[valid]
    unique_ids = np.asarray(sorted(set(stimulus_ids.tolist())))
    tuning = np.zeros((len(unique_ids), responses.shape[1]), dtype=float)
    repeated = 0
    for index, stimulus_id in enumerate(unique_ids):
        mask = stimulus_ids == stimulus_id
        repeated += int(np.sum(mask) > 1)
        tuning[index] = responses[mask].mean(axis=0)
    return tuning.T, unique_ids, repeated


def _zscore_rows(values: np.ndarray) -> np.ndarray:
    mean = values.mean(axis=1, keepdims=True)
    std = values.std(axis=1, keepdims=True)
    std[std == 0] = 1.0
    return (values - mean) / std


def _sample_pairs(n_items: int, pair_count: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Sample unique-ish unordered item pairs without materializing all pairs."""

    left = rng.integers(0, n_items, size=pair_count)
    right = rng.integers(0, n_items - 1, size=pair_count)
    right = right + (right >= left)
    return left, right


def _similarity_for_pairs(z_tuning: np.ndarray, left: np.ndarray, right: np.ndarray) -> np.ndarray:
    return np.mean(z_tuning[left] * z_tuning[right], axis=1)


def _distance_for_pairs(coords: np.ndarray, left: np.ndarray, right: np.ndarray) -> np.ndarray:
    return np.linalg.norm(coords[left] - coords[right], axis=1)


def _safe_spearman(left: np.ndarray, right: np.ndarray) -> float:
    result = spearmanr(left, right)
    value = float(result.statistic)
    if np.isnan(value):
        return 0.0
    return value


def evaluate_topographic_constraint(
    root: str | Path,
    *,
    eval_tier: str = "test",
    max_neurons: int = 1500,
    pair_count: int = 50000,
    null_repeats: int = 200,
    seed: int = 17,
    min_effect: float = 0.03,
) -> dict[str, Any]:
    """Evaluate whether response tuning similarity follows cortical distance."""

    base = Path(root)
    coords = np.load(base / "meta" / "neurons" / "cell_motor_coordinates.npy").astype(float)
    tuning, stimulus_ids, repeated_stimuli = _load_tuning_matrix(base, eval_tier=eval_tier)
    n_neurons = min(coords.shape[0], tuning.shape[0])
    coords = coords[:n_neurons]
    tuning = tuning[:n_neurons]
    finite = np.isfinite(coords).all(axis=1) & np.isfinite(tuning).all(axis=1)
    coords = coords[finite]
    tuning = tuning[finite]
    if len(coords) < 10:
        raise ValueError("topographic constraint requires at least 10 neurons")

    rng = np.random.default_rng(seed)
    if len(coords) > max_neurons:
        selected = rng.choice(len(coords), size=max_neurons, replace=False)
        coords = coords[selected]
        tuning = tuning[selected]

    z_tuning = _zscore_rows(tuning)
    left, right = _sample_pairs(len(coords), pair_count, rng)
    similarities = _similarity_for_pairs(z_tuning, left, right)
    distances = _distance_for_pairs(coords, left, right)
    observed = _safe_spearman(-distances, similarities)

    null_values: list[float] = []
    for _ in range(null_repeats):
        permutation = rng.permutation(len(coords))
        shuffled_distances = _distance_for_pairs(coords[permutation], left, right)
        null_values.append(_safe_spearman(-shuffled_distances, similarities))
    null_array = np.asarray(null_values, dtype=float)
    null_p95 = float(np.percentile(null_array, 95))
    null_median = float(np.median(null_array))
    p_value = float((np.sum(null_array >= observed) + 1) / (len(null_array) + 1))
    passed = bool(observed > null_p95 and observed >= min_effect)
    return {
        "dataset": base.name,
        "root": str(base),
        "eval_tier": eval_tier,
        "n_neurons": int(len(coords)),
        "n_stimuli": int(len(stimulus_ids)),
        "repeated_stimuli": int(repeated_stimuli),
        "pair_count": int(pair_count),
        "null_repeats": int(null_repeats),
        "observed_spearman_similarity_vs_inverse_distance": observed,
        "null_median": null_median,
        "null_p95": null_p95,
        "permutation_p_value": p_value,
        "effect_over_null_median": float(observed - null_median),
        "min_effect": float(min_effect),
        "passed": passed,
        "interpretation": (
            "Pass means empirical neural tuning similarity is anatomically "
            "topographic relative to shuffled coordinates. This is a structural "
            "constraint, not by itself a causal circuit model."
        ),
    }


def summarize_topographic_constraints(rows: list[dict[str, Any]]) -> dict[str, Any]:
    effects = [float(row["observed_spearman_similarity_vs_inverse_distance"]) for row in rows]
    over_null = [float(row["effect_over_null_median"]) for row in rows]
    return {
        "benchmark": "sensorium_topographic_structural_constraint",
        "n_datasets": len(rows),
        "passed_count": sum(bool(row["passed"]) for row in rows),
        "median_observed_spearman": median(effects) if effects else None,
        "median_effect_over_null": median(over_null) if over_null else None,
        "rows": rows,
        "decision": (
            "structural_constraint_supported"
            if rows and sum(bool(row["passed"]) for row in rows) == len(rows)
            else "structural_constraint_mixed_or_failed"
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("roots", nargs="+", type=Path)
    parser.add_argument("--eval-tier", default="test")
    parser.add_argument("--max-neurons", type=int, default=1500)
    parser.add_argument("--pair-count", type=int, default=50000)
    parser.add_argument("--null-repeats", type=int, default=200)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--min-effect", type=float, default=0.03)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/sensorium_topographic_constraint/summary.json"),
    )
    args = parser.parse_args()
    rows = [
        evaluate_topographic_constraint(
            root,
            eval_tier=args.eval_tier,
            max_neurons=args.max_neurons,
            pair_count=args.pair_count,
            null_repeats=args.null_repeats,
            seed=args.seed,
            min_effect=args.min_effect,
        )
        for root in args.roots
    ]
    payload = summarize_topographic_constraints(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2))
    print(json.dumps({"output": str(args.output.resolve())}))


if __name__ == "__main__":
    main()
