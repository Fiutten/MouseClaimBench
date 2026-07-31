"""Synthetic truth-known benchmark for mechanistic-identifiability gates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from mousebrainbench import __version__
from mousebrainbench.artifacts import code_revision
from mousebrainbench.validation.mechanistic_identifiability import (
    Criterion,
    build_mis_from_blocks,
)


def _correlation(left: np.ndarray, right: np.ndarray) -> float:
    if np.std(left.ravel()) == 0 or np.std(right.ravel()) == 0:
        return 0.0
    return float(np.corrcoef(left.ravel(), right.ravel())[0, 1])


def _make_template(
    *,
    n_regions: int,
    n_time: int,
    directed_latency: bool,
    region_specific_amplitude: bool,
) -> np.ndarray:
    time = np.linspace(0.0, 1.0, n_time)
    template = np.zeros((n_time, n_regions), dtype=float)
    for region in range(n_regions):
        center = 0.18 + 0.08 * region if directed_latency else 0.25
        amplitude = 1.0 + 0.25 * region if region_specific_amplitude else 1.0
        width = 0.003 + 0.0015 * region if region_specific_amplitude else 0.006
        template[:, region] = amplitude * np.exp(-((time - center) ** 2) / width)
    return template


def _pairwise_latency_fraction(template: np.ndarray) -> float:
    latencies = np.argmax(template, axis=0)
    differences = np.abs(latencies[:, None] - latencies[None, :])
    mask = np.triu(np.ones_like(differences, dtype=bool), k=1)
    return float(np.mean(differences[mask] >= 1))


def _synthetic_case(
    *,
    name: str,
    directed_latency: bool,
    region_specific_amplitude: bool,
    topology_specific_prediction: bool = True,
    seed: int,
    n_sessions: int = 24,
    n_regions: int = 6,
    n_time: int = 24,
    observation_noise: float = 0.08,
    split_noise: float = 0.04,
    prediction_noise: float = 0.02,
    n_null_per_session: int = 200,
    n_permutation_predictions: int = 100,
) -> dict[str, object]:
    """Generate one truth-known synthetic case and score it with MIS.

    The extra noise and sampling parameters support the MIS 2.0 calibration
    benchmark. Defaults preserve the original four-case benchmark used by the
    submitted manuscript.
    """

    rng = np.random.default_rng(seed)
    template = _make_template(
        n_regions=n_regions,
        n_time=n_time,
        directed_latency=directed_latency,
        region_specific_amplitude=region_specific_amplitude,
    )
    sessions = np.stack(
        [template + rng.normal(0.0, observation_noise, template.shape) for _ in range(n_sessions)]
    )
    odd = sessions + rng.normal(0.0, split_noise, sessions.shape)
    even = sessions + rng.normal(0.0, split_noise, sessions.shape)
    reference = sessions.mean(axis=0)
    reproducibility = np.asarray([_correlation(session, reference) for session in sessions])
    split = np.asarray([_correlation(left, right) for left, right in zip(odd, even, strict=True)])
    null = np.asarray(
        [
            [
                _correlation(session[:, rng.permutation(n_regions)], reference)
                for _ in range(n_null_per_session)
            ]
            for session in sessions
        ]
    )

    disconnected_prediction = _make_template(
        n_regions=n_regions,
        n_time=n_time,
        directed_latency=False,
        region_specific_amplitude=False,
    )
    true_prediction_template = template if topology_specific_prediction else disconnected_prediction
    true_prediction = true_prediction_template + rng.normal(0.0, prediction_noise, template.shape)
    permutation_predictions = np.stack(
        [true_prediction[:, rng.permutation(n_regions)] for _ in range(n_permutation_predictions)]
    )
    true_scores = np.asarray([_correlation(true_prediction, session) for session in sessions])
    disconnected_scores = np.asarray(
        [_correlation(disconnected_prediction, session) for session in sessions]
    )
    permutation_medians = np.asarray(
        [
            np.median([_correlation(prediction, session) for session in sessions])
            for prediction in permutation_predictions
        ]
    )

    latency_fraction = _pairwise_latency_fraction(template)
    nonzero_lead_lag = latency_fraction
    mis = build_mis_from_blocks(
        reproducibility=(
            Criterion("median_cross_session_correlation", float(np.median(reproducibility)), 0.70),
            Criterion("median_split_half_correlation", float(np.median(split)), 0.80),
            Criterion(
                "fraction_above_own_null_95th",
                float(np.mean(reproducibility > np.quantile(null, 0.95, axis=1))),
                0.50,
                "gte",
            ),
        ),
        topology_specificity=(
            Criterion(
                "true_minus_median_permutation",
                float(np.median(true_scores) - np.median(permutation_medians)),
                0.05,
            ),
            Criterion(
                "fraction_permutations_outperformed",
                float(np.mean(np.median(true_scores) > permutation_medians)),
                0.95,
                "gte",
            ),
            Criterion(
                "true_minus_disconnected",
                float(np.median(true_scores) - np.median(disconnected_scores)),
                0.05,
            ),
        ),
        directed_identifiability=(
            Criterion("resolvable_latency_pair_fraction", latency_fraction, 0.50, "gte"),
            Criterion("nonzero_lead_lag_pair_fraction", nonzero_lead_lag, 0.25, "gte"),
        ),
    )
    return {
        "case": name,
        "ground_truth_directed": directed_latency,
        "ground_truth_region_specific": region_specific_amplitude,
        "mis": mis.as_dict(),
        "summary": {
            "median_reproducibility": float(np.median(reproducibility)),
            "median_split_half": float(np.median(split)),
            "median_true_prediction": float(np.median(true_scores)),
            "median_disconnected_prediction": float(np.median(disconnected_scores)),
            "median_permutation_prediction": float(np.median(permutation_medians)),
            "resolvable_latency_pair_fraction": latency_fraction,
        },
    }


def run(output: str | Path = "results/synthetic_identifiability_benchmark.json") -> Path:
    """Run truth-known identifiability cases with positive and failure controls."""

    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    results = {
        "version": __version__,
        "git_revision": code_revision(),
        "cases": [
            _synthetic_case(
                name="directed_truth",
                directed_latency=True,
                region_specific_amplitude=True,
                seed=11,
            ),
            _synthetic_case(
                name="common_drive_nonidentifiable",
                directed_latency=False,
                region_specific_amplitude=False,
                seed=23,
            ),
            _synthetic_case(
                name="topology_without_direction",
                directed_latency=False,
                region_specific_amplitude=True,
                seed=31,
            ),
            _synthetic_case(
                name="direction_without_topology_specificity",
                directed_latency=True,
                region_specific_amplitude=False,
                topology_specific_prediction=False,
                seed=43,
            ),
        ],
        "interpretation": (
            "Only directed_truth should pass the full conjunctive MIS. The other "
            "cases isolate common drive, structure without temporal direction, and "
            "direction-like timing without region-specific topology."
        ),
    }
    path.write_text(json.dumps(results, indent=2))
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/synthetic_identifiability_benchmark.json"),
    )
    args = parser.parse_args()
    print(json.dumps({"output": str(run(args.output).resolve())}))


if __name__ == "__main__":
    main()
