"""Simulation calibration for the directed dyadic covariance implementation.

This benchmark is a software and finite-sample diagnostic, not a preregistered
biological endpoint. It compares naive HC1 and dyadic-cluster tests under a
known null with sender/receiver dependence, then measures power under a fixed
positive coefficient.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from scipy import special, stats

from mousebrainbench import __version__
from mousebrainbench.artifacts import code_revision
from mousebrainbench.benchmarks.microns_network_inference import _linear_inference


DEFAULT_OUTPUT = Path("results/dyadic_inference_calibration/summary.json")
DEFAULT_MARKDOWN = Path("results/dyadic_inference_calibration/summary.md")


def _one_trial(
    *,
    n_units: int,
    coefficient: float,
    rng: np.random.Generator,
) -> dict[str, float | bool]:
    pre, post = np.where(~np.eye(n_units, dtype=bool))
    pair_control = rng.normal(size=len(pre))
    sender_propensity = rng.normal(size=n_units)
    receiver_propensity = rng.normal(size=n_units)
    probability = special.expit(
        -2.7
        + 0.65 * sender_propensity[pre]
        + 0.65 * receiver_propensity[post]
        + 0.15 * pair_control
    )
    focal = rng.binomial(1, probability).astype(float)
    sender_error = rng.normal(scale=0.8, size=n_units)
    receiver_error = rng.normal(scale=0.8, size=n_units)
    outcome = (
        coefficient * focal
        + 0.25 * pair_control
        + sender_error[pre]
        + receiver_error[post]
        + rng.normal(scale=0.7, size=len(pre))
    )
    design = np.column_stack((np.ones(len(pre)), focal, pair_control))
    result = _linear_inference(design, outcome, pre, post, n_units)
    estimate = float(result["connected_coefficient"])
    naive_se = float(result["naive_hc1_standard_error"])
    dyadic_se = float(result["dyadic_cluster_standard_error"])
    return {
        "estimate": estimate,
        "naive_se": naive_se,
        "dyadic_se": dyadic_se,
        "naive_reject_two_sided_005": bool(
            2.0 * stats.norm.sf(abs(estimate / naive_se)) < 0.05
        ),
        "dyadic_reject_two_sided_005": bool(
            result["dyadic_cluster_two_sided_p_value"] < 0.05
        ),
        "naive_covers": bool(abs(estimate - coefficient) <= 1.96 * naive_se),
        "dyadic_covers": bool(
            abs(estimate - coefficient)
            <= stats.t.ppf(0.975, df=n_units - 1) * dyadic_se
        ),
    }


def _regime_summary(rows: list[dict[str, float | bool]], coefficient: float) -> dict[str, Any]:
    estimates = np.asarray([row["estimate"] for row in rows], dtype=float)
    naive_se = np.asarray([row["naive_se"] for row in rows], dtype=float)
    dyadic_se = np.asarray([row["dyadic_se"] for row in rows], dtype=float)
    return {
        "true_coefficient": coefficient,
        "trials": len(rows),
        "mean_estimate": float(estimates.mean()),
        "empirical_estimate_standard_deviation": float(estimates.std(ddof=1)),
        "mean_naive_hc1_standard_error": float(naive_se.mean()),
        "mean_dyadic_cluster_standard_error": float(dyadic_se.mean()),
        "naive_rejection_rate_005": float(
            np.mean([row["naive_reject_two_sided_005"] for row in rows])
        ),
        "dyadic_rejection_rate_005": float(
            np.mean([row["dyadic_reject_two_sided_005"] for row in rows])
        ),
        "naive_coverage_95": float(np.mean([row["naive_covers"] for row in rows])),
        "dyadic_coverage_95": float(np.mean([row["dyadic_covers"] for row in rows])),
    }


def run(
    *,
    output: Path = DEFAULT_OUTPUT,
    markdown: Path = DEFAULT_MARKDOWN,
    n_units: int = 70,
    trials: int = 500,
    seed: int = 2_026_080_301,
    alternative_coefficient: float = 0.35,
) -> Path:
    """Estimate null rejection and positive-regime power for both covariance choices."""

    sequence = np.random.SeedSequence(seed)
    null_children, alternative_children = sequence.spawn(2)
    null_rngs = [np.random.default_rng(child) for child in null_children.spawn(trials)]
    alternative_rngs = [
        np.random.default_rng(child) for child in alternative_children.spawn(trials)
    ]
    null_rows = [
        _one_trial(n_units=n_units, coefficient=0.0, rng=rng) for rng in null_rngs
    ]
    alternative_rows = [
        _one_trial(
            n_units=n_units,
            coefficient=alternative_coefficient,
            rng=rng,
        )
        for rng in alternative_rngs
    ]
    null = _regime_summary(null_rows, 0.0)
    alternative = _regime_summary(alternative_rows, alternative_coefficient)
    calibration_sane = bool(
        null["dyadic_rejection_rate_005"] <= 0.08
        and null["dyadic_coverage_95"] >= 0.92
        and alternative["dyadic_rejection_rate_005"] >= 0.80
    )
    payload = {
        "version": __version__,
        "git_revision": code_revision(),
        "analysis": "directed_dyadic_covariance_simulation_calibration",
        "n_units": n_units,
        "directed_pairs_per_trial": n_units * (n_units - 1),
        "trials_per_regime": trials,
        "seed": seed,
        "null": null,
        "alternative": alternative,
        "calibration_sanity_criteria": {
            "maximum_dyadic_null_rejection_rate": 0.08,
            "minimum_dyadic_null_coverage": 0.92,
            "minimum_dyadic_alternative_power": 0.80,
        },
        "calibration_sane": calibration_sane,
        "decision": (
            "dyadic_inference_implementation_passes_simulation_sanity_check"
            if calibration_sane
            else "dyadic_inference_implementation_requires_investigation"
        ),
        "limits": [
            "The calibration DGP covers additive sender and receiver dependence only.",
            "Five hundred trials estimate but do not prove universal Type-I error control.",
            "Network spillovers beyond shared units are outside the dyadic covariance model.",
        ],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2))
    write_markdown(payload, markdown)
    return output


def write_markdown(payload: dict[str, Any], markdown: Path) -> None:
    null = payload["null"]
    alternative = payload["alternative"]
    lines = [
        "# Dyadic inference calibration",
        "",
        f"- Decision: `{payload['decision']}`",
        f"- Units per trial: `{payload['n_units']}`",
        f"- Trials per regime: `{payload['trials_per_regime']}`",
        "",
        "| Regime | Naive rejection | Dyadic rejection | Naive coverage | Dyadic coverage |",
        "|---|---:|---:|---:|---:|",
        f"| Null | {null['naive_rejection_rate_005']:.3f} | "
        f"{null['dyadic_rejection_rate_005']:.3f} | {null['naive_coverage_95']:.3f} | "
        f"{null['dyadic_coverage_95']:.3f} |",
        f"| Positive | {alternative['naive_rejection_rate_005']:.3f} | "
        f"{alternative['dyadic_rejection_rate_005']:.3f} | "
        f"{alternative['naive_coverage_95']:.3f} | "
        f"{alternative['dyadic_coverage_95']:.3f} |",
        "",
    ]
    markdown.parent.mkdir(parents=True, exist_ok=True)
    markdown.write_text("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--markdown", type=Path, default=DEFAULT_MARKDOWN)
    parser.add_argument("--n-units", type=int, default=70)
    parser.add_argument("--trials", type=int, default=500)
    parser.add_argument("--seed", type=int, default=2_026_080_301)
    parser.add_argument("--alternative-coefficient", type=float, default=0.35)
    args = parser.parse_args()
    print(
        json.dumps(
            {
                "output": str(
                    run(
                        output=args.output,
                        markdown=args.markdown,
                        n_units=args.n_units,
                        trials=args.trials,
                        seed=args.seed,
                        alternative_coefficient=args.alternative_coefficient,
                    ).resolve()
                )
            }
        )
    )


if __name__ == "__main__":
    main()

