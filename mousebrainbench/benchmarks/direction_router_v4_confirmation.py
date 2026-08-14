"""Prospectively confirm the association-aware direction router on new SEMs."""

from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from mousebrainbench import __version__
from mousebrainbench.artifacts import code_revision
from mousebrainbench.validation.direction_router import (
    DirectionAssumptions,
    association_precondition,
    route_direction,
)

DEFAULT_PROTOCOL = Path("configs/benchmarks/semantic_risk_control_v4.yaml")
DEFAULT_OUTPUT = Path("results/direction_router_v4_confirmation/summary.json")
DEFAULT_MARKDOWN = Path("results/direction_router_v4_confirmation/summary.md")

EXPECTED = {
    "independent_student_t": None,
    "independent_nonlinear_marginals": None,
    "direct_linear_laplace": "forward",
    "reverse_linear_uniform": "reverse",
    "direct_nonlinear_additive": "forward",
    "reverse_nonlinear_additive": "reverse",
    "confounded_nonlinear": None,
    "measurement_error_direct": None,
}


def _generate(
    regime: str, n: int, noise_scale: float, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    def noise() -> np.ndarray:
        return rng.normal(0.0, noise_scale, size=n)

    if regime == "independent_student_t":
        return rng.standard_t(4, size=n), rng.laplace(size=n)
    if regime == "independent_nonlinear_marginals":
        return np.square(rng.normal(size=n)), np.sin(rng.uniform(-3.0, 3.0, size=n))
    if regime == "direct_linear_laplace":
        x = rng.laplace(size=n)
        return x, 0.9 * x + noise_scale * rng.laplace(size=n)
    if regime == "reverse_linear_uniform":
        y = rng.uniform(-2.0, 2.0, size=n)
        return 0.85 * y + noise_scale * rng.uniform(-1.0, 1.0, size=n), y
    if regime == "direct_nonlinear_additive":
        x = rng.uniform(-2.5, 2.5, size=n)
        return x, 1.4 * np.tanh(x) + noise()
    if regime == "reverse_nonlinear_additive":
        y = rng.uniform(-2.5, 2.5, size=n)
        return 1.4 * np.tanh(y) + noise(), y
    if regime == "confounded_nonlinear":
        z = rng.uniform(-2.5, 2.5, size=n)
        return z + noise(), np.tanh(z) + noise()
    if regime == "measurement_error_direct":
        latent = rng.normal(size=n)
        return latent + 0.7 * noise(), 0.9 * latent + noise()
    raise ValueError(f"unknown v4 regime: {regime}")


def _assumptions(regime: str, associated: bool) -> DirectionAssumptions:
    return DirectionAssumptions(
        linear=regime in {"direct_linear_laplace", "reverse_linear_uniform"},
        non_gaussian=regime in {"direct_linear_laplace", "reverse_linear_uniform"},
        additive_noise=regime
        in {
            "independent_student_t",
            "independent_nonlinear_marginals",
            "direct_nonlinear_additive",
            "reverse_nonlinear_additive",
        },
        continuous=True,
        acyclic=True,
        hidden_confounding_excluded=regime != "confounded_nonlinear",
        selection_bias_excluded=True,
        material_measurement_error=regime == "measurement_error_direct",
        association_established=associated,
        provenance=f"frozen v4 SEM assumptions:{regime}",
    )


def _one(task: tuple[str, int, float, int]) -> dict[str, Any]:
    regime, n, noise_scale, seed = task
    x, y = _generate(regime, n, noise_scale, np.random.default_rng(seed))
    association = association_precondition(x, y)
    assumptions = _assumptions(regime, bool(association["established"]))
    legacy = route_direction(
        x, y, assumptions, seed=seed, require_association_precondition=False
    )
    prospective = route_direction(
        x, y, assumptions, seed=seed, require_association_precondition=True
    )
    return {
        "regime": regime,
        "sample_size": n,
        "noise_scale": noise_scale,
        "seed": seed,
        "expected": EXPECTED[regime] or "none",
        "association_established": association["established"],
        "legacy_attempted": legacy["attempted"],
        "legacy_direction": legacy["predicted_direction"],
        "prospective_attempted": prospective["attempted"],
        "prospective_direction": prospective["predicted_direction"],
        "prospective_blockers": prospective["blockers"],
    }


def _metrics(rows: list[dict[str, Any]], prefix: str) -> dict[str, Any]:
    attempted = np.asarray([row[f"{prefix}_attempted"] for row in rows], dtype=bool)
    predicted = np.asarray([row[f"{prefix}_direction"] for row in rows])
    expected = np.asarray([row["expected"] for row in rows])
    identifiable = expected != "none"
    correct = attempted & (predicted == expected)
    valid_attempts = attempted & identifiable
    return {
        "cases": len(rows),
        "attempts": int(attempted.sum()),
        "coverage": float(attempted.mean()),
        "spurious_attempts_without_reference_direction": int((attempted & ~identifiable).sum()),
        "attempted_direction_accuracy": (
            float(correct.sum() / attempted.sum()) if attempted.any() else 0.0
        ),
        "valid_route_accuracy": (
            float(correct[valid_attempts].mean()) if valid_attempts.any() else 0.0
        ),
        "valid_route_coverage": float(valid_attempts.sum() / identifiable.sum()),
    }


def run(
    *,
    protocol: Path = DEFAULT_PROTOCOL,
    output: Path = DEFAULT_OUTPUT,
    markdown: Path = DEFAULT_MARKDOWN,
    workers: int = 1,
    test_mode: bool = False,
) -> Path:
    started = time.perf_counter()
    payload = yaml.safe_load(protocol.read_text())
    frozen = payload["router_confirmation"]
    regimes = tuple(frozen["regimes"])
    if regimes != tuple(EXPECTED):
        raise ValueError("protocol regimes and v4 truth map differ")
    sample_sizes = (200,) if test_mode else tuple(int(v) for v in frozen["sample_sizes"])
    noise_scales = (1.0,) if test_mode else tuple(float(v) for v in frozen["noise_scales"])
    seeds_per_cell = 1 if test_mode else int(frozen["seeds_per_cell"])
    namespace = int(frozen["seed_namespace"])
    tasks = []
    for regime_index, regime in enumerate(regimes):
        for n_index, n in enumerate(sample_sizes):
            for noise_index, noise_scale in enumerate(noise_scales):
                cell = regime_index * 100_000 + n_index * 10_000 + noise_index * 1_000
                tasks.extend(
                    (regime, n, noise_scale, namespace + cell + seed)
                    for seed in range(seeds_per_cell)
                )
    if workers > 1:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            rows = list(executor.map(_one, tasks, chunksize=1))
    else:
        rows = [_one(task) for task in tasks]
    legacy = _metrics(rows, "legacy")
    prospective = _metrics(rows, "prospective")
    independent = [row for row in rows if row["regime"].startswith("independent_")]
    valid_accuracy_drop = legacy["valid_route_accuracy"] - prospective["valid_route_accuracy"]
    conditions = {
        "scale_matches_protocol": len(rows)
        == len(regimes) * len(sample_sizes) * len(noise_scales) * seeds_per_cell
        and not test_mode,
        "spurious_attempts_in_independent_regimes_equal_0": not any(
            row["prospective_attempted"] for row in independent
        ),
        "attempted_direction_accuracy_at_least_0_80": prospective[
            "attempted_direction_accuracy"
        ]
        >= 0.80,
        "valid_route_accuracy_drop_at_most_0_02": valid_accuracy_drop <= 0.02,
    }
    result = {
        "version": __version__,
        "git_revision": code_revision(),
        "analysis": "direction_router_v4_prospective_confirmation",
        "protocol": str(protocol),
        "test_mode": test_mode,
        "cases": len(rows),
        "legacy_router": legacy,
        "association_aware_router": prospective,
        "valid_route_accuracy_drop": valid_accuracy_drop,
        "conditions": conditions,
        "primary_passed": all(conditions.values()),
        "wall_time_seconds": time.perf_counter() - started,
        "decision": (
            "association_aware_router_prospectively_confirmed"
            if all(conditions.values())
            else "association_aware_router_not_confirmed"
        ),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    lines = [
        "# Direction router v4 prospective confirmation",
        "",
        f"- Decision: `{result['decision']}`",
        f"- Cases: `{result['cases']}`",
        "",
        "| Router | Attempts | Accuracy | Spurious attempts | Valid-route coverage |",
        "|---|---:|---:|---:|---:|",
        (
            f"| Legacy | {legacy['attempts']} | "
            f"{legacy['attempted_direction_accuracy']:.4f} | "
            f"{legacy['spurious_attempts_without_reference_direction']} | "
            f"{legacy['valid_route_coverage']:.4f} |"
        ),
        (
            f"| Association-aware | {prospective['attempts']} | "
            f"{prospective['attempted_direction_accuracy']:.4f} | "
            f"{prospective['spurious_attempts_without_reference_direction']} | "
            f"{prospective['valid_route_coverage']:.4f} |"
        ),
        "",
    ]
    markdown.parent.mkdir(parents=True, exist_ok=True)
    markdown.write_text("\n".join(lines))
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--markdown", type=Path, default=DEFAULT_MARKDOWN)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--test-mode", action="store_true")
    args = parser.parse_args()
    path = run(
        protocol=args.protocol,
        output=args.output,
        markdown=args.markdown,
        workers=args.workers,
        test_mode=args.test_mode,
    )
    print(json.dumps({"output": str(path.resolve())}))


if __name__ == "__main__":
    main()
