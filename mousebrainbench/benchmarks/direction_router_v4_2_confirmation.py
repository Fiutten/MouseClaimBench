"""Confirm the stricter association-aware router on fresh v4.2 SEM regimes."""

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
from mousebrainbench.benchmarks.direction_router_v4_confirmation import _metrics
from mousebrainbench.validation.direction_router import (
    DirectionAssumptions,
    association_precondition,
    route_direction,
)

DEFAULT_PROTOCOL = Path("configs/benchmarks/direction_router_v4_2.yaml")
DEFAULT_OUTPUT = Path("results/direction_router_v4_2_confirmation/summary.json")
DEFAULT_MARKDOWN = Path("results/direction_router_v4_2_confirmation/summary.md")

EXPECTED = {
    "independent_beta_exponential": None,
    "independent_bimodal_lognormal": None,
    "direct_linear_logistic": "forward",
    "reverse_linear_triangular": "reverse",
    "direct_arctan_additive": "forward",
    "reverse_saturating_additive": "reverse",
    "confounded_cosine": None,
    "measured_latent_nonlinear": None,
}


def _generate(
    regime: str, n: int, scale: float, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    def normal() -> np.ndarray:
        return rng.normal(0.0, scale, size=n)

    if regime == "independent_beta_exponential":
        return rng.beta(2.0, 5.0, size=n), rng.exponential(size=n)
    if regime == "independent_bimodal_lognormal":
        modes = rng.choice((-1.5, 1.5), size=n)
        return modes + rng.normal(0.0, 0.3, size=n), rng.lognormal(0.0, 0.5, size=n)
    if regime == "direct_linear_logistic":
        x = rng.logistic(size=n)
        return x, 0.85 * x + scale * rng.logistic(size=n)
    if regime == "reverse_linear_triangular":
        y = rng.triangular(-2.0, -0.4, 2.5, size=n)
        return 0.9 * y + scale * rng.uniform(-1.0, 1.0, size=n), y
    if regime == "direct_arctan_additive":
        x = rng.uniform(-3.0, 3.0, size=n)
        return x, 1.6 * np.arctan(x) + normal()
    if regime == "reverse_saturating_additive":
        y = rng.uniform(-2.5, 2.5, size=n)
        return y + 0.35 * np.tanh(2.0 * y) + normal(), y
    if regime == "confounded_cosine":
        z = rng.uniform(-2.5, 2.5, size=n)
        return z + normal(), z + 0.3 * np.cos(z) + normal()
    if regime == "measured_latent_nonlinear":
        latent = rng.normal(size=n)
        return latent + 0.8 * normal(), np.tanh(latent) + normal()
    raise ValueError(f"unknown v4.2 regime: {regime}")


def _assumptions(regime: str, associated: bool) -> DirectionAssumptions:
    return DirectionAssumptions(
        linear=regime in {"direct_linear_logistic", "reverse_linear_triangular"},
        non_gaussian=regime in {"direct_linear_logistic", "reverse_linear_triangular"},
        additive_noise=regime
        in {"direct_arctan_additive", "reverse_saturating_additive"},
        continuous=True,
        acyclic=True,
        hidden_confounding_excluded=regime != "confounded_cosine",
        selection_bias_excluded=True,
        material_measurement_error=regime == "measured_latent_nonlinear",
        association_established=associated,
        provenance=f"frozen v4.2 SEM assumptions:{regime}",
    )


def _one(task: tuple[str, int, float, int, dict[str, Any]]) -> dict[str, Any]:
    regime, n, scale, seed, rule = task
    x, y = _generate(regime, n, scale, np.random.default_rng(seed))
    association = association_precondition(
        x,
        y,
        familywise_alpha=float(rule["familywise_alpha"]),
        minimum_absolute_association=float(rule["minimum_absolute_association"]),
        minimum_passing_tests=int(rule["minimum_passing_tests"]),
    )
    routed = route_direction(
        x,
        y,
        _assumptions(regime, bool(association["established"])),
        seed=seed,
        require_association_precondition=True,
    )
    return {
        "regime": regime,
        "expected": EXPECTED[regime] or "none",
        "prospective_attempted": routed["attempted"],
        "prospective_direction": routed["predicted_direction"],
        "association_established": association["established"],
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
    frozen = yaml.safe_load(protocol.read_text())
    regimes = tuple(frozen["regimes"])
    if regimes != tuple(EXPECTED):
        raise ValueError("v4.2 protocol regimes and truth map differ")
    sample_sizes = (200,) if test_mode else tuple(int(v) for v in frozen["sample_sizes"])
    noise_scales = (1.0,) if test_mode else tuple(float(v) for v in frozen["noise_scales"])
    seeds_per_cell = 1 if test_mode else int(frozen["seeds_per_cell"])
    tasks = []
    for regime_index, regime in enumerate(regimes):
        for n_index, n in enumerate(sample_sizes):
            for scale_index, scale in enumerate(noise_scales):
                offset = regime_index * 100_000 + n_index * 10_000 + scale_index * 1_000
                tasks.extend(
                    (
                        regime,
                        n,
                        scale,
                        int(frozen["seed_namespace"]) + offset + seed,
                        frozen["association_rule"],
                    )
                    for seed in range(seeds_per_cell)
                )
    if workers > 1:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            rows = list(executor.map(_one, tasks, chunksize=1))
    else:
        rows = [_one(task) for task in tasks]
    metrics = _metrics(rows, "prospective")
    independent = [row for row in rows if row["regime"].startswith("independent_")]
    targets = frozen["primary_endpoints"]
    conditions = {
        "scale_matches_protocol": len(rows)
        == len(regimes) * len(sample_sizes) * len(noise_scales) * seeds_per_cell
        and not test_mode,
        "spurious_attempts_equal_0": not any(row["prospective_attempted"] for row in independent),
        "attempted_accuracy_at_least_0_80": metrics["attempted_direction_accuracy"]
        >= float(targets["attempted_direction_accuracy_minimum"]),
        "valid_accuracy_at_least_0_80": metrics["valid_route_accuracy"]
        >= float(targets["valid_route_accuracy_minimum"]),
        "valid_coverage_at_least_0_80": metrics["valid_route_coverage"]
        >= float(targets["valid_route_coverage_minimum"]),
    }
    result = {
        "version": __version__,
        "git_revision": code_revision(),
        "analysis": "direction_router_v4_2_prospective_confirmation",
        "protocol": str(protocol),
        "test_mode": test_mode,
        "cases": len(rows),
        "association_aware_router": metrics,
        "conditions": conditions,
        "primary_passed": all(conditions.values()),
        "wall_time_seconds": time.perf_counter() - started,
        "decision": (
            "strict_association_router_prospectively_confirmed"
            if all(conditions.values())
            else "strict_association_router_not_confirmed"
        ),
        "causal_proof_allowed": False,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    lines = [
        "# Direction router v4.2 prospective confirmation",
        "",
        f"- Decision: `{result['decision']}`",
        f"- Cases: `{result['cases']}`",
        f"- Attempts: `{metrics['attempts']}`",
        f"- Attempted accuracy: `{metrics['attempted_direction_accuracy']:.4f}`",
        f"- Valid-route coverage: `{metrics['valid_route_coverage']:.4f}`",
        f"- Spurious attempts: `{metrics['spurious_attempts_without_reference_direction']}`",
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
