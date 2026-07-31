"""Calibrate and externally test mesoscopic models against the confirmed Allen target."""

from __future__ import annotations

import argparse
from hashlib import sha256
from itertools import product
import json
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from mousebrainbench import __version__
from mousebrainbench.artifacts import code_revision
from mousebrainbench.benchmarks.allen_vbn_phase2b import _run_id
from mousebrainbench.benchmarks.allen_vbn_phase2c import transform_timecourse
from mousebrainbench.connectivity.normalization import normalize_connectivity
from mousebrainbench.data.loaders.allen_connectivity import load_allen_visual_connectivity
from mousebrainbench.data.loaders.allen_vbn import AllenVBNRepository
from mousebrainbench.dynamics.linear_rate import LinearRateModel
from mousebrainbench.dynamics.wilson_cowan import WilsonCowanModel


def load_config(path: str | Path) -> dict[str, Any]:
    source = Path(path).resolve()
    config = yaml.safe_load(source.read_text())
    config["_config_path"] = str(source)
    return config


def _file_sha256(path: str | Path) -> str:
    return sha256(Path(path).read_bytes()).hexdigest()


def _correlation(left: np.ndarray, right: np.ndarray) -> float:
    left_flat, right_flat = left.ravel(), right.ravel()
    if np.std(left_flat) == 0 or np.std(right_flat) == 0:
        return 0.0
    return float(np.corrcoef(left_flat, right_flat)[0, 1])


def _extract_targets(
    repository: AllenVBNRepository,
    session_ids: list[int],
    config: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray]:
    """Extract the frozen temporal target; any missing session aborts the cohort."""

    target, data = config["target"], config["data"]
    transformed, times = [], []
    for session_id in session_ids:
        timecourse = repository.extract_change_response_timecourse(
            session_id,
            tuple(data["regions"]),
            min_units_per_region=int(data["min_units_per_region"]),
            start_seconds=float(target["start_seconds"]),
            stop_seconds=float(target["stop_seconds"]),
            bin_size_seconds=float(target["bin_size_seconds"]),
        )
        transformed.append(
            transform_timecourse(
                timecourse.activity_hz,
                timecourse.time,
                str(target["transformation"]),
            )
        )
        times.append(timecourse.time)
    if not all(np.array_equal(times[0], time) for time in times[1:]):
        raise ValueError("empirical target time axes are not identical")
    return times[0], np.stack(transformed)


def _candidate_parameters(config: dict[str, Any], family: str) -> list[dict[str, float]]:
    grid = config["models"][family]
    names = tuple(grid)
    return [
        {name: float(value) for name, value in zip(names, values, strict=True)}
        for values in product(*(grid[name] for name in names))
    ]


def _model(family: str, parameters: dict[str, float]) -> tuple[Any, float]:
    values = {key: value for key, value in parameters.items() if key != "drive_scale"}
    if family == "linear_rate":
        return LinearRateModel(**values), parameters["drive_scale"]
    if family == "wilson_cowan":
        return WilsonCowanModel(**values), parameters["drive_scale"]
    raise ValueError(f"unknown model family: {family}")


def _rk4_step(model: Any, state: np.ndarray, weights: np.ndarray, drive: np.ndarray, dt: float) -> np.ndarray:
    k1 = model.derivative(state, weights, drive)
    k2 = model.derivative(state + dt * k1 / 2, weights, drive)
    k3 = model.derivative(state + dt * k2 / 2, weights, drive)
    k4 = model.derivative(state + dt * k3, weights, drive)
    return state + dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6


def simulate_target(
    family: str,
    parameters: dict[str, float],
    weights: np.ndarray,
    drive: np.ndarray,
    time: np.ndarray,
    transformation: str,
) -> np.ndarray:
    """Generate a deterministic evoked response using the same target transform."""

    model, drive_scale = _model(family, parameters)
    n_regions = weights.shape[0]
    dt = float(np.median(np.diff(time)))
    if family == "linear_rate":
        state = np.zeros(n_regions, dtype=float)
    else:
        state = np.full(2 * n_regions, 0.1, dtype=float)
        for _ in range(200):
            state = _rk4_step(model, state, weights, np.zeros(n_regions), dt)
    onset_index = int(np.flatnonzero(time > 0)[0])
    activity = np.empty((len(time), n_regions), dtype=float)
    for index in range(len(time)):
        external = drive * drive_scale if index == onset_index else np.zeros(n_regions)
        state = _rk4_step(model, state, weights, external, dt)
        activity[index] = model.activity(state, n_regions)
    if not np.all(np.isfinite(activity)):
        raise FloatingPointError("model produced non-finite activity")
    return transform_timecourse(activity, time, transformation)


def _scores(prediction: np.ndarray, targets: np.ndarray, time: np.ndarray, start: float) -> np.ndarray:
    mask = time >= start
    return np.asarray([_correlation(prediction[mask], target[mask]) for target in targets])


def _calibrate(
    family: str,
    weights: np.ndarray,
    drive: np.ndarray,
    time: np.ndarray,
    targets: np.ndarray,
    config: dict[str, Any],
) -> dict[str, Any]:
    best: dict[str, Any] | None = None
    for parameters in _candidate_parameters(config, family):
        prediction = simulate_target(
            family, parameters, weights, drive, time, config["target"]["transformation"]
        )
        scores = _scores(
            prediction,
            targets,
            time,
            float(config["target"]["evaluation_start_seconds"]),
        )
        candidate = {
            "parameters": parameters,
            "median_correlation": float(np.median(scores)),
            "mean_correlation": float(np.mean(scores)),
        }
        if best is None or (
            candidate["median_correlation"],
            candidate["mean_correlation"],
            json.dumps(parameters, sort_keys=True),
        ) > (
            best["median_correlation"],
            best["mean_correlation"],
            json.dumps(best["parameters"], sort_keys=True),
        ):
            best = candidate
    assert best is not None
    return best


def _permuted_weights(weights: np.ndarray, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    result = weights.copy()
    off_diagonal = ~np.eye(len(weights), dtype=bool)
    result[off_diagonal] = rng.permutation(result[off_diagonal])
    return result


def _derive_drive(targets: np.ndarray, time: np.ndarray) -> np.ndarray:
    onset = int(np.flatnonzero(time > 0)[0])
    drive = targets.mean(axis=0)[onset].copy()
    scale = float(np.max(np.abs(drive)))
    if scale == 0:
        raise ValueError("development-derived drive is zero")
    return drive / scale


def _load_weights(config: dict[str, Any]) -> tuple[np.ndarray, dict[str, Any]]:
    _, connectivity, payload = load_allen_visual_connectivity(
        config["data"]["connectivity_path"], tuple(config["data"]["regions"])
    )
    normalized = normalize_connectivity(connectivity, "spectral_radius")
    return normalized.dense_weights(), payload


def run_development(config: dict[str, Any]) -> Path:
    repository = AllenVBNRepository(config["data"]["vbn_root"])
    session_ids = [int(value) for value in config["data"]["development_session_ids"]]
    time, targets = _extract_targets(repository, session_ids, config)
    weights, connectivity_payload = _load_weights(config)
    drive = _derive_drive(targets, time)
    family_results = {
        family: _calibrate(family, weights, drive, time, targets, config)
        for family in config["models"]
    }
    selected_family = max(
        family_results,
        key=lambda family: (
            family_results[family]["median_correlation"],
            family_results[family]["mean_correlation"],
            family,
        ),
    )
    disconnected = _calibrate(selected_family, np.zeros_like(weights), drive, time, targets, config)
    transposed = _calibrate(selected_family, weights.T, drive, time, targets, config)
    seeds = [
        int(config["analysis"]["seed"]) + index + 1
        for index in range(int(config["analysis"]["permutation_graphs"]))
    ]
    permutations = [
        {"seed": seed, **_calibrate(selected_family, _permuted_weights(weights, seed), drive, time, targets, config)}
        for seed in seeds
    ]
    allen = family_results[selected_family]
    metrics = {
        "interpretation": "phase3_development_only",
        "sessions": len(session_ids),
        "selected_family": selected_family,
        "family_results": family_results,
        "controls": {
            "disconnected": disconnected,
            "transposed": transposed,
            "permutations": permutations,
        },
        "development_diagnostics": {
            "allen_minus_disconnected": allen["median_correlation"] - disconnected["median_correlation"],
            "allen_minus_transposed": allen["median_correlation"] - transposed["median_correlation"],
            "fraction_permutations_outperformed": float(
                np.mean(
                    allen["median_correlation"]
                    > np.asarray([item["median_correlation"] for item in permutations])
                )
            ),
        },
        "connectivity_experiment_counts": connectivity_payload["experiment_counts_by_source"],
    }
    output = Path(config["output"]["root"]) / f"{config['output']['development_name']}-{_run_id(config)}"
    output.mkdir(parents=True, exist_ok=True)
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2))
    (output / "provenance.json").write_text(
        json.dumps({"version": __version__, "git_revision": code_revision()}, indent=2)
    )
    np.savez_compressed(
        output / "development_arrays.npz",
        session_ids=np.asarray(session_ids),
        time=time,
        targets=targets,
        drive=drive,
        weights=weights,
    )
    return output


def seal_confirmation(config: dict[str, Any], development_output: str | Path) -> Path:
    development_output = Path(development_output)
    metrics = json.loads((development_output / "metrics.json").read_text())
    arrays = np.load(development_output / "development_arrays.npz")
    phase2c = json.loads(Path(config["data"]["confirmation_plan"]).read_text())
    family = metrics["selected_family"]
    plan = {
        "sealed_before_confirmation": True,
        "connectivity_sha256": _file_sha256(config["data"]["connectivity_path"]),
        "regions": config["data"]["regions"],
        "target": config["target"],
        "thresholds": config["analysis"]["thresholds"],
        "selected_family": family,
        "drive": arrays["drive"].tolist(),
        "model_parameters": metrics["family_results"][family]["parameters"],
        "control_parameters": {
            "disconnected": metrics["controls"]["disconnected"]["parameters"],
            "transposed": metrics["controls"]["transposed"]["parameters"],
            "permutations": [
                {"seed": item["seed"], "parameters": item["parameters"]}
                for item in metrics["controls"]["permutations"]
            ],
        },
        "development_session_ids": config["data"]["development_session_ids"],
        "confirmation_sessions": phase2c["sessions"],
        "development_output": str(development_output.resolve()),
    }
    destination = Path(config["confirmation"]["sealed_plan"])
    destination.write_text(json.dumps(plan, indent=2))
    return destination


def _bootstrap_median_interval(values: np.ndarray, samples: int, seed: int) -> list[float]:
    rng = np.random.default_rng(seed)
    medians = np.asarray(
        [np.median(rng.choice(values, size=len(values), replace=True)) for _ in range(samples)]
    )
    return [float(value) for value in np.quantile(medians, [0.025, 0.975])]


def _validate_plan(plan: dict[str, Any], config: dict[str, Any]) -> None:
    if not plan.get("sealed_before_confirmation"):
        raise ValueError("phase3 plan is not sealed")
    if plan["target"] != config["target"] or plan["thresholds"] != config["analysis"]["thresholds"]:
        raise ValueError("phase3 configuration differs from the sealed plan")
    if plan["connectivity_sha256"] != _file_sha256(config["data"]["connectivity_path"]):
        raise ValueError("Allen connectivity file differs from the sealed plan")


def run_confirmation(config: dict[str, Any]) -> Path:
    plan = json.loads(Path(config["confirmation"]["sealed_plan"]).read_text())
    _validate_plan(plan, config)
    output = Path(config["output"]["root"]) / f"{config['output']['confirmation_name']}-{_run_id(config)}"
    if (output / "metrics.json").exists():
        raise RuntimeError("phase3 confirmation has already been executed")
    repository = AllenVBNRepository(config["data"]["vbn_root"])
    sessions = plan["confirmation_sessions"]
    session_ids = [int(item["session_id"]) for item in sessions]
    mouse_ids = np.asarray([int(item["mouse_id"]) for item in sessions])
    time, targets = _extract_targets(repository, session_ids, config)
    weights, _ = _load_weights(config)
    drive = np.asarray(plan["drive"], dtype=float)
    family = plan["selected_family"]

    def evaluate(matrix: np.ndarray, parameters: dict[str, float]) -> np.ndarray:
        prediction = simulate_target(
            family, parameters, matrix, drive, time, config["target"]["transformation"]
        )
        return _scores(
            prediction, targets, time, float(config["target"]["evaluation_start_seconds"])
        )

    allen_scores = evaluate(weights, plan["model_parameters"])
    disconnected_scores = evaluate(np.zeros_like(weights), plan["control_parameters"]["disconnected"])
    transposed_scores = evaluate(weights.T, plan["control_parameters"]["transposed"])
    permutation_scores = np.stack(
        [
            evaluate(
                _permuted_weights(weights, int(item["seed"])),
                item["parameters"],
            )
            for item in plan["control_parameters"]["permutations"]
        ]
    )
    allen_median = float(np.median(allen_scores))
    permutation_medians = np.median(permutation_scores, axis=1)
    paired_advantage = allen_scores - np.median(permutation_scores, axis=0)
    interval = _bootstrap_median_interval(
        paired_advantage,
        int(config["analysis"]["bootstrap_samples"]),
        int(config["analysis"]["seed"]),
    )
    thresholds = plan["thresholds"]
    checks = {
        "median_confirmation_correlation_passed": allen_median
        > thresholds["median_confirmation_correlation_gt"],
        "allen_minus_median_permutation_passed": allen_median - float(np.median(permutation_medians))
        > thresholds["allen_minus_median_permutation_gt"],
        "fraction_permutations_outperformed_passed": float(np.mean(allen_median > permutation_medians))
        >= thresholds["fraction_permutations_outperformed_gte"],
        "paired_bootstrap_lower_bound_passed": interval[0]
        > thresholds["paired_bootstrap_lower_bound_gt"],
        "allen_minus_disconnected_passed": allen_median - float(np.median(disconnected_scores))
        > thresholds["allen_minus_disconnected_gt"],
    }
    passed = all(checks.values())
    metrics = {
        "interpretation": "single_external_phase3_confirmation",
        "selected_family": family,
        "sessions": len(session_ids),
        "unique_mice": len(set(mouse_ids)),
        "allen_median_correlation": allen_median,
        "disconnected_median_correlation": float(np.median(disconnected_scores)),
        "transposed_median_correlation": float(np.median(transposed_scores)),
        "median_permutation_correlation": float(np.median(permutation_medians)),
        "allen_minus_median_permutation": allen_median - float(np.median(permutation_medians)),
        "fraction_permutations_outperformed": float(np.mean(allen_median > permutation_medians)),
        "paired_advantage_bootstrap_95_interval": interval,
        "allen_minus_disconnected": allen_median - float(np.median(disconnected_scores)),
        **checks,
        "confirmation_passed": passed,
        "decision": "anatomical_connectivity_predictive" if passed else "anatomical_connectivity_not_supported",
    }
    output.mkdir(parents=True, exist_ok=False)
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2))
    (output / "sealed_plan.json").write_text(json.dumps(plan, indent=2))
    (output / "provenance.json").write_text(
        json.dumps({"version": __version__, "git_revision": code_revision()}, indent=2)
    )
    np.savez_compressed(
        output / "confirmation_arrays.npz",
        session_ids=np.asarray(session_ids),
        mouse_ids=mouse_ids,
        allen_scores=allen_scores,
        disconnected_scores=disconnected_scores,
        transposed_scores=transposed_scores,
        permutation_scores=permutation_scores,
        paired_advantage=paired_advantage,
    )
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("develop", "seal", "confirm"))
    parser.add_argument("config", type=Path)
    parser.add_argument("--development-output", type=Path)
    args = parser.parse_args()
    config = load_config(args.config)
    if args.mode == "develop":
        output = run_development(config)
    elif args.mode == "seal":
        if args.development_output is None:
            parser.error("--development-output is required for seal")
        output = seal_confirmation(config, args.development_output)
    else:
        output = run_confirmation(config)
    print(json.dumps({"output": str(Path(output).resolve())}))


if __name__ == "__main__":
    main()
