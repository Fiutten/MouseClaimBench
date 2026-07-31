"""Develop, seal, and externally confirm one temporal Allen VBN target."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from mousebrainbench import __version__
from mousebrainbench.artifacts import code_revision
from mousebrainbench.benchmarks.allen_vbn_phase2b import (
    _cluster_bootstrap_median_interval,
    _run_id,
)
from mousebrainbench.data.loaders.allen_vbn import AllenVBNRepository


def load_config(path: str | Path) -> dict[str, Any]:
    source = Path(path).resolve()
    config = yaml.safe_load(source.read_text())
    config["_config_path"] = str(source)
    return config


def transform_timecourse(activity: np.ndarray, time: np.ndarray, name: str) -> np.ndarray:
    """Apply a frozen transformation while preserving time-by-region layout."""

    baseline = activity[time < 0].mean(axis=0, keepdims=True)
    corrected = activity - baseline
    if name == "baseline_subtracted":
        return corrected
    if name == "region_peak_normalized":
        scale = np.max(np.abs(corrected), axis=0, keepdims=True)
        return np.divide(corrected, scale, out=np.zeros_like(corrected), where=scale > 0)
    if name == "temporal_derivative":
        return np.diff(corrected, axis=0, prepend=corrected[:1])
    raise ValueError(f"unknown temporal target transformation: {name}")


def _correlation(left: np.ndarray, right: np.ndarray) -> float:
    left_flat, right_flat = left.ravel(), right.ravel()
    if np.std(left_flat) == 0 or np.std(right_flat) == 0:
        return 0.0
    return float(np.corrcoef(left_flat, right_flat)[0, 1])


def evaluate_target(
    full: np.ndarray,
    odd: np.ndarray,
    even: np.ndarray,
    mouse_ids: np.ndarray,
    *,
    permutations: int,
    bootstrap_samples: int,
    seed: int,
    thresholds: dict[str, float],
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    rng = np.random.default_rng(seed)
    real = np.empty(len(full))
    split = np.empty(len(full))
    null = np.empty((len(full), permutations))
    for index in range(len(full)):
        reference = full[mouse_ids != mouse_ids[index]].mean(axis=0)
        real[index] = _correlation(full[index], reference)
        split[index] = _correlation(odd[index], even[index])
        for permutation in range(permutations):
            order = rng.permutation(full.shape[2])
            null[index, permutation] = _correlation(full[index], reference[:, order])
    interval = _cluster_bootstrap_median_interval(
        real, mouse_ids, samples=bootstrap_samples, rng=rng
    )
    fraction = float(np.mean(real > np.quantile(null, 0.95, axis=1)))
    metrics = {
        "median_cross_mouse_correlation": float(np.median(real)),
        "median_split_half_correlation": float(np.median(split)),
        "fraction_above_own_null_95th": fraction,
        "cluster_bootstrap_95_interval_for_median": interval,
        "median_cross_mouse_passed": float(np.median(real))
        > thresholds["median_cross_mouse_correlation_gt"],
        "median_split_half_passed": float(np.median(split))
        > thresholds["median_split_half_correlation_gt"],
        "fraction_above_null_passed": fraction >= thresholds["fraction_above_null_95_gte"],
        "bootstrap_lower_bound_passed": interval[0] > thresholds["bootstrap_lower_bound_gt"],
    }
    metrics["passed"] = all(value for key, value in metrics.items() if key.endswith("_passed"))
    return metrics, {"real": real, "split": split, "null": null}


def _extract(
    repository: AllenVBNRepository,
    session_ids: list[int],
    config: dict[str, Any],
) -> tuple[list[Any], list[dict[str, Any]]]:
    data, target = config["data"], config["target"]
    timecourses, failures = [], []
    for session_id in session_ids:
        try:
            timecourses.append(
                repository.extract_change_response_timecourse(
                    session_id,
                    tuple(data["regions"]),
                    min_units_per_region=int(data["min_units_per_region"]),
                    start_seconds=float(target["start_seconds"]),
                    stop_seconds=float(target["stop_seconds"]),
                    bin_size_seconds=float(target["bin_size_seconds"]),
                )
            )
        except (OSError, KeyError, ValueError) as error:
            failures.append({"session_id": session_id, "error": str(error)})
    return timecourses, failures


def _evaluate_candidates(
    timecourses: list[Any],
    mouse_ids: np.ndarray,
    config: dict[str, Any],
    candidates: list[str],
) -> tuple[dict[str, Any], dict[str, dict[str, np.ndarray]]]:
    results, arrays = {}, {}
    for offset, candidate in enumerate(candidates):
        full = np.stack(
            [transform_timecourse(item.activity_hz, item.time, candidate) for item in timecourses]
        )
        odd = np.stack(
            [transform_timecourse(item.odd_event_activity_hz, item.time, candidate) for item in timecourses]
        )
        even = np.stack(
            [transform_timecourse(item.even_event_activity_hz, item.time, candidate) for item in timecourses]
        )
        results[candidate], arrays[candidate] = evaluate_target(
            full,
            odd,
            even,
            mouse_ids,
            permutations=int(config["analysis"]["permutations"]),
            bootstrap_samples=int(config["analysis"]["bootstrap_samples"]),
            seed=int(config["analysis"]["seed"]) + offset,
            thresholds=config["analysis"]["thresholds"],
        )
    return results, arrays


def run_development(config: dict[str, Any]) -> Path:
    repository = AllenVBNRepository(config["data"]["root"])
    decisions = repository.qualify_sessions(
        tuple(config["data"]["regions"]),
        min_units_per_region=int(config["data"]["min_units_per_region"]),
    )
    accepted = [decision for decision in decisions if decision.accepted]
    timecourses, failures = _extract(repository, [item.session_id for item in accepted], config)
    decision_by_session = {item.session_id: item for item in accepted}
    mouse_ids = np.asarray([decision_by_session[item.session_id].mouse_id for item in timecourses])
    results, arrays = _evaluate_candidates(
        timecourses, mouse_ids, config, list(config["target"]["candidates"])
    )
    passed = [name for name, result in results.items() if result["passed"]]
    selected = (
        max(
            passed,
            key=lambda name: (
                results[name]["fraction_above_own_null_95th"],
                results[name]["median_cross_mouse_correlation"],
            ),
        )
        if passed
        else None
    )
    metrics = {
        "interpretation": "phase2c_development_only",
        "sessions": len(timecourses),
        "unique_mice": len(set(mouse_ids)),
        "candidate_results": results,
        "selected_target": selected,
        "decision": "target_selected" if selected else "no_target_selected",
        "failures": failures,
    }
    output = Path(config["output"]["root"]) / f"{config['output']['development_name']}-{_run_id(config)}"
    output.mkdir(parents=True, exist_ok=True)
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2))
    (output / "config.yaml").write_text(yaml.safe_dump({k: v for k, v in config.items() if k != "_config_path"}))
    (output / "provenance.json").write_text(
        json.dumps({"version": __version__, "git_revision": code_revision()}, indent=2)
    )
    packed = {"session_ids": np.asarray([item.session_id for item in timecourses]), "mouse_ids": mouse_ids}
    for name, values in arrays.items():
        for metric, array in values.items():
            packed[f"{name}_{metric}"] = array
    np.savez_compressed(output / "development_arrays.npz", **packed)
    return output


def seal_confirmation(config: dict[str, Any], development_output: str | Path) -> Path:
    metrics = json.loads((Path(development_output) / "metrics.json").read_text())
    selected = metrics["selected_target"]
    if selected is None:
        raise RuntimeError("development selected no target; confirmation must not proceed")
    repository = AllenVBNRepository(config["data"]["root"])
    decisions = repository.qualify_sessions(
        tuple(config["data"]["regions"]),
        min_units_per_region=int(config["data"]["min_units_per_region"]),
        available_only=False,
    )
    local = set(repository.available_session_ids())
    development_mice = {item.mouse_id for item in decisions if item.session_id in local}
    manifest_records = {
        int(record["url"].split("/")[-2]): record
        for record in repository.manifest["data_files"].values()
        if "/ecephys_session_" in record["url"] and "/probe_" not in record["url"]
    }
    candidates = [
        item
        for item in decisions
        if item.accepted
        and item.session_id not in local
        and item.mouse_id not in development_mice
        and item.session_id in manifest_records
    ]
    selected_sessions = []
    used_mice = set()
    for level, count in (
        ("Familiar", int(config["confirmation"]["familiar_sessions"])),
        ("Novel", int(config["confirmation"]["novel_sessions"])),
    ):
        eligible = [
            item
            for item in candidates
            if item.experience_level == level and item.mouse_id not in used_mice
        ]
        for item in sorted(eligible, key=lambda value: value.session_id)[:count]:
            record = manifest_records[item.session_id]
            selected_sessions.append(
                {
                    "session_id": item.session_id,
                    "mouse_id": item.mouse_id,
                    "experience_level": item.experience_level,
                    "url": record["url"],
                    "file_hash_blake2b": record["file_hash"],
                }
            )
            used_mice.add(item.mouse_id)
    if len(selected_sessions) != int(config["confirmation"]["sessions"]):
        raise RuntimeError("could not seal requested balanced confirmation cohort")
    plan = {
        "sealed_before_download": True,
        "selected_target": selected,
        "thresholds": config["analysis"]["thresholds"],
        "target": config["target"],
        "development_output": str(Path(development_output).resolve()),
        "sessions": selected_sessions,
    }
    destination = Path(config["confirmation"]["sealed_plan"])
    destination.write_text(json.dumps(plan, indent=2))
    return destination


def _validate_confirmation_plan(plan: dict[str, Any], config: dict[str, Any]) -> None:
    """Reject configuration drift before opening the sealed confirmation cohort."""

    if not plan.get("sealed_before_download"):
        raise ValueError("confirmation plan is not marked as sealed before download")
    if plan["target"] != config["target"]:
        raise ValueError("configured target differs from the sealed confirmation target")
    if plan["thresholds"] != config["analysis"]["thresholds"]:
        raise ValueError("configured thresholds differ from the sealed confirmation thresholds")
    if plan["selected_target"] not in plan["target"]["candidates"]:
        raise ValueError("selected target is absent from the sealed candidate set")


def run_confirmation(config: dict[str, Any]) -> Path:
    plan = json.loads(Path(config["confirmation"]["sealed_plan"]).read_text())
    _validate_confirmation_plan(plan, config)
    repository = AllenVBNRepository(config["data"]["root"])
    session_ids = [int(item["session_id"]) for item in plan["sessions"]]
    timecourses, failures = _extract(repository, session_ids, config)
    mouse_by_session = {int(item["session_id"]): int(item["mouse_id"]) for item in plan["sessions"]}
    experience_by_session = {
        int(item["session_id"]): str(item["experience_level"]) for item in plan["sessions"]
    }
    successful_session_ids = [item.session_id for item in timecourses]
    mouse_ids = np.asarray([mouse_by_session[item.session_id] for item in timecourses])
    experience_levels = np.asarray(
        [experience_by_session[item.session_id] for item in timecourses]
    )
    results, arrays = _evaluate_candidates(timecourses, mouse_ids, config, [plan["selected_target"]])
    result = results[plan["selected_target"]]
    sufficient = len(timecourses) >= int(config["confirmation"]["minimum_successful_sessions"])
    metrics = {
        "interpretation": "single_external_confirmation",
        "selected_target": plan["selected_target"],
        "sealed_sessions": len(session_ids),
        "successful_sessions": len(timecourses),
        "successful_session_ids": successful_session_ids,
        "failures": failures,
        "target_result": result,
        "minimum_successful_sessions_passed": sufficient,
        "confirmation_passed": sufficient and result["passed"],
        "decision": "target_confirmed" if sufficient and result["passed"] else "target_rejected",
    }
    output = Path(config["output"]["root"]) / f"{config['output']['confirmation_name']}-{_run_id(config)}"
    output.mkdir(parents=True, exist_ok=True)
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2))
    (output / "sealed_plan.json").write_text(json.dumps(plan, indent=2))
    (output / "provenance.json").write_text(
        json.dumps({"version": __version__, "git_revision": code_revision()}, indent=2)
    )
    np.savez_compressed(
        output / "confirmation_arrays.npz",
        session_ids=np.asarray(successful_session_ids),
        mouse_ids=mouse_ids,
        experience_levels=experience_levels,
        **arrays[plan["selected_target"]],
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
