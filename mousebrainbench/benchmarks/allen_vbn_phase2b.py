"""Decompose Allen VBN variability and qualify alternative regional targets."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from scipy.stats import spearmanr

from mousebrainbench import __version__
from mousebrainbench.artifacts import code_revision
from mousebrainbench.data.loaders.allen_vbn import AllenVBNRepository, qualification_records
from mousebrainbench.validation.functional_metrics import fc_correlation, functional_connectivity


def load_config(path: str | Path) -> dict[str, Any]:
    source = Path(path).resolve()
    config = yaml.safe_load(source.read_text())
    if not isinstance(config, dict):
        raise ValueError("Phase 2b config must be a YAML mapping")
    config["_config_path"] = str(source)
    return config


def _clean_config(config: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in config.items() if key != "_config_path"}


def _run_id(config: dict[str, Any]) -> str:
    return hashlib.sha256(yaml.safe_dump(_clean_config(config), sort_keys=True).encode()).hexdigest()[
        :12
    ]


def _correlation(left: np.ndarray, right: np.ndarray) -> float:
    if left.ndim == 2:
        return fc_correlation(left, right)
    if np.std(left) == 0 or np.std(right) == 0:
        return 0.0
    return float(np.corrcoef(left, right)[0, 1])


def _spearman(left: np.ndarray, right: np.ndarray) -> float:
    if np.std(left) == 0 or np.std(right) == 0:
        return 0.0
    return float(spearmanr(left, right).statistic)


def _permute_labels(value: np.ndarray, order: np.ndarray) -> np.ndarray:
    return value[np.ix_(order, order)] if value.ndim == 2 else value[order]


def _reliability(
    values: np.ndarray,
    split_left: np.ndarray,
    split_right: np.ndarray,
    mouse_ids: np.ndarray,
    *,
    permutations: int,
    rng: np.random.Generator,
    thresholds: dict[str, float],
) -> tuple[dict[str, Any], np.ndarray, np.ndarray, np.ndarray]:
    real = np.empty(len(values), dtype=float)
    split = np.empty(len(values), dtype=float)
    null = np.empty((len(values), permutations), dtype=float)
    for index, value in enumerate(values):
        reference = values[mouse_ids != mouse_ids[index]].mean(axis=0)
        real[index] = _correlation(value, reference)
        split[index] = _correlation(split_left[index], split_right[index])
        for permutation in range(permutations):
            order = rng.permutation(value.shape[-1])
            null[index, permutation] = _correlation(value, _permute_labels(reference, order))
    fraction_above_null = float(np.mean(real > np.quantile(null, 0.95, axis=1)))
    checks = {
        "median_cross_mouse_correlation": float(np.median(real)),
        "median_split_half_correlation": float(np.median(split)),
        "fraction_above_own_null_95th": fraction_above_null,
        "median_cross_mouse_passed": float(np.median(real))
        > thresholds["median_cross_mouse_correlation_gt"],
        "median_split_half_passed": float(np.median(split))
        > thresholds["median_split_half_correlation_gt"],
        "fraction_above_null_passed": fraction_above_null
        >= thresholds["fraction_above_null_95_gte"],
    }
    checks["passed"] = all(
        checks[key]
        for key in (
            "median_cross_mouse_passed",
            "median_split_half_passed",
            "fraction_above_null_passed",
        )
    )
    return checks, real, split, null


def _fc_components(activities: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    full = np.stack([functional_connectivity(activity) for activity in activities])
    left = np.stack([functional_connectivity(activity[: len(activity) // 2]) for activity in activities])
    right = np.stack(
        [functional_connectivity(activity[len(activity) // 2 :]) for activity in activities]
    )
    return full, left, right


def _rate_components(activities: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    full = np.stack([activity.mean(axis=0) for activity in activities])
    left = np.stack([activity[: len(activity) // 2].mean(axis=0) for activity in activities])
    right = np.stack([activity[len(activity) // 2 :].mean(axis=0) for activity in activities])
    return full, left, right


def _cross_mouse_correlations(values: np.ndarray, mouse_ids: np.ndarray) -> np.ndarray:
    return np.asarray(
        [
            _correlation(value, values[mouse_ids != mouse_ids[index]].mean(axis=0))
            for index, value in enumerate(values)
        ]
    )


def _cluster_bootstrap_median_interval(
    values: np.ndarray,
    mouse_ids: np.ndarray,
    *,
    samples: int,
    rng: np.random.Generator,
) -> list[float]:
    """Bootstrap mice as clusters and return a percentile interval for the median."""

    unique_mice = np.asarray(sorted(set(mouse_ids)))
    medians = np.empty(samples, dtype=float)
    for index in range(samples):
        sampled_mice = rng.choice(unique_mice, size=len(unique_mice), replace=True)
        sampled_values = np.concatenate([values[mouse_ids == mouse] for mouse in sampled_mice])
        medians[index] = np.median(sampled_values)
    return [float(value) for value in np.quantile(medians, [0.025, 0.975])]


def _experience_permutation(
    values: np.ndarray,
    mouse_ids: np.ndarray,
    experience: np.ndarray,
    *,
    permutations: int,
    rng: np.random.Generator,
) -> dict[str, Any]:
    """Exploratory mouse-level label permutation excluding mixed-experience mice."""

    records = []
    for mouse in sorted(set(mouse_ids)):
        labels = set(experience[mouse_ids == mouse])
        if len(labels) == 1:
            records.append((mouse, labels.pop(), float(np.median(values[mouse_ids == mouse]))))
    labels = np.asarray([record[1] for record in records])
    mouse_values = np.asarray([record[2] for record in records])
    levels = sorted(set(labels))
    if len(levels) != 2:
        return {"reason": "requires exactly two experience levels"}
    observed = float(np.median(mouse_values[labels == levels[0]]) - np.median(mouse_values[labels == levels[1]]))
    null = np.empty(permutations, dtype=float)
    for index in range(permutations):
        shuffled = rng.permutation(labels)
        null[index] = np.median(mouse_values[shuffled == levels[0]]) - np.median(
            mouse_values[shuffled == levels[1]]
        )
    return {
        "levels": levels,
        "included_mice": len(records),
        "excluded_mixed_experience_mice": len(set(mouse_ids)) - len(records),
        "median_difference_first_minus_second": observed,
        "two_sided_permutation_p": float((np.count_nonzero(np.abs(null) >= abs(observed)) + 1) / (permutations + 1)),
    }


def run_phase2b(config: dict[str, Any]) -> Path:
    """Run the frozen Phase 2b protocol and persist all decisions."""

    data, analysis = config["data"], config["analysis"]
    regions = tuple(data["regions"])
    repository = AllenVBNRepository(data["root"])
    decisions = repository.qualify_sessions(
        regions, min_units_per_region=int(data["min_units_per_region"])
    )
    accepted = [decision for decision in decisions if decision.accepted]
    spontaneous = []
    evoked = []
    failures = []
    for decision in accepted:
        try:
            session_spontaneous = repository.extract_spontaneous_activity(
                decision.session_id,
                regions,
                bin_size_seconds=float(analysis["base_bin_seconds"]),
                min_units_per_region=int(data["min_units_per_region"]),
            )
            session_evoked = repository.extract_change_response_profile(
                decision.session_id,
                regions,
                min_units_per_region=int(data["min_units_per_region"]),
                window_seconds=float(analysis["evoked_window_seconds"]),
            )
            spontaneous.append(session_spontaneous)
            evoked.append(session_evoked)
        except (OSError, KeyError, ValueError) as error:
            failures.append({"session_id": decision.session_id, "error": str(error)})
    if len(spontaneous) < int(analysis["minimum_sessions"]):
        raise RuntimeError("insufficient successfully extracted sessions for Phase 2b")
    decision_by_session = {decision.session_id: decision for decision in decisions}
    mouse_ids = np.asarray([decision_by_session[item.session_id].mouse_id for item in spontaneous])
    experience = np.asarray(
        [decision_by_session[item.session_id].experience_level for item in spontaneous]
    )
    unit_counts = np.asarray([item.unit_counts for item in spontaneous])
    activities = [item.activity_hz for item in spontaneous]
    fc = _fc_components(activities)
    rate = _rate_components(activities)
    evoked_components = (
        np.stack([item.response_hz for item in evoked]),
        np.stack([item.odd_event_response_hz for item in evoked]),
        np.stack([item.even_event_response_hz for item in evoked]),
    )
    rng = np.random.default_rng(int(analysis["seed"]))
    candidate_results = {}
    candidate_arrays = {}
    for name, components in (
        ("spontaneous_fc", fc),
        ("spontaneous_rate_profile", rate),
        ("visual_change_response_profile", evoked_components),
    ):
        result, real, split, null = _reliability(
            *components,
            mouse_ids,
            permutations=int(analysis["permutations"]),
            rng=rng,
            thresholds=analysis["candidate_thresholds"],
        )
        candidate_results[name] = result
        result["cluster_bootstrap_95_interval_for_median_cross_mouse"] = (
            _cluster_bootstrap_median_interval(
                real, mouse_ids, samples=int(analysis["bootstrap_samples"]), rng=rng
            )
        )
        candidate_arrays[name] = (real, split, null)

    passed = [name for name, result in candidate_results.items() if result["passed"]]
    selected = (
        max(
            passed,
            key=lambda name: (
                candidate_results[name]["fraction_above_own_null_95th"],
                candidate_results[name]["median_cross_mouse_correlation"],
            ),
        )
        if passed
        else None
    )

    bin_sensitivity = {}
    for bin_size in analysis["sensitivity_bin_seconds"]:
        if float(bin_size) == float(analysis["base_bin_seconds"]):
            values = fc[0]
        else:
            values = np.stack(
                [
                    functional_connectivity(
                        repository.extract_spontaneous_activity(
                            item.session_id,
                            regions,
                            bin_size_seconds=float(bin_size),
                            min_units_per_region=int(data["min_units_per_region"]),
                        ).activity_hz
                    )
                    for item in spontaneous
                ]
            )
        correlations = _cross_mouse_correlations(values, mouse_ids)
        bin_sensitivity[str(bin_size)] = {
            "median_cross_mouse_fc_correlation": float(np.median(correlations))
        }

    window_sensitivity = {}
    for duration in [*analysis["sensitivity_window_seconds"], "full"]:
        cropped = (
            activities
            if duration == "full"
            else [activity[: int(float(duration) / analysis["base_bin_seconds"])] for activity in activities]
        )
        values = np.stack([functional_connectivity(activity) for activity in cropped])
        correlations = _cross_mouse_correlations(values, mouse_ids)
        window_sensitivity[str(duration)] = {
            "median_cross_mouse_fc_correlation": float(np.median(correlations))
        }

    fc_real = candidate_arrays["spontaneous_fc"][0]
    minimum_units = unit_counts.min(axis=1)
    mean_units = unit_counts.mean(axis=1)
    repeated_mouse_correlations = []
    for mouse_id in sorted(set(mouse_ids)):
        indices = np.flatnonzero(mouse_ids == mouse_id)
        if len(indices) == 2:
            repeated_mouse_correlations.append(
                {"mouse_id": int(mouse_id), "fc_correlation": _correlation(fc[0][indices[0]], fc[0][indices[1]])}
            )
    diagnostics = {
        "bin_size_sensitivity": bin_sensitivity,
        "window_duration_sensitivity": window_sensitivity,
        "unit_count_association": {
            "minimum_units_spearman_r": _spearman(minimum_units, fc_real),
            "mean_units_spearman_r": _spearman(mean_units, fc_real),
        },
        "by_experience": {
            level: {
                "n_sessions": int(np.sum(experience == level)),
                "median_fc_cross_mouse_correlation": float(np.median(fc_real[experience == level])),
                "median_evoked_cross_mouse_correlation": float(
                    np.median(candidate_arrays["visual_change_response_profile"][0][experience == level])
                ),
            }
            for level in sorted(set(experience))
        },
        "experience_permutation_for_fc": _experience_permutation(
            fc_real,
            mouse_ids,
            experience,
            permutations=int(analysis["permutations"]),
            rng=rng,
        ),
        "experience_permutation_for_evoked": _experience_permutation(
            candidate_arrays["visual_change_response_profile"][0],
            mouse_ids,
            experience,
            permutations=int(analysis["permutations"]),
            rng=rng,
        ),
        "repeated_mouse_session_fc_correlations": repeated_mouse_correlations,
    }
    metrics = {
        "interpretation": "phase2b_target_selection_and_exploratory_variance_decomposition",
        "dataset": {
            "manifest_version": repository.manifest["manifest_version"],
            "metadata_hashes_valid": repository.verify_metadata_hashes(),
            "sessions": len(spontaneous),
            "unique_mice": len(set(mouse_ids)),
            "regions": regions,
        },
        "candidate_thresholds": analysis["candidate_thresholds"],
        "candidate_results": candidate_results,
        "selected_candidate": selected,
        "decision": "candidate_selected" if selected else "no_candidate_passed",
        "exploratory_diagnostics_not_selection_criteria": diagnostics,
        "limitations": [
            "Targets are region-aggregated observations, not anatomical or causal connectivity.",
            "Exploratory diagnostics cannot override candidate gates.",
            "Only two mice have repeated sessions, so within-mouse decomposition is descriptive.",
        ],
    }
    output = config["output"]
    output_dir = Path(output["root"]) / f"{output['name']}-{_run_id(config)}"
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "config.yaml").write_text(yaml.safe_dump(_clean_config(config), sort_keys=False))
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    (output_dir / "qualification.json").write_text(
        json.dumps({"decisions": qualification_records(decisions), "failures": failures}, indent=2)
    )
    (output_dir / "provenance.json").write_text(
        json.dumps(
            {
                "mousebrainbench_version": __version__,
                "git_revision": code_revision(),
                "source_config": config.get("_config_path"),
                "manifest_path": str(repository.manifest_path),
            },
            indent=2,
        )
    )
    arrays = {
        "session_ids": np.asarray([item.session_id for item in spontaneous]),
        "mouse_ids": mouse_ids,
        "experience_levels": experience,
        "region_acronyms": np.asarray(regions),
        "unit_counts": unit_counts,
        "evoked_event_counts": np.asarray([item.event_count for item in evoked]),
    }
    for name, (real, split, null) in candidate_arrays.items():
        arrays[f"{name}_cross_mouse"] = real
        arrays[f"{name}_split_half"] = split
        arrays[f"{name}_null"] = null
    np.savez_compressed(output_dir / "phase2b_arrays.npz", **arrays)
    return output_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    args = parser.parse_args()
    output = run_phase2b(load_config(args.config))
    print(json.dumps({"output_dir": str(output.resolve())}))


if __name__ == "__main__":
    main()
