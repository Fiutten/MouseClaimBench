"""Qualify regional functional-connectivity targets from Allen VBN sessions."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from mousebrainbench import __version__
from mousebrainbench.artifacts import code_revision
from mousebrainbench.data.loaders.allen_vbn import (
    AllenVBNRepository,
    qualification_records,
)
from mousebrainbench.validation.functional_metrics import fc_correlation, functional_connectivity


def _clean_config(config: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in config.items() if key != "_config_path"}


def _run_id(config: dict[str, Any]) -> str:
    payload = yaml.safe_dump(_clean_config(config), sort_keys=True).encode()
    return hashlib.sha256(payload).hexdigest()[:12]


def load_benchmark_config(path: str | Path) -> dict[str, Any]:
    source = Path(path).resolve()
    config = yaml.safe_load(source.read_text())
    if not isinstance(config, dict):
        raise ValueError("benchmark config must be a YAML mapping")
    config["_config_path"] = str(source)
    return config


def _permutation_null(
    observed: np.ndarray,
    reference: np.ndarray,
    *,
    permutations: int,
    rng: np.random.Generator,
) -> np.ndarray:
    values = np.empty(permutations, dtype=float)
    for index in range(permutations):
        order = rng.permutation(len(reference))
        values[index] = fc_correlation(observed, reference[np.ix_(order, order)])
    return values


def _vector_correlation(left: np.ndarray, right: np.ndarray) -> float:
    """Return a finite Pearson correlation, treating constant vectors as undefined/zero."""

    if np.std(left) == 0 or np.std(right) == 0:
        return 0.0
    return float(np.corrcoef(left, right)[0, 1])


def run_benchmark(config: dict[str, Any]) -> Path:
    """Extract a fixed cohort and measure leave-one-session-out target reliability."""

    data = config["data"]
    analysis = config["analysis"]
    regions = tuple(data["regions"])
    repository = AllenVBNRepository(data["root"])
    decisions = repository.qualify_sessions(
        regions, min_units_per_region=int(data["min_units_per_region"])
    )
    accepted = [decision for decision in decisions if decision.accepted]
    activities = []
    extraction_failures = []
    for decision in accepted:
        try:
            activities.append(
                repository.extract_spontaneous_activity(
                    decision.session_id,
                    regions,
                    bin_size_seconds=float(data["bin_size_seconds"]),
                    min_units_per_region=int(data["min_units_per_region"]),
                )
            )
        except (OSError, KeyError, ValueError) as error:
            extraction_failures.append({"session_id": decision.session_id, "error": str(error)})
    if len(activities) < int(analysis["minimum_sessions"]):
        raise RuntimeError(
            f"only {len(activities)} sessions extracted; minimum is {analysis['minimum_sessions']}"
        )

    session_fcs = np.stack([functional_connectivity(item.activity_hz) for item in activities])
    rate_profiles = np.stack([item.activity_hz.mean(axis=0) for item in activities])
    real_fc_correlations = []
    rate_correlations = []
    null_correlations = []
    rng = np.random.default_rng(int(analysis["seed"]))
    session_rows = {item.session_id: item for item in decisions}
    mouse_ids = np.asarray([session_rows[item.session_id].mouse_id for item in activities])
    experience_levels = np.asarray(
        [session_rows[item.session_id].experience_level for item in activities]
    )
    for index in range(len(activities)):
        train_indices = mouse_ids != mouse_ids[index]
        fc_reference = session_fcs[train_indices].mean(axis=0)
        rate_reference = rate_profiles[train_indices].mean(axis=0)
        real_fc_correlations.append(fc_correlation(session_fcs[index], fc_reference))
        rate_correlations.append(_vector_correlation(rate_profiles[index], rate_reference))
        null_correlations.append(
            _permutation_null(
                session_fcs[index],
                fc_reference,
                permutations=int(analysis["permutations"]),
                rng=rng,
            )
        )
    real = np.asarray(real_fc_correlations)
    null = np.stack(null_correlations)
    split_half = np.asarray(
        [
            fc_correlation(
                functional_connectivity(item.activity_hz[: len(item.time) // 2]),
                functional_connectivity(item.activity_hz[len(item.time) // 2 :]),
            )
            for item in activities
        ]
    )
    by_experience = {
        level: {
            "n_sessions": int(np.sum(experience_levels == level)),
            "median_leave_one_mouse_out_fc_correlation": float(
                np.median(real[experience_levels == level])
            ),
            "median_split_half_fc_correlation": float(
                np.median(split_half[experience_levels == level])
            ),
        }
        for level in sorted(set(experience_levels))
    }
    metrics = {
        "interpretation": "empirical_target_qualification_not_model_validation",
        "dataset": {
            "name": "Allen Visual Behavior Neuropixels",
            "manifest_version": repository.manifest["manifest_version"],
            "metadata_hashes_valid": repository.verify_metadata_hashes(),
            "available_sessions": len(repository.available_session_ids()),
            "metadata_qualified_sessions": len(accepted),
            "successfully_extracted_sessions": len(activities),
            "unique_mice": len({session_rows[item.session_id].mouse_id for item in activities}),
            "regions": regions,
        },
        "quality_filter": activities[0].metadata["quality_filter"],
        "target_reliability": {
            "cross_validation_unit": "mouse",
            "median_leave_one_mouse_out_fc_correlation": float(np.median(real)),
            "mean_leave_one_mouse_out_fc_correlation": float(np.mean(real)),
            "median_leave_one_mouse_out_rate_profile_correlation": float(
                np.median(rate_correlations)
            ),
            "median_permutation_fc_correlation": float(np.median(null)),
            "fraction_sessions_above_own_null_95th_percentile": float(
                np.mean(real > np.quantile(null, 0.95, axis=1))
            ),
        },
        "exploratory_diagnostics_not_decision_criteria": {
            "median_within_session_split_half_fc_correlation": float(np.median(split_half)),
            "by_experience_level": by_experience,
        },
        "limitations": [
            "Functional connectivity is correlation, not anatomical connectivity.",
            "The target is spontaneous population firing rate aggregated by recorded units.",
            "Sessions differ in sampled neurons and probe trajectories.",
            "This benchmark qualifies a target; it does not validate a brain model.",
        ],
    }
    metrics["decision_gate"] = {
        "minimum_sessions": len(activities) >= int(analysis["minimum_sessions"]),
        "all_metadata_hashes_valid": all(repository.verify_metadata_hashes().values()),
        "positive_median_fc_correlation": float(np.median(real)) > 0,
        "at_least_half_sessions_above_null_95th": float(
            np.mean(real > np.quantile(null, 0.95, axis=1))
        )
        >= 0.5,
    }
    metrics["decision_gate"]["passed"] = all(metrics["decision_gate"].values())
    output = config["output"]
    output_dir = Path(output["root"]) / f"{output['name']}-{_run_id(config)}"
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "config.yaml").write_text(yaml.safe_dump(_clean_config(config), sort_keys=False))
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    (output_dir / "session_qualification.json").write_text(
        json.dumps(
            {
                "decisions": qualification_records(decisions),
                "extraction_failures": extraction_failures,
            },
            indent=2,
        )
    )
    (output_dir / "provenance.json").write_text(
        json.dumps(
            {
                "mousebrainbench_version": __version__,
                "git_revision": code_revision(),
                "source_config": config.get("_config_path"),
                "manifest_path": str(repository.manifest_path),
                "metadata_hashes_valid": repository.verify_metadata_hashes(),
            },
            indent=2,
        )
    )
    session_report = [
        {
            "session_id": item.session_id,
            "mouse_id": int(mouse_ids[index]),
            "experience_level": str(experience_levels[index]),
            "leave_one_mouse_out_fc_correlation": float(real[index]),
            "null_95th_percentile": float(np.quantile(null[index], 0.95)),
            "above_null_95th": bool(real[index] > np.quantile(null[index], 0.95)),
            "split_half_fc_correlation": float(split_half[index]),
            "unit_counts": dict(zip(regions, item.unit_counts)),
        }
        for index, item in enumerate(activities)
    ]
    (output_dir / "session_metrics.json").write_text(json.dumps(session_report, indent=2))
    np.savez_compressed(
        output_dir / "benchmark_arrays.npz",
        session_ids=np.asarray([item.session_id for item in activities]),
        region_acronyms=np.asarray(regions),
        session_fcs=session_fcs,
        rate_profiles=rate_profiles,
        mouse_ids=mouse_ids,
        experience_levels=experience_levels,
        leave_one_mouse_out_fc_correlations=real,
        leave_one_mouse_out_rate_correlations=np.asarray(rate_correlations),
        permutation_fc_correlations=null,
        split_half_fc_correlations=split_half,
        unit_counts=np.asarray([item.unit_counts for item in activities]),
    )
    return output_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    args = parser.parse_args()
    output = run_benchmark(load_benchmark_config(args.config))
    print(json.dumps({"output_dir": str(output.resolve())}))


if __name__ == "__main__":
    main()
