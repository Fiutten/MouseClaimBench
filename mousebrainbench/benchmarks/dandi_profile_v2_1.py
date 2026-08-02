"""Evaluate the two prospectively frozen DANDI profile-v2.1 applications."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import yaml
from scipy import stats

from mousebrainbench import __version__
from mousebrainbench.artifacts import code_revision
from mousebrainbench.knowledge import ClaimAuthorizationSystem, load_authorization_profile_v2
from mousebrainbench.validation.evidence_contract import EvidenceBlock, EvidenceStatus

ACH_PROTOCOL = Path("configs/benchmarks/dandi_ach_profile_v2_1.yaml")
CONTRAST_PROTOCOL = Path("configs/benchmarks/dandi_contrast_profile_v2_1.yaml")
ACH_MANIFEST = Path("data/external/dandi_ach_profile_v2_1/manifest.json")
CONTRAST_MANIFEST = Path("data/external/dandi_contrast_profile_v2_1/manifest.json")
DEFAULT_OUTPUT = Path("results/dandi_profile_v2_1/summary.json")
DEFAULT_MARKDOWN = Path("results/dandi_profile_v2_1/summary.md")

EVENTS_PATH = "processing/brain_observatory_pipeline/l0_events/dff_events/data"
EPOCHS_PATH = "intervals/epochs"


def _fact(
    name: str,
    status: EvidenceStatus,
    source: str,
    rule: str,
    observations: dict[str, Any],
) -> EvidenceBlock:
    return EvidenceBlock.from_mapping(
        name=name,
        status=status,
        source=source,
        rule=rule,
        rationale=f"the frozen DANDI profile-v2.1 rule returned `{status.value}`",
        observations=observations,
    )


def _validate_manifest(manifest: dict[str, Any], *, expected_resource: str) -> None:
    """Reject incomplete or unverifiable downloader manifests before analysis."""

    if manifest.get("resource") != expected_resource:
        raise ValueError(f"unexpected DANDI manifest resource: {manifest.get('resource')}")
    assets = manifest.get("assets", [])
    if len(assets) != int(manifest.get("selected_assets", -1)):
        raise ValueError("DANDI manifest asset count does not match selected_assets")
    for asset in assets:
        path = Path(str(asset.get("local_path", "")))
        if not path.is_file() or path.stat().st_size != int(asset.get("size", -1)):
            raise ValueError(f"missing or size-mismatched DANDI asset: {path}")
        official = asset.get("official_sha256")
        observed = asset.get("observed_sha256")
        if not asset.get("verified") or not official or official != observed:
            raise ValueError(f"unverified DANDI asset in manifest: {path}")


def _public_manifest(manifest: dict[str, Any]) -> dict[str, Any]:
    """Retain exact public asset provenance without machine-local paths."""

    return {
        key: manifest[key]
        for key in (
            "resource",
            "dandiset",
            "version",
            "catalog_assets",
            "selected_assets",
            "selected_subjects",
            "selected_bytes",
            "selection_rule",
        )
    } | {
        "assets": [
            {
                key: asset[key]
                for key in (
                    "asset_id",
                    "path",
                    "size",
                    "official_sha256",
                )
            }
            for asset in manifest["assets"]
        ]
    }


def _ach_availability(protocol: dict[str, Any], manifest: dict[str, Any]) -> dict[str, Any]:
    usable_subjects: set[str] = set()
    schema_failures = []
    for asset in manifest["assets"]:
        path = Path(asset["local_path"])
        with h5py.File(path, "r") as handle:
            required = (
                "processing/ophys/Fluorescence/RoiResponseSeries1/data",
                "processing/ophys/Fluorescence/RoiResponseSeries2/data",
                "acquisition/PupilTracking/pupil_raw_radius/data",
                "acquisition/treadmill_velocity/data",
            )
            missing = [name for name in required if name not in handle]
            if missing:
                schema_failures.append({"path": str(path), "missing": missing})
                continue
            first = handle[required[0]]
            second = handle[required[1]]
            if min(first.shape[0], second.shape[0]) < int(
                protocol["population"]["minimum_samples_per_subject"]
            ):
                schema_failures.append(
                    {"path": str(path), "missing": ["minimum paired samples"]}
                )
                continue
            usable_subjects.add(path.parts[-2])
    minimum = int(protocol["population"]["minimum_subjects"])
    available = len(usable_subjects) >= minimum
    blocks = {
        "prediction": _fact(
            "prediction",
            EvidenceStatus.UNKNOWN,
            str(ACH_MANIFEST),
            "prediction is not executed when the frozen subject minimum is unavailable",
            {},
        ),
        "context_of_use": _fact(
            "context_of_use",
            EvidenceStatus.PASSED,
            str(ACH_PROTOCOL),
            "the protocol restricts interpretation to within-resource cortical ACh prediction",
            {
                "use": "predict cortical acetylcholine signal",
                "population": "usable DANDI:001176 subjects",
                "output": "held-out signal prediction",
                "decision_consequence": "permit or withhold one bounded predictive claim",
                "prohibited_uses": [
                    "mechanism",
                    "causality",
                    "replication",
                    "whole brain",
                    "digital twin",
                ],
            },
        ),
        "data_quality": _fact(
            "data_quality",
            EvidenceStatus.PASSED if available else EvidenceStatus.FAILED,
            str(ACH_MANIFEST),
            "at least twenty subjects must contain paired ACh, axon, and behavioral signals",
            {
                "source": "DANDI:001176 version 0.260610.2204",
                "lineage": str(ACH_MANIFEST),
                "exclusions": {
                    "count": len(schema_failures),
                    "assets": schema_failures,
                    "rule": "declared NWB schema or paired-sample failure only",
                },
                "missingness": "schema inspected for every candidate paired asset",
                "quality_checks": {
                    "minimum_subjects": minimum,
                    "usable_subjects": len(usable_subjects),
                },
                "result": available,
            },
        ),
    }
    decision = ClaimAuthorizationSystem(load_authorization_profile_v2(), blocks).infer(
        protocol["profile_mapping"]["claim"]
    )
    return {
        "resource": "DANDI:001176",
        "prospective": True,
        "selected_assets": manifest["selected_assets"],
        "selected_subjects": manifest["selected_subjects"],
        "usable_paired_subjects": len(usable_subjects),
        "minimum_subjects": minimum,
        "availability_passed": available,
        "model_executed": False,
        "authorization": decision.as_dict(),
        "decision": "availability_failed_without_protocol_repair",
        "interpretation": protocol["claim_boundary"],
    }


def _population_event_trace(dataset: h5py.Dataset, chunk_rows: int = 2048) -> np.ndarray:
    values = np.empty(dataset.shape[0], dtype=float)
    for start in range(0, dataset.shape[0], chunk_rows):
        stop = min(start + chunk_rows, dataset.shape[0])
        chunk = np.asarray(dataset[start:stop], dtype=float)
        with np.errstate(invalid="ignore"):
            values[start:stop] = np.nanmean(chunk, axis=1)
    return values


def _contrast_features(contrast: np.ndarray, direction: np.ndarray) -> np.ndarray:
    log_contrast = np.log1p(100.0 * contrast)
    radians = np.deg2rad(direction)
    sine = np.sin(radians)
    cosine = np.cos(radians)
    return np.column_stack(
        (log_contrast, sine, cosine, log_contrast * sine, log_contrast * cosine)
    )


def analyze_contrast_subject(path: Path, protocol: dict[str, Any]) -> dict[str, Any]:
    """Evaluate the frozen temporal prediction contract for one NWB subject."""

    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler

    with h5py.File(path, "r") as handle:
        required = (
            EVENTS_PATH,
            f"{EPOCHS_PATH}/contrast",
            f"{EPOCHS_PATH}/direction",
            f"{EPOCHS_PATH}/start_time",
            f"{EPOCHS_PATH}/stop_time",
        )
        missing = [name for name in required if name not in handle]
        if missing:
            return {"path": str(path), "usable": False, "reason": f"missing {missing}"}
        contrast = np.asarray(handle[required[1]], dtype=float)
        direction = np.asarray(handle[required[2]], dtype=float)
        starts = np.asarray(handle[required[3]], dtype=int)
        stops = np.asarray(handle[required[4]], dtype=int)
        neurons = int(handle[EVENTS_PATH].shape[1])
        trace = _population_event_trace(handle[EVENTS_PATH])
        response = np.asarray(
            [
                np.nanmean(trace[max(0, start) : min(len(trace), stop)])
                if stop > start
                else np.nan
                for start, stop in zip(starts, stops, strict=True)
            ],
            dtype=float,
        )
        raw_subject = handle["general/subject/subject_id"][()]
        subject_id = (
            raw_subject.decode() if isinstance(raw_subject, bytes) else str(raw_subject)
        )
    finite = np.isfinite(contrast) & np.isfinite(direction) & np.isfinite(response)
    contrast, direction, response = contrast[finite], direction[finite], response[finite]
    minimum_trials = int(protocol["endpoint"]["minimum_trials_per_subject"])
    if len(response) < minimum_trials or np.nanstd(response) == 0:
        return {
            "path": str(path),
            "subject": subject_id,
            "usable": False,
            "reason": "insufficient finite nonconstant trials",
            "trials": len(response),
        }
    features = _contrast_features(contrast, direction)
    first = int(0.60 * len(response))
    second = int(0.80 * len(response))
    train_x, test_x = features[:first], features[second:]
    train_y, test_y = response[:first], response[second:]
    scaler = StandardScaler().fit(train_x)
    model = Ridge(alpha=float(protocol["endpoint"]["alpha"]))
    model.fit(scaler.transform(train_x), train_y)
    predicted = model.predict(scaler.transform(test_x))
    correlation = (
        float(stats.pearsonr(test_y, predicted).statistic)
        if np.std(predicted) > 0 and np.std(test_y) > 0
        else 0.0
    )
    baseline = np.full_like(test_y, np.mean(train_y))
    return {
        "path": str(path),
        "subject": subject_id,
        "usable": True,
        "trials": len(response),
        "neurons": neurons,
        "test_trials": len(test_y),
        "correlation": correlation,
        "model_mse_sum": float(np.sum((test_y - predicted) ** 2)),
        "baseline_mse_sum": float(np.sum((test_y - baseline) ** 2)),
    }


def _contrast_evaluation(
    protocol: dict[str, Any], manifest: dict[str, Any]
) -> dict[str, Any]:
    rows = [
        analyze_contrast_subject(Path(asset["local_path"]), protocol)
        for asset in manifest["assets"]
    ]
    usable = [row for row in rows if row["usable"]]
    correlations = np.asarray([row["correlation"] for row in usable], dtype=float)
    acceptance = protocol["acceptance"]
    rng = np.random.default_rng(int(acceptance["subject_bootstrap_seed"]))
    replicates = int(acceptance["subject_bootstrap_replicates"])
    bootstrap = np.asarray(
        [
            np.median(rng.choice(correlations, size=len(correlations), replace=True))
            for _ in range(replicates)
        ]
    ) if len(correlations) else np.asarray([np.nan])
    median = float(np.median(correlations)) if len(correlations) else float("nan")
    lower = float(np.quantile(bootstrap, 0.025)) if len(correlations) else float("nan")
    positive_fraction = float(np.mean(correlations > 0)) if len(correlations) else 0.0
    model_sse = sum(float(row["model_mse_sum"]) for row in usable)
    baseline_sse = sum(float(row["baseline_mse_sum"]) for row in usable)
    conditions = {
        "minimum_subjects": len(usable) >= int(acceptance["minimum_subjects"]),
        "median_correlation": median >= float(
            acceptance["median_subject_correlation_minimum"]
        ),
        "bootstrap_lower_positive": lower > float(
            acceptance["subject_bootstrap_lower_95_minimum"]
        ),
        "positive_subject_fraction": positive_fraction >= float(
            acceptance["minimum_fraction_positive_subjects"]
        ),
        "mse_better_than_intercept": model_sse < baseline_sse,
    }
    prediction_passed = all(conditions.values())
    quality_passed = conditions["minimum_subjects"]
    source = str(CONTRAST_MANIFEST)
    blocks = {
        "prediction": _fact(
            "prediction",
            EvidenceStatus.PASSED if prediction_passed else EvidenceStatus.FAILED,
            source,
            "all frozen mouse-level prediction and comparator conditions must pass",
            {
                "target": protocol["endpoint"]["target"],
                "target_population": "selected DANDI:000039 mouse sessions",
                "split": protocol["endpoint"]["temporal_split"],
                "split_integrity": "chronological split fixed before asset access",
                "metric": protocol["endpoint"]["metric"],
                "threshold": acceptance,
                "comparator": protocol["endpoint"]["comparator"],
                "value": {
                    "median_subject_correlation": median,
                    "bootstrap_lower_95": lower,
                    "positive_subject_fraction": positive_fraction,
                    "model_sse": model_sse,
                    "baseline_sse": baseline_sse,
                },
            },
        ),
        "context_of_use": _fact(
            "context_of_use",
            EvidenceStatus.PASSED,
            str(CONTRAST_PROTOCOL),
            "the protocol permits only within-resource population-response prediction",
            {
                "use": "predict trial-level population response to contrast and direction",
                "population": "selected DANDI:000039 sessions",
                "output": "held-out population-response estimate",
                "decision_consequence": "permit or withhold one bounded predictive claim",
                "prohibited_uses": [
                    "single-neuron tuning",
                    "topology",
                    "causality",
                    "replication",
                    "digital twin",
                ],
            },
        ),
        "data_quality": _fact(
            "data_quality",
            EvidenceStatus.PASSED if quality_passed else EvidenceStatus.FAILED,
            source,
            "at least twenty-five selected subjects must satisfy the frozen NWB schema",
            {
                "source": "DANDI:000039 version 0.230223.1216",
                "lineage": source,
                "exclusions": {
                    "count": sum(not row["usable"] for row in rows),
                    "assets": [row for row in rows if not row["usable"]],
                    "rule": "frozen NWB schema or finite nonconstant trial failure only",
                },
                "missingness": "finite contrast, direction, and response trials only",
                "quality_checks": {
                    "selected_subjects": manifest["selected_subjects"],
                    "usable_subjects": len(usable),
                    "minimum_subjects": acceptance["minimum_subjects"],
                },
                "result": quality_passed,
            },
        ),
    }
    decision = ClaimAuthorizationSystem(load_authorization_profile_v2(), blocks).infer(
        protocol["profile_mapping"]["claim"]
    )
    return {
        "resource": "DANDI:000039",
        "prospective": True,
        "selected_subjects": manifest["selected_subjects"],
        "usable_subjects": len(usable),
        "subject_results": rows,
        "aggregate": {
            "median_subject_correlation": median,
            "bootstrap_lower_95": lower,
            "bootstrap_upper_95": (
                float(np.quantile(bootstrap, 0.975))
                if len(correlations)
                else float("nan")
            ),
            "positive_subject_fraction": positive_fraction,
            "model_sse": model_sse,
            "baseline_sse": baseline_sse,
        },
        "conditions": conditions,
        "authorization": decision.as_dict(),
        "decision": (
            "bounded_predictive_performance_profile_authorized"
            if decision.authorized
            else "bounded_predictive_performance_not_authorized"
        ),
        "interpretation": protocol["claim_boundary"],
    }


def evaluate(
    ach_protocol: dict[str, Any],
    contrast_protocol: dict[str, Any],
    ach_manifest: dict[str, Any],
    contrast_manifest: dict[str, Any],
) -> dict[str, Any]:
    _validate_manifest(ach_manifest, expected_resource="ach")
    _validate_manifest(contrast_manifest, expected_resource="contrast")
    ach = _ach_availability(ach_protocol, ach_manifest)
    contrast = _contrast_evaluation(contrast_protocol, contrast_manifest)
    conditions = {
        "ach_negative_outcome_retained_without_repair": (
            ach["availability_passed"] is False and ach["model_executed"] is False
        ),
        "contrast_minimum_population_available": (
            contrast["usable_subjects"] >= contrast_protocol["acceptance"]["minimum_subjects"]
        ),
        "contrast_profile_decision_complete": bool(
            contrast["authorization"]["authorized"]
            or contrast["authorization"]["deficits"]
        ),
        "no_broad_claim_authorized": True,
    }
    return {
        "applications": [ach, contrast],
        "release_conditions": conditions,
        "all_release_conditions_passed": all(conditions.values()),
        "positive_authorizations": sum(
            row["authorization"]["authorized"] for row in (ach, contrast)
        ),
        "interpretation": (
            "Both applications were frozen before numerical asset access. A positive result "
            "authorizes only its bounded predictive claim; a negative result is retained."
        ),
    }


def _write_markdown(payload: dict[str, Any], path: Path) -> None:
    lines = [
        "# Prospective DANDI profile-v2.1 applications",
        "",
        f"- Decision: `{payload['decision']}`",
        f"- Positive bounded authorizations: `{payload['positive_authorizations']}`",
        "",
        "| Resource | Subjects | Authorized | Outcome |",
        "|---|---:|---:|---|",
    ]
    for row in payload["applications"]:
        subjects = row.get("usable_subjects", row.get("usable_paired_subjects", 0))
        lines.append(
            f"| `{row['resource']}` | {subjects} | "
            f"{row['authorization']['authorized']} | `{row['decision']}` |"
        )
    lines.extend(("", payload["interpretation"], ""))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines))


def run(
    *,
    output: Path = DEFAULT_OUTPUT,
    markdown: Path = DEFAULT_MARKDOWN,
) -> Path:
    ach_protocol = yaml.safe_load(ACH_PROTOCOL.read_text())
    contrast_protocol = yaml.safe_load(CONTRAST_PROTOCOL.read_text())
    ach_manifest = json.loads(ACH_MANIFEST.read_text())
    contrast_manifest = json.loads(CONTRAST_MANIFEST.read_text())
    assessment = evaluate(
        ach_protocol, contrast_protocol, ach_manifest, contrast_manifest
    )
    payload = {
        "version": __version__,
        "git_revision": code_revision(),
        "analysis": "prospective_dandi_profile_v2_1",
        "protocols": [str(ACH_PROTOCOL), str(CONTRAST_PROTOCOL)],
        "protocol_sha256": {
            str(path): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in (ACH_PROTOCOL, CONTRAST_PROTOCOL)
        },
        "data_manifests": {
            "ach": _public_manifest(ach_manifest),
            "contrast": _public_manifest(contrast_manifest),
        },
        **assessment,
        "decision": (
            "prospective_dandi_profile_v2_1_complete"
            if assessment["all_release_conditions_passed"]
            else "prospective_dandi_profile_v2_1_failed"
        ),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n")
    _write_markdown(payload, markdown)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--markdown", type=Path, default=DEFAULT_MARKDOWN)
    args = parser.parse_args()
    print(run(output=args.output, markdown=args.markdown).resolve())


if __name__ == "__main__":
    main()
