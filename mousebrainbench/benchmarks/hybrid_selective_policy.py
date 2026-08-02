"""Fit and freeze the leakage-controlled hybrid selective policy.

The learned component estimates claim support from already-consumed synthetic
development cases. It cannot override the explicit semantic support vetoes.
Model fitting, isotonic calibration, threshold selection, and the locked
development audit use disjoint deterministic partitions.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import numpy as np
import yaml
from scipy import stats

from mousebrainbench import __version__
from mousebrainbench.artifacts import code_revision
from mousebrainbench.benchmarks.hybrid_development_features import hybrid_feature_names
from mousebrainbench.knowledge import load_default_profile


DEFAULT_PROTOCOL = Path("configs/benchmarks/hybrid_selective_claim_validation_v2.yaml")
DEFAULT_MATRIX = Path("results/hybrid_development_features/cases.npz")
DEFAULT_MANIFEST = Path("results/hybrid_development_features/summary.json")
DEFAULT_OUTPUT = Path("results/hybrid_selective_policy/model.json")


def _sha256(path: Path) -> str:
    return f"sha256:{hashlib.sha256(path.read_bytes()).hexdigest()}"


def _load_protocol(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text())
    if payload.get("status") != "frozen_before_implementation":
        raise ValueError("hybrid validation protocol is not implementation-frozen")
    return payload


def clopper_pearson_upper(errors: int, decisions: int, confidence: float = 0.95) -> float:
    """Return the exact one-sided binomial upper confidence bound."""

    if decisions <= 0:
        if errors != 0:
            raise ValueError("zero decisions require zero errors")
        return 1.0
    if errors < 0 or errors > decisions:
        raise ValueError("errors must lie between zero and decisions")
    if errors == decisions:
        return 1.0
    return float(stats.beta.ppf(confidence, errors + 1, decisions - errors))


def _decision_metrics(decisions: np.ndarray, labels: np.ndarray) -> dict[str, Any]:
    decided = decisions != 0
    support = decisions == 1
    blocked = decisions == -1
    errors = (support & ~labels) | (blocked & labels)
    false_support = support & ~labels
    decision_count = int(decided.sum())
    error_count = int((errors & decided).sum())
    support_count = int(support.sum())
    return {
        "total": int(decisions.size),
        "decisions": decision_count,
        "abstentions": int((~decided).sum()),
        "supports": support_count,
        "blocks": int(blocked.sum()),
        "errors": error_count,
        "false_authorizations": int(false_support.sum()),
        "coverage": float(decision_count / decisions.size),
        "selective_error": float(error_count / decision_count) if decision_count else 1.0,
        "selective_error_cp95_upper": clopper_pearson_upper(error_count, decision_count),
        "false_authorization_fraction": (
            float(false_support.sum() / support_count) if support_count else 0.0
        ),
    }


def _status_indices(feature_names: Sequence[str]) -> dict[str, dict[str, int]]:
    indices: dict[str, dict[str, int]] = {}
    for index, name in enumerate(feature_names):
        if not name.startswith("status:"):
            continue
        _, block, status = name.split(":", maxsplit=2)
        indices.setdefault(block, {})[status] = index
    return indices


def _semantic_veto_state(
    features: np.ndarray,
    *,
    required_blocks: Sequence[str],
    status_indices: Mapping[str, Mapping[str, int]],
) -> str:
    """Return pass, fail, or unresolved for a semantic support candidate."""

    unresolved = False
    for block in required_blocks:
        indices = status_indices.get(block)
        if not indices:
            unresolved = True
            continue
        if features[indices["passed"]] > 0.5:
            continue
        if features[indices["failed"]] > 0.5:
            return "failed"
        unresolved = True
    return "unresolved" if unresolved else "passed"


def selective_decisions(
    probabilities: np.ndarray,
    features: np.ndarray,
    *,
    threshold: float,
    claim_names: Sequence[str],
    feature_names: Sequence[str],
    support_vetoes: Mapping[str, Sequence[str]],
    constrained: bool,
) -> tuple[np.ndarray, int]:
    """Map probabilities to support/block/abstain, then enforce support vetoes."""

    if probabilities.shape != (len(features), len(claim_names)):
        raise ValueError("probability matrix has incompatible shape")
    decisions = np.zeros(probabilities.shape, dtype=np.int8)
    decisions[probabilities >= threshold] = 1
    decisions[probabilities <= 1.0 - threshold] = -1
    if not constrained:
        return decisions, 0

    indices = _status_indices(feature_names)
    veto_violations = 0
    for row_index, row in enumerate(features):
        for claim_index, claim in enumerate(claim_names):
            if decisions[row_index, claim_index] != 1:
                continue
            required = support_vetoes.get(claim, ())
            state = _semantic_veto_state(
                row,
                required_blocks=required,
                status_indices=indices,
            )
            if state == "failed":
                decisions[row_index, claim_index] = -1
            elif state == "unresolved":
                decisions[row_index, claim_index] = 0

    # This postcondition makes the non-compensatory property executable.
    for row_index, row in enumerate(features):
        for claim_index, claim in enumerate(claim_names):
            if decisions[row_index, claim_index] != 1:
                continue
            if _semantic_veto_state(
                row,
                required_blocks=support_vetoes.get(claim, ()),
                status_indices=indices,
            ) != "passed":
                veto_violations += 1
    return decisions, veto_violations


def _fit_claim_model(
    features: np.ndarray,
    labels: np.ndarray,
    *,
    selected_indices: np.ndarray,
    regularization_c: float,
    max_iterations: int,
) -> dict[str, Any]:
    from sklearn.linear_model import LogisticRegression

    selected = features[:, selected_indices]
    mean = selected.mean(axis=0)
    scale = selected.std(axis=0)
    scale[scale < 1e-12] = 1.0
    transformed = (selected - mean) / scale
    if np.all(labels == labels[0]):
        return {
            "constant": True,
            "constant_probability": float(labels[0]),
            "selected_indices": selected_indices.tolist(),
            "scaler_mean": mean.tolist(),
            "scaler_scale": scale.tolist(),
            "intercept": 0.0,
            "coefficients": [0.0] * len(selected_indices),
        }
    estimator = LogisticRegression(
        C=regularization_c,
        l1_ratio=0.0,
        class_weight="balanced",
        solver="lbfgs",
        max_iter=max_iterations,
        random_state=0,
    )
    estimator.fit(transformed, labels)
    if not (
        np.all(np.isfinite(estimator.coef_))
        and np.all(np.isfinite(estimator.intercept_))
    ):
        raise RuntimeError("logistic fit produced non-finite parameters")
    return {
        "constant": False,
        "constant_probability": None,
        "selected_indices": selected_indices.tolist(),
        "scaler_mean": mean.tolist(),
        "scaler_scale": scale.tolist(),
        "intercept": float(estimator.intercept_[0]),
        "coefficients": estimator.coef_[0].tolist(),
    }


def _raw_probability(model: Mapping[str, Any], features: np.ndarray) -> np.ndarray:
    if model["constant"]:
        return np.full(len(features), float(model["constant_probability"]))
    selected = features[:, np.asarray(model["selected_indices"], dtype=int)]
    transformed = (
        selected - np.asarray(model["scaler_mean"], dtype=float)
    ) / np.asarray(model["scaler_scale"], dtype=float)
    score = transformed @ np.asarray(model["coefficients"], dtype=float) + float(
        model["intercept"]
    )
    if not np.all(np.isfinite(score)):
        raise RuntimeError("serialized logistic model produced non-finite scores")
    return 1.0 / (1.0 + np.exp(-np.clip(score, -40.0, 40.0)))


def _fit_isotonic(raw_probability: np.ndarray, labels: np.ndarray) -> dict[str, Any]:
    from sklearn.isotonic import IsotonicRegression

    if len(np.unique(raw_probability)) < 2 or len(np.unique(labels)) < 2:
        return {
            "kind": "constant",
            "value": float(labels.mean()),
            "x_thresholds": [],
            "y_thresholds": [],
        }
    calibrator = IsotonicRegression(out_of_bounds="clip")
    calibrator.fit(raw_probability, labels)
    return {
        "kind": "isotonic",
        "value": None,
        "x_thresholds": calibrator.X_thresholds_.tolist(),
        "y_thresholds": calibrator.y_thresholds_.tolist(),
    }


def _apply_calibration(calibration: Mapping[str, Any], raw: np.ndarray) -> np.ndarray:
    if calibration["kind"] == "constant":
        return np.full(len(raw), float(calibration["value"]))
    return np.interp(
        raw,
        np.asarray(calibration["x_thresholds"], dtype=float),
        np.asarray(calibration["y_thresholds"], dtype=float),
    )


def _fit_model_set(
    fit_features: np.ndarray,
    fit_labels: np.ndarray,
    calibration_features: np.ndarray,
    calibration_labels: np.ndarray,
    *,
    claim_names: Sequence[str],
    selected_indices: np.ndarray,
    regularization_c: float,
    max_iterations: int,
) -> dict[str, Any]:
    claims: dict[str, Any] = {}
    for claim_index, claim in enumerate(claim_names):
        model = _fit_claim_model(
            fit_features,
            fit_labels[:, claim_index],
            selected_indices=selected_indices,
            regularization_c=regularization_c,
            max_iterations=max_iterations,
        )
        raw = _raw_probability(model, calibration_features)
        model["calibration"] = _fit_isotonic(raw, calibration_labels[:, claim_index])
        claims[claim] = model
    return {"claims": claims}


def predict_probabilities(
    model_set: Mapping[str, Any],
    features: np.ndarray,
    claim_names: Sequence[str],
    *,
    calibrated: bool = True,
) -> np.ndarray:
    """Apply a serialized model set without depending on live estimators."""

    columns: list[np.ndarray] = []
    for claim in claim_names:
        model = model_set["claims"][claim]
        raw = _raw_probability(model, features)
        columns.append(_apply_calibration(model["calibration"], raw) if calibrated else raw)
    return np.column_stack(columns)


def _select_threshold(
    probabilities: np.ndarray,
    features: np.ndarray,
    labels: np.ndarray,
    *,
    claim_names: Sequence[str],
    feature_names: Sequence[str],
    support_vetoes: Mapping[str, Sequence[str]],
    settings: Mapping[str, Any],
    constrained: bool,
) -> tuple[float, list[dict[str, Any]]]:
    grid = np.round(
        np.arange(
            float(settings["threshold_grid_start"]),
            float(settings["threshold_grid_stop"]) + 1e-12,
            float(settings["threshold_grid_step"]),
        ),
        10,
    )
    target = float(
        settings["calibration_target"]["clopper_pearson_upper_bound_at_most"]
    )
    curve: list[dict[str, Any]] = []
    eligible: list[dict[str, Any]] = []
    for threshold in grid:
        decisions, violations = selective_decisions(
            probabilities,
            features,
            threshold=float(threshold),
            claim_names=claim_names,
            feature_names=feature_names,
            support_vetoes=support_vetoes,
            constrained=constrained,
        )
        metrics = _decision_metrics(decisions, labels.astype(bool))
        row = {
            "threshold": float(threshold),
            "semantic_support_veto_violations": violations,
            **metrics,
        }
        curve.append(row)
        if metrics["decisions"] and metrics["selective_error_cp95_upper"] <= target:
            eligible.append(row)
    if not eligible:
        raise RuntimeError("no selective threshold satisfies the frozen calibration target")
    selected = max(
        eligible,
        key=lambda row: (
            row["coverage"],
            -row["selective_error"],
            row["threshold"],
        ),
    )
    return float(selected["threshold"]), curve


def evaluate_model_set(
    model_set: Mapping[str, Any],
    features: np.ndarray,
    labels: np.ndarray,
    *,
    threshold: float,
    claim_names: Sequence[str],
    feature_names: Sequence[str],
    support_vetoes: Mapping[str, Sequence[str]],
    constrained: bool,
) -> dict[str, Any]:
    probabilities = predict_probabilities(model_set, features, claim_names)
    decisions, violations = selective_decisions(
        probabilities,
        features,
        threshold=threshold,
        claim_names=claim_names,
        feature_names=feature_names,
        support_vetoes=support_vetoes,
        constrained=constrained,
    )
    return {
        "threshold": threshold,
        "semantic_support_veto_violations": violations,
        **_decision_metrics(decisions, labels.astype(bool)),
    }


def _validate_inputs(
    matrix_path: Path,
    manifest_path: Path,
    *,
    test_mode: bool,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    manifest = json.loads(manifest_path.read_text())
    if manifest["matrix_sha256"] != _sha256(matrix_path):
        raise ValueError("development matrix hash differs from its manifest")
    if manifest.get("confirmatory_v2_cases_used") != 0:
        raise ValueError("confirmatory v2 cases leaked into development features")
    if not test_mode and (
        manifest.get("cases") != 3740
        or manifest.get("decision")
        != "hybrid_development_features_frozen_without_v2_confirmation_leakage"
    ):
        raise ValueError("full consumed development matrix is incomplete")
    archive = np.load(matrix_path)
    arrays = {name: archive[name] for name in archive.files}
    if arrays["features"].shape[0] != manifest["cases"]:
        raise ValueError("matrix row count differs from manifest")
    if not np.all(np.isfinite(arrays["features"])):
        raise ValueError("development features contain non-finite values")
    return manifest, arrays


def train(
    *,
    output: Path = DEFAULT_OUTPUT,
    protocol_path: Path = DEFAULT_PROTOCOL,
    matrix_path: Path = DEFAULT_MATRIX,
    manifest_path: Path = DEFAULT_MANIFEST,
    test_mode: bool = False,
) -> Path:
    """Fit, calibrate, threshold, audit, and serialize the hybrid policy."""

    protocol = _load_protocol(protocol_path)
    manifest, arrays = _validate_inputs(matrix_path, manifest_path, test_mode=test_mode)
    profile = load_default_profile()
    expected_profile = protocol["knowledge_profile"]
    if (
        profile.profile_id != expected_profile["profile_id"]
        or profile.version != expected_profile["version"]
        or profile.source_hash != expected_profile["source_hash"]
    ):
        raise ValueError("knowledge profile differs from frozen protocol")

    features = arrays["features"].astype(float)
    labels = arrays["labels"].astype(np.uint8)
    roles = arrays["split_role"].astype(str)
    claim_names = tuple(manifest["claim_names"])
    feature_names = tuple(manifest["feature_names"])
    if feature_names != hybrid_feature_names():
        raise ValueError("feature schema differs from implementation")
    masks = {
        role: roles == role
        for role in ("model_fit", "threshold_calibration", "locked_development_audit")
    }
    if any(not mask.any() for mask in masks.values()):
        raise ValueError("one or more deterministic development partitions are empty")

    settings = protocol["feature_model"]
    full_indices = np.arange(features.shape[1], dtype=int)
    anm_ablation_indices = np.asarray(
        [
            index
            for index, name in enumerate(feature_names)
            if not name.startswith("status:directed_identifiability:")
            and name != "missing:directed_identifiability"
            and not name.startswith("anm_")
        ],
        dtype=int,
    )
    model_sets = {
        "full": _fit_model_set(
            features[masks["model_fit"]],
            labels[masks["model_fit"]],
            features[masks["threshold_calibration"]],
            labels[masks["threshold_calibration"]],
            claim_names=claim_names,
            selected_indices=full_indices,
            regularization_c=float(settings["regularization_C"]),
            max_iterations=int(settings["max_iterations"]),
        ),
        "anm_predictor_ablation": _fit_model_set(
            features[masks["model_fit"]],
            labels[masks["model_fit"]],
            features[masks["threshold_calibration"]],
            labels[masks["threshold_calibration"]],
            claim_names=claim_names,
            selected_indices=anm_ablation_indices,
            regularization_c=float(settings["regularization_C"]),
            max_iterations=int(settings["max_iterations"]),
        ),
    }
    support_vetoes = protocol["semantic_support_vetoes"]
    support_vetoes = {
        claim: blocks
        for claim, blocks in support_vetoes.items()
        if claim not in {"support_veto_rule", "veto_override_permitted"}
    }
    calibration_features = features[masks["threshold_calibration"]]
    calibration_labels = labels[masks["threshold_calibration"]]
    thresholds: dict[str, float] = {}
    curves: dict[str, Any] = {}
    for name, model_set in model_sets.items():
        probabilities = predict_probabilities(model_set, calibration_features, claim_names)
        threshold, curve = _select_threshold(
            probabilities,
            calibration_features,
            calibration_labels,
            claim_names=claim_names,
            feature_names=feature_names,
            support_vetoes=support_vetoes,
            settings=protocol["selective_policy"],
            constrained=True,
        )
        thresholds[name] = threshold
        curves[name] = curve
    unconstrained_probabilities = predict_probabilities(
        model_sets["full"], calibration_features, claim_names
    )
    unconstrained_threshold, unconstrained_curve = _select_threshold(
        unconstrained_probabilities,
        calibration_features,
        calibration_labels,
        claim_names=claim_names,
        feature_names=feature_names,
        support_vetoes=support_vetoes,
        settings=protocol["selective_policy"],
        constrained=False,
    )
    thresholds["unconstrained_full"] = unconstrained_threshold
    curves["unconstrained_full"] = unconstrained_curve

    audit_features = features[masks["locked_development_audit"]]
    audit_labels = labels[masks["locked_development_audit"]]
    audit = {
        "constrained_full": evaluate_model_set(
            model_sets["full"],
            audit_features,
            audit_labels,
            threshold=thresholds["full"],
            claim_names=claim_names,
            feature_names=feature_names,
            support_vetoes=support_vetoes,
            constrained=True,
        ),
        "unconstrained_full": evaluate_model_set(
            model_sets["full"],
            audit_features,
            audit_labels,
            threshold=thresholds["unconstrained_full"],
            claim_names=claim_names,
            feature_names=feature_names,
            support_vetoes=support_vetoes,
            constrained=False,
        ),
        "constrained_anm_predictor_ablation": evaluate_model_set(
            model_sets["anm_predictor_ablation"],
            audit_features,
            audit_labels,
            threshold=thresholds["anm_predictor_ablation"],
            claim_names=claim_names,
            feature_names=feature_names,
            support_vetoes=support_vetoes,
            constrained=True,
        ),
    }

    payload = {
        "version": __version__,
        "git_revision": code_revision(),
        "runtime": {
            "python": platform.python_version(),
            "python_implementation": platform.python_implementation(),
            "platform": platform.platform(),
            "executable": sys.executable,
            "packages": {
                package: importlib.metadata.version(package)
                for package in (
                    "causal-learn",
                    "numpy",
                    "scikit-learn",
                    "scipy",
                )
            },
        },
        "analysis": "hybrid_selective_policy_frozen_before_v2_confirmation",
        "protocol_id": protocol["protocol_id"],
        "protocol_version": str(protocol["version"]),
        "protocol_hash": _sha256(protocol_path),
        "profile_id": profile.profile_id,
        "profile_version": profile.version,
        "profile_hash": profile.source_hash,
        "development_manifest_hash": _sha256(manifest_path),
        "development_matrix_hash": _sha256(matrix_path),
        "development_cases": int(len(features)),
        "split_counts": {role: int(mask.sum()) for role, mask in masks.items()},
        "feature_names": list(feature_names),
        "claim_names": list(claim_names),
        "support_vetoes": support_vetoes,
        "model_settings": settings,
        "model_sets": model_sets,
        "selected_thresholds": thresholds,
        "calibration_risk_coverage_curves": curves,
        "locked_development_audit": audit,
        "confirmatory_v2_cases_used": 0,
        "confirmatory_refitting_permitted": False,
        "anm_ablation_scope": (
            "ANM-derived status and continuous predictors are removed from the learned "
            "component; the frozen ANM semantic support veto remains active"
        ),
        "decision": "hybrid_policy_frozen_for_unseen_v2_confirmation",
        "test_mode": test_mode,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2))
    return output


def load_frozen_policy(
    path: Path = DEFAULT_OUTPUT,
    *,
    protocol_path: Path = DEFAULT_PROTOCOL,
) -> dict[str, Any]:
    """Load a policy only when its protocol and profile identities still match."""

    payload = json.loads(path.read_text())
    profile = load_default_profile()
    expected = {
        "protocol_hash": _sha256(protocol_path),
        "profile_id": profile.profile_id,
        "profile_version": profile.version,
        "profile_hash": profile.source_hash,
        "confirmatory_v2_cases_used": 0,
        "confirmatory_refitting_permitted": False,
        "decision": "hybrid_policy_frozen_for_unseen_v2_confirmation",
    }
    for key, value in expected.items():
        if payload.get(key) != value:
            raise ValueError(f"frozen hybrid policy has incompatible {key}")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--matrix", type=Path, default=DEFAULT_MATRIX)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--test-mode", action="store_true")
    args = parser.parse_args()
    print(
        json.dumps(
            {
                "output": str(
                    train(
                        output=args.output,
                        protocol_path=args.protocol,
                        matrix_path=args.matrix,
                        manifest_path=args.manifest,
                        test_mode=args.test_mode,
                    ).resolve()
                )
            }
        )
    )


if __name__ == "__main__":
    main()
