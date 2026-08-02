"""External mouse-level validation on the frozen IBL behavioral population.

The endpoint asks whether a claim authorizer distinguishes the actual alignment
between randomized visual evidence and choice from three circularly shifted
controls. Trial rows produce evidence features only. All uncertainty statements
collapse decisions to one worst-case event per mouse.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import resource
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from scipy.stats import pearsonr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from mousebrainbench import __version__
from mousebrainbench.artifacts import code_revision
from mousebrainbench.benchmarks.hybrid_development_features import encode_hybrid_features
from mousebrainbench.benchmarks.hybrid_selective_policy import (
    predict_probabilities,
    semantic_admissibility_matrix,
)
from mousebrainbench.knowledge import load_default_profile
from mousebrainbench.validation.evidence_contract import (
    EvidenceBlock,
    EvidenceStatus,
    blocks_by_name,
)
from mousebrainbench.validation.hierarchical_risk_control import (
    evaluate_hierarchical_decisions,
    evaluate_hierarchical_policy,
)

DEFAULT_PROTOCOL = Path("configs/benchmarks/ibl_behavior_v5_confirmation.yaml")
DEFAULT_MANIFEST = Path("data/external/ibl_behavior_v5/manifest.json")
DEFAULT_SCORE_MODEL = Path("results/hybrid_selective_policy/model.json")
DEFAULT_OUTPUT = Path("results/ibl_behavior_v5_confirmation/summary.json")
DEFAULT_MARKDOWN = Path("results/ibl_behavior_v5_confirmation/summary.md")
DEFAULT_DETAILS = Path("results/ibl_behavior_v5_confirmation/mouse_candidates.csv")
DEFAULT_SOURCE_MANIFEST = Path("results/ibl_behavior_v5_confirmation/source_manifest.json")


@dataclass(frozen=True)
class CandidateDiagnostic:
    """Held-out behavior metrics for one candidate stimulus alignment."""

    offset: int
    correlation: float
    p_value: float
    tjur_r_squared: float
    mean_log_loss_improvement: float
    positive_improvement_folds: int
    fold_correlations: tuple[float, ...]
    fold_tjur_r_squared: tuple[float, ...]
    fold_log_loss_improvements: tuple[float, ...]


@dataclass(frozen=True)
class RoleData:
    """Mouse-nested candidate claims for one frozen role."""

    role: str
    records: tuple[dict[str, Any], ...]
    scores: np.ndarray
    labels: np.ndarray
    admissible: np.ndarray
    top_level_ids: np.ndarray
    subgroup_ids: np.ndarray
    strata: np.ndarray
    selected_mice: int
    usable_mice: int


def deterministic_trial_folds(eid: str, trial_indices: np.ndarray, folds: int = 3) -> np.ndarray:
    """Assign immutable trial indices without Python hash randomization."""

    return np.asarray(
        [
            int(hashlib.sha256(f"ibl-v5:{eid}:{int(index)}".encode()).hexdigest(), 16)
            % folds
            for index in trial_indices
        ],
        dtype=np.int8,
    )


def _model() -> Any:
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(C=1.0, max_iter=2_000, random_state=0),
    )


def _tjur_r_squared(observed: np.ndarray, probability: np.ndarray) -> float:
    if len(np.unique(observed)) != 2:
        return 0.0
    return float(probability[observed == 1].mean() - probability[observed == 0].mean())


def candidate_diagnostic(
    choice_right: np.ndarray,
    candidate_contrast: np.ndarray,
    probability_left: np.ndarray,
    folds: np.ndarray,
    *,
    offset: int,
) -> CandidateDiagnostic:
    """Compare candidate-plus-block prediction with the block-only baseline."""

    full_predictions = np.full(len(choice_right), np.nan, dtype=float)
    baseline_predictions = np.full(len(choice_right), np.nan, dtype=float)
    fold_correlations: list[float] = []
    fold_tjur: list[float] = []
    fold_improvements: list[float] = []
    for fold in sorted(np.unique(folds)):
        test = folds == fold
        train = ~test
        if len(np.unique(choice_right[train])) != 2 or len(np.unique(choice_right[test])) != 2:
            raise ValueError("each behavioral fold must contain both choices")
        block_train = probability_left[train, None]
        block_test = probability_left[test, None]
        full_train = np.column_stack((probability_left[train], candidate_contrast[train]))
        full_test = np.column_stack((probability_left[test], candidate_contrast[test]))
        baseline = _model().fit(block_train, choice_right[train])
        augmented = _model().fit(full_train, choice_right[train])
        baseline_probability = baseline.predict_proba(block_test)[:, 1]
        full_probability = augmented.predict_proba(full_test)[:, 1]
        baseline_predictions[test] = baseline_probability
        full_predictions[test] = full_probability
        correlation = pearsonr(full_probability, choice_right[test])
        fold_correlations.append(float(correlation.statistic))
        fold_tjur.append(_tjur_r_squared(choice_right[test], full_probability))
        fold_improvements.append(
            float(
                log_loss(choice_right[test], baseline_probability, labels=[0, 1])
                - log_loss(choice_right[test], full_probability, labels=[0, 1])
            )
        )
    if not np.all(np.isfinite(full_predictions)) or not np.all(np.isfinite(baseline_predictions)):
        raise RuntimeError("cross-validation did not predict every usable trial")
    pooled = pearsonr(full_predictions, choice_right)
    return CandidateDiagnostic(
        offset=offset,
        correlation=float(pooled.statistic),
        p_value=float(pooled.pvalue),
        tjur_r_squared=_tjur_r_squared(choice_right, full_predictions),
        mean_log_loss_improvement=float(np.mean(fold_improvements)),
        positive_improvement_folds=sum(value > 0.0 for value in fold_improvements),
        fold_correlations=tuple(fold_correlations),
        fold_tjur_r_squared=tuple(fold_tjur),
        fold_log_loss_improvements=tuple(fold_improvements),
    )


def _status_block(
    name: str,
    status: EvidenceStatus,
    rule: str,
    observations: dict[str, Any],
) -> EvidenceBlock:
    return EvidenceBlock.from_mapping(
        name=name,
        status=status,
        source="frozen IBL behavioral adapter v5.3.1",
        rule=rule,
        rationale="the diagnostic follows the outcome-frozen protocol",
        observations=observations,
    )


def _candidate_blocks(
    diagnostic: CandidateDiagnostic,
    *,
    best_other_tjur: float,
    eid: str,
    thresholds: dict[str, Any],
) -> dict[str, EvidenceBlock]:
    prediction_rule = thresholds["prediction_pass_requires"]
    topology_rule = thresholds["topology_specificity_requires"]
    reproduction_rule = thresholds["internal_reproduction_requires"]
    prediction_passed = bool(
        diagnostic.correlation
        >= float(prediction_rule["pooled_probability_choice_correlation_minimum"])
        and diagnostic.p_value
        <= float(prediction_rule["pooled_correlation_p_value_maximum"])
        and diagnostic.tjur_r_squared >= float(prediction_rule["held_out_tjur_r2_minimum"])
    )
    reproduction_passed = bool(
        diagnostic.positive_improvement_folds
        >= int(reproduction_rule["minimum_folds_with_positive_log_loss_improvement"])
    )
    margin = diagnostic.tjur_r_squared - best_other_tjur
    topology_passed = bool(
        margin
        >= float(topology_rule["held_out_tjur_r2_margin_over_best_other_candidate"])
    )
    prediction = {
        "correlation": diagnostic.correlation,
        "p_value": diagnostic.p_value,
        "r_squared": diagnostic.tjur_r_squared,
        "mean_log_loss_improvement": diagnostic.mean_log_loss_improvement,
    }
    reproduction = {
        "first": {
            "correlation": diagnostic.fold_correlations[0],
            "r_squared": diagnostic.fold_tjur_r_squared[0],
        },
        "second": {
            "correlation": diagnostic.fold_correlations[1],
            "r_squared": diagnostic.fold_tjur_r_squared[1],
        },
        "positive_improvement_folds": diagnostic.positive_improvement_folds,
        "fold_log_loss_improvements": diagnostic.fold_log_loss_improvements,
    }
    topology = {
        "candidate_r_squared": diagnostic.tjur_r_squared,
        "best_control_r_squared": best_other_tjur,
        "best_control_margin": margin,
        "candidate_offset": diagnostic.offset,
    }
    return blocks_by_name(
        (
            _status_block(
                "prediction",
                EvidenceStatus.PASSED if prediction_passed else EvidenceStatus.FAILED,
                "pooled correlation >= 0.10, p <= 0.01, and held-out Tjur R2 >= 0.02",
                prediction,
            ),
            _status_block(
                "reproducible_compute",
                EvidenceStatus.PASSED,
                "Alyx UUID, revision, byte count, and MD5 are frozen",
                {"eid": eid},
            ),
            _status_block(
                "internal_reproduction",
                EvidenceStatus.PASSED if reproduction_passed else EvidenceStatus.FAILED,
                "at least two of three folds improve held-out log loss",
                reproduction,
            ),
            _status_block(
                "external_replication",
                EvidenceStatus.NOT_APPLICABLE,
                "mice share one consortium task and are not cross-laboratory replications",
                {},
            ),
            _status_block(
                "topology_specificity",
                EvidenceStatus.PASSED if topology_passed else EvidenceStatus.FAILED,
                "candidate Tjur R2 exceeds every other frozen alignment by at least 0.02",
                topology,
            ),
            _status_block(
                "directed_identifiability",
                EvidenceStatus.REQUIRES_REVIEW,
                "behavioral alignment does not identify a directed neural mechanism",
                {},
            ),
            _status_block(
                "structure_function_association",
                EvidenceStatus.NOT_APPLICABLE,
                "the endpoint contains no anatomical structure variable",
                {},
            ),
            _status_block(
                "causal_intervention",
                EvidenceStatus.REQUIRES_REVIEW,
                "randomized stimulus assignment does not identify a neural intervention",
                {},
            ),
            _status_block(
                "whole_brain_coverage",
                EvidenceStatus.NOT_APPLICABLE,
                "the behavioral endpoint is not a whole-brain measurement",
                {},
            ),
            _status_block(
                "independent_validation",
                EvidenceStatus.PASSED,
                "selected IBL mice were excluded from authorizer development",
                {},
            ),
            _status_block(
                "entity_specificity",
                EvidenceStatus.NOT_APPLICABLE,
                "no individual digital-twin claim is evaluated",
                {},
            ),
            _status_block(
                "operational_compute",
                EvidenceStatus.NOT_APPLICABLE,
                "no digital-twin runtime target is declared",
                {},
            ),
        )
    )


def _usable_trials(frame: pd.DataFrame, eid: str, minimum: int) -> tuple[np.ndarray, ...]:
    required = {"contrastLeft", "contrastRight", "choice", "probabilityLeft"}
    if not required.issubset(frame.columns):
        raise ValueError(f"{eid} is missing required trial columns")
    choice = pd.to_numeric(frame["choice"], errors="coerce").to_numpy(dtype=float)
    left = pd.to_numeric(frame["contrastLeft"], errors="coerce").to_numpy(dtype=float)
    right = pd.to_numeric(frame["contrastRight"], errors="coerce").to_numpy(dtype=float)
    probability_left = pd.to_numeric(frame["probabilityLeft"], errors="coerce").to_numpy(dtype=float)
    valid = np.isin(choice, (-1.0, 1.0)) & np.isfinite(probability_left)
    indices = np.flatnonzero(valid)
    if len(indices) < minimum:
        raise ValueError(f"{eid} has {len(indices)} usable trials, below {minimum}")
    signed_contrast = 100.0 * (np.nan_to_num(right[valid]) - np.nan_to_num(left[valid]))
    choice_right = (choice[valid] == -1.0).astype(np.uint8)
    return choice_right, signed_contrast, probability_left[valid], indices


def _mouse_records(
    entry: dict[str, Any],
    *,
    protocol: dict[str, Any],
    claim_names: tuple[str, ...],
) -> list[dict[str, Any]]:
    task = protocol["behavioral_task"]
    frame = pd.read_parquet(entry["local_path"])
    choice, contrast, probability_left, trial_indices = _usable_trials(
        frame,
        str(entry["eid"]),
        int(protocol["source_contract"]["minimum_usable_trials"]),
    )
    folds = deterministic_trial_folds(str(entry["eid"]), trial_indices, int(task["folds"]))
    diagnostics = [
        candidate_diagnostic(
            choice,
            np.roll(contrast, int(offset)),
            probability_left,
            folds,
            offset=int(offset),
        )
        for offset in task["candidate_offsets"]
    ]
    output: list[dict[str, Any]] = []
    for diagnostic in diagnostics:
        best_other = max(
            item.tjur_r_squared for item in diagnostics if item.offset != diagnostic.offset
        )
        blocks = _candidate_blocks(
            diagnostic,
            best_other_tjur=best_other,
            eid=str(entry["eid"]),
            thresholds=task,
        )
        feature = encode_hybrid_features(
            blocks,
            legacy_direction={},
            anm={"status": "requires_review"},
            sample_size=len(choice),
            noise_scale=0.0,
        )
        output.append(
            {
                "subject": str(entry["subject"]),
                "eid": str(entry["eid"]),
                "lab": str(entry["lab"]),
                "qc": str(entry["qc"]),
                "offset": diagnostic.offset,
                "true_candidate": diagnostic.offset == int(task["true_candidate_offset"]),
                "usable_trials": len(choice),
                "features": feature,
                "labels": np.asarray(
                    [
                        name == "topology_specific" and diagnostic.offset == 0
                        for name in claim_names
                    ],
                    dtype=np.uint8,
                ),
                "diagnostic": diagnostic,
                "topology_margin": diagnostic.tjur_r_squared - best_other,
            }
        )
    return output


def _role_data(
    role: str,
    *,
    manifest: dict[str, Any],
    protocol: dict[str, Any],
    score_model: dict[str, Any],
) -> RoleData:
    claim_names = tuple(str(value) for value in score_model["claim_names"])
    claim_index = claim_names.index("topology_specific")
    selected = [entry for entry in manifest["entries"] if entry["role"] == role]
    records: list[dict[str, Any]] = []
    for entry in selected:
        if entry["status"] != "verified":
            continue
        try:
            records.extend(
                _mouse_records(entry, protocol=protocol, claim_names=claim_names)
            )
        except (ValueError, RuntimeError):
            continue
    if not records:
        raise RuntimeError(f"no usable IBL mice in role {role}")
    features = np.vstack([row["features"] for row in records])
    scores = predict_probabilities(
        score_model["model_sets"]["full"], features, claim_names
    )[:, [claim_index]]
    labels = np.vstack([row["labels"] for row in records])[:, [claim_index]].astype(bool)
    requirements = {
        item.claim: item.required_blocks for item in load_default_profile().requirements
    }
    admissible = semantic_admissibility_matrix(
        features,
        claim_names=claim_names,
        feature_names=score_model["feature_names"],
        support_vetoes=requirements,
    )[:, [claim_index]]
    return RoleData(
        role=role,
        records=tuple(records),
        scores=scores,
        labels=labels,
        admissible=admissible,
        top_level_ids=np.asarray([row["subject"] for row in records]),
        subgroup_ids=np.asarray([f"{row['eid']}/offset-{row['offset']}" for row in records]),
        strata=np.asarray([f"qc-{row['qc']}" for row in records]),
        selected_mice=len(selected),
        usable_mice=len({row["subject"] for row in records}),
    )


def _limits(protocol: dict[str, Any]) -> dict[str, float | int]:
    contract = protocol["inferential_contract"]
    return {
        "target_risk": float(contract["target_mouse_failure_probability"]),
        "minimum_coverage": float(contract["minimum_authorized_mouse_coverage"]),
        "minimum_positive_recovery": float(contract["minimum_positive_recovery"]),
        "minimum_independent_units": int(contract["minimum_independent_mice"]),
        "confidence": float(contract["confidence_level"]),
    }


def _evaluate(data: RoleData, protocol: dict[str, Any]) -> dict[str, Any]:
    threshold = float(protocol["authorization"]["fixed_threshold"])
    limits = _limits(protocol)
    zeros = np.zeros_like(data.admissible, dtype=bool)
    decisions = {
        "abstain_all": zeros,
        "fixed_probability_0_5": data.scores >= 0.5,
        "evidence_contract_only": data.admissible,
        "frozen_v5_1_complete_authorizer": data.admissible & (data.scores >= threshold),
    }
    thresholds = {
        "abstain_all": 2.0,
        "fixed_probability_0_5": 0.5,
        "evidence_contract_only": 0.0,
        "frozen_v5_1_complete_authorizer": threshold,
    }
    output = {
        name: evaluate_hierarchical_decisions(
            value,
            data.labels,
            data.admissible,
            data.top_level_ids,
            data.subgroup_ids,
            data.strata,
            threshold=thresholds[name],
            **limits,
        ).as_dict()
        for name, value in decisions.items()
    }
    # PASS-only is intentionally descriptive because each locked split has <29 mice.
    pass_rows = np.asarray([row["qc"] == "PASS" for row in data.records])
    if np.any(pass_rows):
        output["pass_only_sensitivity"] = evaluate_hierarchical_policy(
            data.scores[pass_rows],
            data.labels[pass_rows],
            data.admissible[pass_rows],
            data.top_level_ids[pass_rows],
            data.subgroup_ids[pass_rows],
            data.strata[pass_rows],
            threshold=threshold,
            **limits,
        ).as_dict()
    return output


def _role_summary(data: RoleData, evaluations: dict[str, Any]) -> dict[str, Any]:
    actual = [row for row in data.records if row["true_candidate"]]
    prediction_rule = [
        row
        for row in data.records
        if row["diagnostic"].correlation >= 0.10 and row["diagnostic"].p_value <= 0.01
    ]
    return {
        "selected_mice": data.selected_mice,
        "usable_mice": data.usable_mice,
        "candidate_rows": len(data.records),
        "qc_counts": {
            qc: len({row["subject"] for row in data.records if row["qc"] == qc})
            for qc in ("PASS", "WARNING")
        },
        "actual_alignment": {
            "median_tjur_r_squared": float(
                np.median([row["diagnostic"].tjur_r_squared for row in actual])
            ),
            "median_topology_margin": float(np.median([row["topology_margin"] for row in actual])),
        },
        "posthoc_prediction_only_diagnostic": {
            "status": "added_after_primary_outcomes_descriptive_only",
            "authorized_candidate_rows": len(prediction_rule),
            "mice_with_false_temporal_control": len(
                {
                    row["subject"]
                    for row in prediction_rule
                    if not row["true_candidate"]
                }
            ),
            "interpretation": (
                "correlation without alignment specificity is not a prespecified comparator"
            ),
        },
        "comparators": evaluations,
    }


def _detail_rows(data: RoleData) -> list[dict[str, Any]]:
    return [
        {
            "role": data.role,
            "subject": row["subject"],
            "eid": row["eid"],
            "lab": row["lab"],
            "qc": row["qc"],
            "offset": row["offset"],
            "true_candidate": row["true_candidate"],
            "usable_trials": row["usable_trials"],
            "correlation": row["diagnostic"].correlation,
            "p_value": row["diagnostic"].p_value,
            "tjur_r_squared": row["diagnostic"].tjur_r_squared,
            "mean_log_loss_improvement": row["diagnostic"].mean_log_loss_improvement,
            "topology_margin": row["topology_margin"],
            "score": float(row_score),
            "admissible": bool(row_gate),
        }
        for row, row_score, row_gate in zip(
            data.records, data.scores[:, 0], data.admissible[:, 0], strict=True
        )
    ]


def _peak_rss_mb() -> float:
    value = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return value / (1024.0 * 1024.0) if sys.platform == "darwin" else value / 1024.0


def run(
    *,
    protocol_path: Path = DEFAULT_PROTOCOL,
    manifest_path: Path = DEFAULT_MANIFEST,
    score_model_path: Path = DEFAULT_SCORE_MODEL,
    output: Path = DEFAULT_OUTPUT,
    markdown: Path = DEFAULT_MARKDOWN,
    details: Path = DEFAULT_DETAILS,
    source_manifest: Path = DEFAULT_SOURCE_MANIFEST,
) -> Path:
    """Run calibration context, locked risk, and conditionally opened final mice."""

    started = time.perf_counter()
    protocol = yaml.safe_load(protocol_path.read_text())
    if protocol["status"] != "amended_from_catalog_metadata_before_behavioral_value_access":
        raise ValueError("IBL behavioral protocol is not frozen with its disclosed amendment")
    manifest = json.loads(manifest_path.read_text())
    score_model = json.loads(score_model_path.read_text())
    expected_hash = str(protocol["authorization"]["score_model_sha256"])
    observed_hash = hashlib.sha256(score_model_path.read_bytes()).hexdigest()
    if observed_hash != expected_hash:
        raise ValueError("frozen score-model hash mismatch")

    calibration = _role_data(
        "calibration", manifest=manifest, protocol=protocol, score_model=score_model
    )
    calibration_evaluation = _evaluate(calibration, protocol)
    risk = _role_data("risk_lock", manifest=manifest, protocol=protocol, score_model=score_model)
    risk_evaluation = _evaluate(risk, protocol)
    primary = "frozen_v5_1_complete_authorizer"
    risk_passed = bool(risk_evaluation[primary]["certified"])

    final_data: RoleData | None = None
    final_payload: dict[str, Any] = {"opened": False, "reason": "risk_lock_did_not_pass"}
    if risk_passed:
        final_data = _role_data("final", manifest=manifest, protocol=protocol, score_model=score_model)
        final_evaluation = _evaluate(final_data, protocol)
        final_payload = {
            "opened": True,
            **_role_summary(final_data, final_evaluation),
            "passed": bool(final_evaluation[primary]["certified"]),
        }
    supported = bool(risk_passed and final_payload.get("passed") is True)
    result = {
        "version": __version__,
        "git_revision": code_revision(),
        "analysis": "ibl_behavior_v5_external_mouse_population",
        "protocol": str(protocol_path),
        "protocol_version": protocol["version"],
        "protocol_status": protocol["status"],
        "protocol_amendment": protocol["amendment"],
        "manifest": str(manifest_path),
        "manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        "selected_mice": manifest["selected_mice"],
        "verified_tables": manifest["verified_tables"],
        "inferential_unit": "mouse",
        "trial_level_inference_permitted": False,
        "fixed_claim": protocol["authorization"]["claim"],
        "fixed_threshold": protocol["authorization"]["fixed_threshold"],
        "threshold_refitted": False,
        "calibration_context": _role_summary(calibration, calibration_evaluation),
        "risk_lock": {
            **_role_summary(risk, risk_evaluation),
            "passed": risk_passed,
        },
        "final_evaluation": final_payload,
        "decision": (
            "external_ibl_behavioral_population_supported"
            if supported
            else "external_ibl_behavioral_population_not_supported"
        ),
        "claim_boundary": protocol["claim_boundary"],
        "efficiency": {
            "wall_time_seconds": time.perf_counter() - started,
            "peak_rss_megabytes": _peak_rss_mb(),
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    detail_data = _detail_rows(calibration) + _detail_rows(risk)
    if final_data is not None:
        detail_data += _detail_rows(final_data)
    details.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(detail_data).to_csv(details, index=False)
    source_manifest.write_text(json.dumps(manifest, indent=2) + "\n")
    _write_markdown(result, markdown)
    return output


def _write_markdown(payload: dict[str, Any], path: Path) -> None:
    primary = "frozen_v5_1_complete_authorizer"
    risk = payload["risk_lock"]["comparators"][primary]
    final = payload["final_evaluation"]
    lines = [
        "# IBL behavioral v5 external population",
        "",
        f"- Decision: `{payload['decision']}`",
        f"- Verified tables: `{payload['verified_tables']}`",
        f"- Inferential unit: `{payload['inferential_unit']}`",
        f"- Risk lock passed: `{str(payload['risk_lock']['passed']).lower()}`",
        f"- Risk-lock risk UCB: `{risk['risk_upper_bound']:.6f}`",
        f"- Risk-lock coverage LCB: `{risk['coverage_lower_bound']:.6f}`",
        f"- Final opened: `{str(final['opened']).lower()}`",
        "",
        "## Risk-lock comparators",
        "",
        "| Comparator | Certified | Failures | Risk UCB | Coverage LCB | Recovery LCB |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name, row in payload["risk_lock"]["comparators"].items():
        lines.append(
            f"| `{name}` | {str(row['certified']).lower()} | "
            f"{row['failing_experiments']} | {row['risk_upper_bound']:.4f} | "
            f"{row['coverage_lower_bound']:.4f} | {row['positive_recovery_lower_bound']:.4f} |"
        )
    lines.extend(("", "## Scope", "", payload["claim_boundary"], ""))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--score-model", type=Path, default=DEFAULT_SCORE_MODEL)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--markdown", type=Path, default=DEFAULT_MARKDOWN)
    parser.add_argument("--details", type=Path, default=DEFAULT_DETAILS)
    parser.add_argument("--source-manifest", type=Path, default=DEFAULT_SOURCE_MANIFEST)
    args = parser.parse_args()
    print(
        run(
            protocol_path=args.protocol,
            manifest_path=args.manifest,
            score_model_path=args.score_model,
            output=args.output,
            markdown=args.markdown,
            details=args.details,
            source_manifest=args.source_manifest,
        ).resolve()
    )


if __name__ == "__main__":
    main()
