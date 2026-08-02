"""Target-calibrated, non-degenerate validation on unseen Causal Chambers data.

The script enforces the locked order calibration -> risk lock -> final.  Final
CSV values are not read unless the complete policy passes the risk-lock contract.
Inference is performed at the physical-experiment level.  Pair-level summaries
are descriptive because candidate pairs within an experiment share observations.
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

from mousebrainbench import __version__
from mousebrainbench.artifacts import code_revision
from mousebrainbench.benchmarks.causal_chambers_transport import (
    _eligible_columns,
    _ground_truth,
    _pair_record,
    direct_and_control_pairs,
)
from mousebrainbench.benchmarks.hybrid_selective_policy import (
    predict_probabilities,
    semantic_admissibility_matrix,
)
from mousebrainbench.knowledge import load_default_profile
from mousebrainbench.validation.nondegenerate_risk_control import (
    NonDegenerateCertificate,
    calibrate_nondegenerate_policy,
    evaluate_authorization_decisions,
)
from mousebrainbench.validation.semantic_risk_control import (
    SemanticRiskPolicy,
    authorize_with_policy,
)
from mousebrainbench.validation.shift_diagnostics import diagnose_shift

DEFAULT_PROTOCOL = Path("configs/benchmarks/semantic_risk_control_v4.yaml")
DEFAULT_POPULATION = Path("configs/benchmarks/causal_chambers_v4_population.yaml")
DEFAULT_ROOT = Path("data/external/causal_chambers_v4")
DEFAULT_SCORE_MODEL = Path("results/hybrid_selective_policy/model.json")
DEFAULT_RISK_POLICY = Path("results/semantic_risk_policy/model.json")
DEFAULT_OUTPUT = Path("results/causal_chambers_v4_confirmation/summary.json")
DEFAULT_MARKDOWN = Path("results/causal_chambers_v4_confirmation/summary.md")


@dataclass(frozen=True)
class RoleData:
    role: str
    records: tuple[dict[str, Any], ...]
    scores: np.ndarray
    labels: np.ndarray
    admissible: np.ndarray
    features: np.ndarray
    experiment_ids: np.ndarray
    datasets: np.ndarray
    file_hashes: dict[str, str]
    source_bytes: int
    exclusions: dict[str, int]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return f"sha256:{digest.hexdigest()}"


def _select_pairs(
    pairs: tuple[tuple[str, str, bool], ...],
    *,
    namespace: str,
    direct_count: int,
    control_count: int,
) -> tuple[tuple[str, str, bool], ...]:
    def order(item: tuple[str, str, bool]) -> str:
        return hashlib.sha256(f"{namespace}:{item[0]}:{item[1]}".encode()).hexdigest()

    direct = sorted((item for item in pairs if item[2]), key=order)[:direct_count]
    controls = sorted((item for item in pairs if not item[2]), key=order)[:control_count]
    return tuple(direct + controls)


def _role_data(
    role: str,
    *,
    population: dict[str, Any],
    protocol: dict[str, Any],
    root: Path,
    score_model: dict[str, Any],
    variable_claims: tuple[str, ...],
) -> RoleData:
    claim_names = tuple(str(value) for value in score_model["claim_names"])
    adapter = protocol["external_population"]["adapter"]
    records: list[dict[str, Any]] = []
    file_hashes: dict[str, str] = {}
    source_bytes = 0
    exclusions = {"insufficient_columns": 0, "no_selected_pair": 0, "invalid_pair_record": 0}
    for item in population["datasets"]:
        if item["role"] != role:
            continue
        dataset = str(item["name"])
        chamber = str(item["chamber"])
        dataset_root = root / dataset
        if not dataset_root.exists():
            raise FileNotFoundError(f"missing frozen dataset directory: {dataset_root}")
        variables, edges = _ground_truth(chamber)
        for path in sorted(dataset_root.glob("*.csv")):
            experiment_id = f"{dataset}/{path.stem}"
            file_hashes[str(path)] = _sha256(path)
            source_bytes += path.stat().st_size
            frame = pd.read_csv(path)
            eligible = _eligible_columns(frame, variables)
            if len(eligible) < 5:
                exclusions["insufficient_columns"] += 1
                continue
            available = direct_and_control_pairs(
                eligible, edges, namespace=experiment_id
            )
            selected = _select_pairs(
                available,
                namespace=f"mouseclaimbench-v4:{experiment_id}",
                direct_count=int(adapter["direct_pairs_per_experiment"]),
                control_count=int(adapter["control_pairs_per_experiment"]),
            )
            if not selected:
                exclusions["no_selected_pair"] += 1
                continue
            for source, target, direct_edge in selected:
                record = _pair_record(
                    frame,
                    dataset=dataset,
                    chamber=chamber,
                    experiment=path.stem,
                    source=source,
                    target=target,
                    direct_edge=direct_edge,
                    eligible=eligible,
                    claim_names=claim_names,
                )
                if record is None:
                    exclusions["invalid_pair_record"] += 1
                    continue
                records.append({**record, "experiment_id": experiment_id})
    if not records:
        raise RuntimeError(f"no usable records in role {role}")
    features = np.vstack([row["features"] for row in records])
    labels = np.vstack([row["labels"] for row in records]).astype(bool)
    probabilities = predict_probabilities(
        score_model["model_sets"]["full"], features, claim_names
    )
    requirements = {
        item.claim: item.required_blocks for item in load_default_profile().requirements
    }
    admissible = semantic_admissibility_matrix(
        features,
        claim_names=claim_names,
        feature_names=score_model["feature_names"],
        support_vetoes=requirements,
    )
    indices = np.asarray([claim_names.index(claim) for claim in variable_claims])
    return RoleData(
        role=role,
        records=tuple(records),
        scores=probabilities[:, indices],
        labels=labels[:, indices],
        admissible=admissible[:, indices],
        features=features,
        experiment_ids=np.asarray([row["experiment_id"] for row in records]),
        datasets=np.asarray([row["dataset"] for row in records]),
        file_hashes=file_hashes,
        source_bytes=source_bytes,
        exclusions=exclusions,
    )


def _limits(protocol: dict[str, Any]) -> dict[str, float]:
    contract = protocol["risk_and_activation_contract"]
    return {
        "target_risk": float(contract["target_experiment_failure_probability"]),
        "minimum_coverage": float(contract["minimum_authorized_experiment_coverage"]),
        "minimum_positive_recovery": float(contract["minimum_positive_recovery"]),
        "confidence": float(contract["confidence_level"]),
    }


def _certificate_for_decisions(
    decisions: np.ndarray, data: RoleData, limits: dict[str, float], *, threshold: float
) -> NonDegenerateCertificate:
    return evaluate_authorization_decisions(
        decisions,
        data.labels,
        data.admissible,
        data.experiment_ids,
        threshold=threshold,
        **limits,
    )


def _evaluate_comparators(
    data: RoleData,
    *,
    complete_threshold: float | None,
    confidence_threshold: float | None,
    frozen_semantic: SemanticRiskPolicy,
    frozen_unconstrained: SemanticRiskPolicy,
    limits: dict[str, float],
) -> dict[str, Any]:
    ones = np.ones_like(data.admissible, dtype=bool)
    decisions: dict[str, np.ndarray] = {
        "fixed_probability_0_5": data.scores >= 0.5,
        "evidence_contract_only": data.admissible,
        "unconstrained_ltt": authorize_with_policy(
            frozen_unconstrained, data.scores, ones
        ).astype(bool),
        "semantic_ltt_without_activation_floor": authorize_with_policy(
            frozen_semantic, data.scores, data.admissible
        ).astype(bool),
        "confidence_only_target_calibrated": (
            data.scores >= confidence_threshold
            if confidence_threshold is not None
            else np.zeros_like(data.admissible)
        ),
        "semantic_ltt_nondegenerate": (
            data.admissible & (data.scores >= complete_threshold)
            if complete_threshold is not None
            else np.zeros_like(data.admissible)
        ),
    }
    thresholds = {
        "fixed_probability_0_5": 0.5,
        "evidence_contract_only": 0.0,
        "unconstrained_ltt": -1.0,
        "semantic_ltt_without_activation_floor": -1.0,
        "confidence_only_target_calibrated": confidence_threshold or -1.0,
        "semantic_ltt_nondegenerate": complete_threshold or -1.0,
    }
    return {
        name: _certificate_for_decisions(
            value, data, limits, threshold=thresholds[name]
        ).as_dict()
        for name, value in decisions.items()
    }


def _experiment_feature_means(data: RoleData) -> np.ndarray:
    return np.vstack(
        [data.features[data.experiment_ids == unit].mean(axis=0) for unit in np.unique(data.experiment_ids)]
    )


def _failure_taxonomy(
    data: RoleData,
    *,
    threshold: float | None,
    calibration_failure: str | None,
    shift_warning: bool,
) -> dict[str, Any]:
    counts = {
        "invalid_input": 0,
        "out_of_population_scope": 0,
        "semantic_inadmissibility": 0,
        "uncertifiable_risk": 0,
        "activation_floor_not_met": 0,
        "score_below_threshold": 0,
        "numerical_failure": 0,
        "authorized": 0,
    }
    for score, gate in zip(data.scores.ravel(), data.admissible.ravel(), strict=True):
        if not np.isfinite(score):
            reason = "invalid_input"
        elif not gate:
            reason = "semantic_inadmissibility"
        elif threshold is None:
            reason = calibration_failure or "uncertifiable_risk"
        elif score < threshold:
            reason = "score_below_threshold"
        else:
            reason = "authorized"
        counts[reason] += 1
    total = sum(counts.values())
    return {
        "mutually_exclusive_counts": counts,
        "total_claim_candidates": total,
        "orthogonal_context_flags": {"detected_shift_warning": shift_warning},
    }


def _peak_rss_mb() -> float:
    value = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return value / (1024.0 * 1024.0) if sys.platform == "darwin" else value / 1024.0


def run(
    *,
    protocol_path: Path = DEFAULT_PROTOCOL,
    population_path: Path = DEFAULT_POPULATION,
    root: Path = DEFAULT_ROOT,
    score_model_path: Path = DEFAULT_SCORE_MODEL,
    risk_policy_path: Path = DEFAULT_RISK_POLICY,
    output: Path = DEFAULT_OUTPUT,
    markdown: Path = DEFAULT_MARKDOWN,
) -> Path:
    started = time.perf_counter()
    protocol = yaml.safe_load(protocol_path.read_text())
    population = yaml.safe_load(population_path.read_text())
    score_model = json.loads(score_model_path.read_text())
    frozen = json.loads(risk_policy_path.read_text())
    variable_claims = tuple(str(value) for value in frozen["variable_claims"])
    frozen_semantic = SemanticRiskPolicy.from_dict(frozen["semantic_policy"])
    frozen_unconstrained = SemanticRiskPolicy.from_dict(frozen["unconstrained_policy"])
    limits = _limits(protocol)

    calibration = _role_data(
        "target_calibration",
        population=population,
        protocol=protocol,
        root=root,
        score_model=score_model,
        variable_claims=variable_claims,
    )
    calibration_started = time.perf_counter()
    complete = calibrate_nondegenerate_policy(
        calibration.scores,
        calibration.labels,
        calibration.admissible,
        calibration.experiment_ids,
        **limits,
    )
    risk_only = calibrate_nondegenerate_policy(
        calibration.scores,
        calibration.labels,
        calibration.admissible,
        calibration.experiment_ids,
        target_risk=limits["target_risk"],
        minimum_coverage=0.0,
        minimum_positive_recovery=0.0,
        confidence=limits["confidence"],
    )
    confidence_only = calibrate_nondegenerate_policy(
        calibration.scores,
        calibration.labels,
        np.ones_like(calibration.admissible),
        calibration.experiment_ids,
        **limits,
    )
    calibration_seconds = time.perf_counter() - calibration_started
    calibration_failure = (
        None
        if complete is not None
        else "activation_floor_not_met"
        if risk_only is not None
        else "uncertifiable_risk"
    )

    risk_lock = _role_data(
        "risk_lock",
        population=population,
        protocol=protocol,
        root=root,
        score_model=score_model,
        variable_claims=variable_claims,
    )
    risk_shift = diagnose_shift(
        _experiment_feature_means(calibration),
        _experiment_feature_means(risk_lock),
        feature_names=tuple(score_model["feature_names"]),
    )
    complete_threshold = complete.threshold if complete is not None else None
    confidence_threshold = confidence_only.threshold if confidence_only is not None else None
    risk_comparators = _evaluate_comparators(
        risk_lock,
        complete_threshold=complete_threshold,
        confidence_threshold=confidence_threshold,
        frozen_semantic=frozen_semantic,
        frozen_unconstrained=frozen_unconstrained,
        limits=limits,
    )
    risk_primary = risk_comparators["semantic_ltt_nondegenerate"]
    risk_lock_passed = complete is not None and bool(risk_primary["certified"])

    final_payload: dict[str, Any] = {
        "opened": False,
        "reason": "risk_lock_did_not_pass",
    }
    total_bytes = calibration.source_bytes + risk_lock.source_bytes
    total_candidates = calibration.scores.size + risk_lock.scores.size
    total_authorizations = int(risk_primary["authorizations"])
    if risk_lock_passed:
        final = _role_data(
            "final_evaluation",
            population=population,
            protocol=protocol,
            root=root,
            score_model=score_model,
            variable_claims=variable_claims,
        )
        final_shift = diagnose_shift(
            _experiment_feature_means(calibration),
            _experiment_feature_means(final),
            feature_names=tuple(score_model["feature_names"]),
        )
        final_comparators = _evaluate_comparators(
            final,
            complete_threshold=complete_threshold,
            confidence_threshold=confidence_threshold,
            frozen_semantic=frozen_semantic,
            frozen_unconstrained=frozen_unconstrained,
            limits=limits,
        )
        final_primary = final_comparators["semantic_ltt_nondegenerate"]
        final_payload = {
            "opened": True,
            "experiments": len(np.unique(final.experiment_ids)),
            "pair_records": len(final.records),
            "datasets": sorted(set(final.datasets)),
            "source_bytes": final.source_bytes,
            "file_hashes": final.file_hashes,
            "exclusions": final.exclusions,
            "shift_diagnostic": final_shift,
            "comparators": final_comparators,
            "failure_taxonomy": _failure_taxonomy(
                final,
                threshold=complete_threshold,
                calibration_failure=calibration_failure,
                shift_warning=bool(final_shift["warning"]),
            ),
            "externally_supported": bool(final_primary["certified"]),
        }
        total_bytes += final.source_bytes
        total_candidates += final.scores.size
        total_authorizations += int(final_primary["authorizations"])

    elapsed = time.perf_counter() - started
    result = {
        "version": __version__,
        "git_revision": code_revision(),
        "analysis": "causal_chambers_v4_target_calibrated_confirmation",
        "protocol": str(protocol_path),
        "population_manifest": str(population_path),
        "protocol_status": protocol["status"],
        "population_outcome_access_before_freeze": population[
            "outcome_access_before_freeze"
        ],
        "inferential_unit": "physical_experiment_csv",
        "pair_level_inference_permitted": False,
        "exact_bound_assumption": "exchangeable independent experiment-level events",
        "within_dataset_dependence_sensitivity_required": True,
        "variable_claims": list(variable_claims),
        "target_limits": limits,
        "calibration": {
            "experiments": len(np.unique(calibration.experiment_ids)),
            "pair_records": len(calibration.records),
            "datasets": sorted(set(calibration.datasets)),
            "source_bytes": calibration.source_bytes,
            "file_hashes": calibration.file_hashes,
            "exclusions": calibration.exclusions,
            "complete_policy": complete.as_dict() if complete else None,
            "risk_only_policy": risk_only.as_dict() if risk_only else None,
            "confidence_only_policy": confidence_only.as_dict() if confidence_only else None,
            "failure_reason": calibration_failure,
        },
        "risk_lock": {
            "experiments": len(np.unique(risk_lock.experiment_ids)),
            "pair_records": len(risk_lock.records),
            "datasets": sorted(set(risk_lock.datasets)),
            "source_bytes": risk_lock.source_bytes,
            "file_hashes": risk_lock.file_hashes,
            "exclusions": risk_lock.exclusions,
            "shift_diagnostic": risk_shift,
            "comparators": risk_comparators,
            "failure_taxonomy": _failure_taxonomy(
                risk_lock,
                threshold=complete_threshold,
                calibration_failure=calibration_failure,
                shift_warning=bool(risk_shift["warning"]),
            ),
            "passed": risk_lock_passed,
        },
        "final_evaluation": final_payload,
        "efficiency": {
            "wall_time_seconds": elapsed,
            "calibration_overhead_seconds": calibration_seconds,
            "peak_rss_megabytes": _peak_rss_mb(),
            "source_bytes_read": total_bytes,
            "claim_candidates": total_candidates,
            "claim_candidates_per_second": total_candidates / elapsed,
            "authorized_claims": total_authorizations,
            "authorized_claims_per_second": total_authorizations / elapsed,
            "abstentions": total_candidates - total_authorizations,
            "abstention_rate": (total_candidates - total_authorizations) / total_candidates,
        },
        "decision": (
            "v4_external_contract_supported"
            if final_payload.get("externally_supported") is True
            else "v4_external_contract_not_supported"
        ),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    _write_markdown(result, markdown)
    return output


def _write_markdown(payload: dict[str, Any], path: Path) -> None:
    risk = payload["risk_lock"]["comparators"]
    lines = [
        "# Causal Chambers v4 target-calibrated confirmation",
        "",
        f"- Decision: `{payload['decision']}`",
        f"- Risk lock passed: `{str(payload['risk_lock']['passed']).lower()}`",
        f"- Final evaluation opened: `{str(payload['final_evaluation']['opened']).lower()}`",
        f"- Calibration experiments: `{payload['calibration']['experiments']}`",
        f"- Risk-lock experiments: `{payload['risk_lock']['experiments']}`",
        "",
        "## Risk-lock comparators",
        "",
        "| Policy | Risk UCB | Coverage LCB | Recovery LCB | Certified |",
        "|---|---:|---:|---:|---:|",
    ]
    lines.extend(
        f"| `{name}` | {row['risk_upper_bound']:.4f} | "
        f"{row['coverage_lower_bound']:.4f} | {row['positive_recovery_lower_bound']:.4f} | "
        f"{str(row['certified']).lower()} |"
        for name, row in risk.items()
    )
    if payload["final_evaluation"]["opened"]:
        lines.extend(("", "## Final evaluation", ""))
        lines.append(
            f"Externally supported: `{str(payload['final_evaluation']['externally_supported']).lower()}`"
        )
    lines.extend(
        (
            "",
            (
                "The exact interval is conditional on exchangeability of experiment-level "
                "events. Pair rows are not inferential replicates, and within-dataset "
                "dependence remains a declared sensitivity requirement."
            ),
            "",
        )
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--population", type=Path, default=DEFAULT_POPULATION)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--score-model", type=Path, default=DEFAULT_SCORE_MODEL)
    parser.add_argument("--risk-policy", type=Path, default=DEFAULT_RISK_POLICY)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--markdown", type=Path, default=DEFAULT_MARKDOWN)
    args = parser.parse_args()
    path = run(
        protocol_path=args.protocol,
        population_path=args.population,
        root=args.root,
        score_model_path=args.score_model,
        risk_policy_path=args.risk_policy,
        output=args.output,
        markdown=args.markdown,
    )
    print(json.dumps({"output": str(path.resolve())}))


if __name__ == "__main__":
    main()
