"""Run the locked v4.1 claim-family policy on untouched final experiments."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from mousebrainbench import __version__
from mousebrainbench.artifacts import code_revision
from mousebrainbench.benchmarks.causal_chambers_v4_confirmation import (
    DEFAULT_POPULATION,
    DEFAULT_RISK_POLICY,
    DEFAULT_ROOT,
    DEFAULT_SCORE_MODEL,
    RoleData,
    _certificate_for_decisions,
    _experiment_feature_means,
    _failure_taxonomy,
    _limits,
    _peak_rss_mb,
    _role_data,
)
from mousebrainbench.validation.nondegenerate_risk_control import (
    NonDegenerateCertificate,
    calibrate_nondegenerate_policy,
)
from mousebrainbench.validation.semantic_risk_control import (
    SemanticRiskPolicy,
    authorize_with_policy,
)
from mousebrainbench.validation.shift_diagnostics import diagnose_shift

DEFAULT_PARENT_PROTOCOL = Path("configs/benchmarks/semantic_risk_control_v4.yaml")
DEFAULT_PROTOCOL = Path("configs/benchmarks/semantic_risk_control_v4_1.yaml")
DEFAULT_OUTPUT = Path("results/causal_chambers_v4_1_final/summary.json")
DEFAULT_MARKDOWN = Path("results/causal_chambers_v4_1_final/summary.md")


def _subset(data: RoleData, indices: list[int]) -> RoleData:
    return RoleData(
        role=data.role,
        records=data.records,
        scores=data.scores[:, indices],
        labels=data.labels[:, indices],
        admissible=data.admissible[:, indices],
        features=data.features,
        experiment_ids=data.experiment_ids,
        datasets=data.datasets,
        file_hashes=data.file_hashes,
        source_bytes=data.source_bytes,
        exclusions=data.exclusions,
    )


def _combine(first: RoleData, second: RoleData) -> RoleData:
    return RoleData(
        role="pooled_consumed_development",
        records=first.records + second.records,
        scores=np.vstack((first.scores, second.scores)),
        labels=np.vstack((first.labels, second.labels)),
        admissible=np.vstack((first.admissible, second.admissible)),
        features=np.vstack((first.features, second.features)),
        experiment_ids=np.concatenate((first.experiment_ids, second.experiment_ids)),
        datasets=np.concatenate((first.datasets, second.datasets)),
        file_hashes={**first.file_hashes, **second.file_hashes},
        source_bytes=first.source_bytes + second.source_bytes,
        exclusions={
            key: first.exclusions.get(key, 0) + second.exclusions.get(key, 0)
            for key in set(first.exclusions) | set(second.exclusions)
        },
    )


def _discover_families(
    development: RoleData,
    claim_names: tuple[str, ...],
    limits: dict[str, float],
) -> tuple[tuple[str, ...], dict[str, NonDegenerateCertificate | None]]:
    policies: dict[str, NonDegenerateCertificate | None] = {}
    for index, claim in enumerate(claim_names):
        policies[claim] = calibrate_nondegenerate_policy(
            development.scores[:, [index]],
            development.labels[:, [index]],
            development.admissible[:, [index]],
            development.experiment_ids,
            **limits,
        )
    return tuple(claim for claim in claim_names if policies[claim] is not None), policies


def _comparator_decisions(
    data: RoleData,
    *,
    selected_indices: list[int],
    threshold: float,
    confidence_threshold: float | None,
    frozen_semantic: SemanticRiskPolicy,
    frozen_unconstrained: SemanticRiskPolicy,
) -> dict[str, np.ndarray]:
    full_ones = np.ones((len(data.scores), len(frozen_semantic.claims)), dtype=bool)
    full_scores = np.zeros_like(full_ones, dtype=float)
    full_gates = np.zeros_like(full_ones, dtype=bool)
    full_scores[:, selected_indices] = data.scores
    full_gates[:, selected_indices] = data.admissible
    return {
        "abstain_all": np.zeros_like(data.admissible),
        "fixed_probability_0_5": data.scores >= 0.5,
        "evidence_contract_only": data.admissible,
        "unconstrained_ltt": authorize_with_policy(
            frozen_unconstrained, full_scores, full_ones
        )[:, selected_indices].astype(bool),
        "semantic_ltt_without_activation_floor": authorize_with_policy(
            frozen_semantic, full_scores, full_gates
        )[:, selected_indices].astype(bool),
        "confidence_only_target_calibrated": (
            data.scores >= confidence_threshold
            if confidence_threshold is not None
            else np.zeros_like(data.admissible)
        ),
        "semantic_ltt_nondegenerate_v4_1": data.admissible & (data.scores >= threshold),
    }


def _evaluate_decisions(
    decisions: dict[str, np.ndarray],
    data: RoleData,
    limits: dict[str, float],
) -> dict[str, Any]:
    return {
        name: _certificate_for_decisions(value, data, limits, threshold=-1.0).as_dict()
        for name, value in decisions.items()
    }


def run(
    *,
    protocol_path: Path = DEFAULT_PROTOCOL,
    parent_protocol_path: Path = DEFAULT_PARENT_PROTOCOL,
    population_path: Path = DEFAULT_POPULATION,
    root: Path = DEFAULT_ROOT,
    score_model_path: Path = DEFAULT_SCORE_MODEL,
    risk_policy_path: Path = DEFAULT_RISK_POLICY,
    output: Path = DEFAULT_OUTPUT,
    markdown: Path = DEFAULT_MARKDOWN,
) -> Path:
    started = time.perf_counter()
    lock = yaml.safe_load(protocol_path.read_text())
    parent = yaml.safe_load(parent_protocol_path.read_text())
    population = yaml.safe_load(population_path.read_text())
    model = json.loads(score_model_path.read_text())
    frozen = json.loads(risk_policy_path.read_text())
    claims = tuple(str(value) for value in frozen["variable_claims"])
    limits = _limits(parent)
    common = {
        "population": population,
        "protocol": parent,
        "root": root,
        "score_model": model,
        "variable_claims": claims,
    }

    calibration = _role_data("target_calibration", **common)
    risk_lock = _role_data("risk_lock", **common)
    development = _combine(calibration, risk_lock)
    discovered, family_policies = _discover_families(development, claims, limits)
    expected = tuple(lock["claim_family_selection"]["selected_families"])
    if discovered != expected:
        raise RuntimeError(
            f"frozen family selection is not reproducible: {discovered} != {expected}"
        )
    indices = [claims.index(claim) for claim in expected]
    selected_development = _subset(development, indices)
    joint = calibrate_nondegenerate_policy(
        selected_development.scores,
        selected_development.labels,
        selected_development.admissible,
        selected_development.experiment_ids,
        **limits,
    )
    if joint is None:
        raise RuntimeError("frozen joint development policy is no longer certifiable")
    expected_threshold = float(lock["frozen_policy"]["score_threshold"])
    if not np.isclose(joint.threshold, expected_threshold, rtol=0.0, atol=1e-12):
        raise RuntimeError(
            f"frozen threshold is not reproducible: {joint.threshold} != {expected_threshold}"
        )

    confidence = calibrate_nondegenerate_policy(
        selected_development.scores,
        selected_development.labels,
        np.ones_like(selected_development.admissible),
        selected_development.experiment_ids,
        **limits,
    )

    # This is the first point at which final CSV values are read.
    final_full = _role_data("final_evaluation", **common)
    final = _subset(final_full, indices)
    frozen_semantic = SemanticRiskPolicy.from_dict(frozen["semantic_policy"])
    frozen_unconstrained = SemanticRiskPolicy.from_dict(frozen["unconstrained_policy"])
    decisions = _comparator_decisions(
        final,
        selected_indices=indices,
        threshold=expected_threshold,
        confidence_threshold=confidence.threshold if confidence else None,
        frozen_semantic=frozen_semantic,
        frozen_unconstrained=frozen_unconstrained,
    )
    comparators = _evaluate_decisions(decisions, final, limits)
    primary = comparators["semantic_ltt_nondegenerate_v4_1"]
    shift = diagnose_shift(
        _experiment_feature_means(development),
        _experiment_feature_means(final),
        feature_names=tuple(model["feature_names"]),
    )
    by_dataset = {}
    for dataset in sorted(set(final.datasets)):
        selected = final.datasets == dataset
        dataset_data = RoleData(
            role="final_dataset_sensitivity",
            records=(),
            scores=final.scores[selected],
            labels=final.labels[selected],
            admissible=final.admissible[selected],
            features=final.features[selected],
            experiment_ids=final.experiment_ids[selected],
            datasets=final.datasets[selected],
            file_hashes={},
            source_bytes=0,
            exclusions={},
        )
        by_dataset[dataset] = _certificate_for_decisions(
            decisions["semantic_ltt_nondegenerate_v4_1"][selected],
            dataset_data,
            limits,
            threshold=expected_threshold,
        ).as_dict()

    elapsed = time.perf_counter() - started
    total_candidates = final.scores.size
    authorizations = int(primary["authorizations"])
    result = {
        "version": __version__,
        "git_revision": code_revision(),
        "analysis": "causal_chambers_v4_1_locked_final_evaluation",
        "protocol": str(protocol_path),
        "parent_protocol": str(parent_protocol_path),
        "final_values_accessed_only_after_v4_1_lock": True,
        "selected_claim_families": list(expected),
        "excluded_claim_families": lock["claim_family_selection"]["excluded_families"],
        "family_selection_data_dependent": True,
        "development": {
            "experiments": len(np.unique(development.experiment_ids)),
            "family_policies": {
                claim: policy.as_dict() if policy else None
                for claim, policy in family_policies.items()
            },
            "joint_policy": joint.as_dict(),
            "confidence_only_policy": confidence.as_dict() if confidence else None,
        },
        "final": {
            "experiments": len(np.unique(final.experiment_ids)),
            "pair_records": len(final.records),
            "datasets": sorted(set(final.datasets)),
            "source_bytes": final.source_bytes,
            "file_hashes": final.file_hashes,
            "exclusions": final.exclusions,
            "comparators": comparators,
            "primary": primary,
            "by_dataset_sensitivity": by_dataset,
            "shift_diagnostic": shift,
            "failure_taxonomy": _failure_taxonomy(
                final,
                threshold=expected_threshold,
                calibration_failure=None,
                shift_warning=bool(shift["warning"]),
            ),
        },
        "inferential_conditions": {
            "pair_rows_are_independent": False,
            "exact_interval_conditional_on_experiment_exchangeability": True,
            "dataset_stratified_results_are_descriptive_sensitivity": True,
        },
        "efficiency": {
            "wall_time_seconds": elapsed,
            "peak_rss_megabytes": _peak_rss_mb(),
            "source_bytes_read": development.source_bytes + final.source_bytes,
            "final_claim_candidates": total_candidates,
            "final_authorizations": authorizations,
            "final_abstentions": total_candidates - authorizations,
            "final_abstention_rate": (total_candidates - authorizations) / total_candidates,
            "claim_candidates_per_second": total_candidates / elapsed,
        },
        "decision": (
            "v4_1_bounded_predictive_external_confirmation_passed"
            if primary["certified"]
            else "v4_1_bounded_predictive_external_confirmation_failed"
        ),
        "stronger_external_claims_supported": False,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    _write_markdown(result, markdown)
    return output


def _write_markdown(payload: dict[str, Any], path: Path) -> None:
    primary = payload["final"]["primary"]
    lines = [
        "# Causal Chambers v4.1 locked final evaluation",
        "",
        f"- Decision: `{payload['decision']}`",
        f"- Selected claims: `{', '.join(payload['selected_claim_families'])}`",
        f"- Final experiments: `{payload['final']['experiments']}`",
        f"- Risk upper bound: `{primary['risk_upper_bound']:.4f}`",
        f"- Coverage lower bound: `{primary['coverage_lower_bound']:.4f}`",
        f"- Positive-recovery lower bound: `{primary['positive_recovery_lower_bound']:.4f}`",
        f"- Semantic violations: `{primary['semantic_violations']}`",
        "",
        "## Comparator results",
        "",
        "| Policy | Risk UCB | Coverage LCB | Recovery LCB | Certified |",
        "|---|---:|---:|---:|---:|",
    ]
    lines.extend(
        f"| `{name}` | {row['risk_upper_bound']:.4f} | "
        f"{row['coverage_lower_bound']:.4f} | {row['positive_recovery_lower_bound']:.4f} | "
        f"{str(row['certified']).lower()} |"
        for name, row in payload["final"]["comparators"].items()
    )
    lines.extend(
        (
            "",
            (
                "The result is bounded to predictive and internal-reproduction claims. "
                "The final data do not confirm topology, direction, mechanism, or causality."
            ),
            "",
        )
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--parent-protocol", type=Path, default=DEFAULT_PARENT_PROTOCOL)
    parser.add_argument("--population", type=Path, default=DEFAULT_POPULATION)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--score-model", type=Path, default=DEFAULT_SCORE_MODEL)
    parser.add_argument("--risk-policy", type=Path, default=DEFAULT_RISK_POLICY)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--markdown", type=Path, default=DEFAULT_MARKDOWN)
    args = parser.parse_args()
    path = run(
        protocol_path=args.protocol,
        parent_protocol_path=args.parent_protocol,
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
