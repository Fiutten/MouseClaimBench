"""Prospective hierarchical confirmation on the official TimeGraph generators.

The source files and seed ranges are frozen in the v5 protocol. Two compatibility
lines are removed before execution: Colab shell magic and an unused Tigramite
plotting import. No numerical equation or generator state update is replaced.

Inference is performed at the seed-bundle level. Generator scenarios, directed
pairs, and claim candidates are dependent lower-level observations and are
reported descriptively only.
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
from types import MappingProxyType
from typing import Any

import numpy as np
import yaml

from mousebrainbench import __version__
from mousebrainbench.artifacts import code_revision
from mousebrainbench.benchmarks.causal_chambers_transport import (
    _pair_record,
    direct_and_control_pairs,
)
from mousebrainbench.benchmarks.hybrid_selective_policy import (
    predict_probabilities,
    semantic_admissibility_matrix,
)
from mousebrainbench.knowledge import load_default_profile
from mousebrainbench.validation.hierarchical_risk_control import (
    HierarchicalCertificate,
    calibrate_hierarchical_policy,
    evaluate_hierarchical_decisions,
    evaluate_hierarchical_policy,
)
from mousebrainbench.validation.semantic_risk_control import (
    SemanticRiskPolicy,
    authorize_with_policy,
)
from mousebrainbench.validation.shift_diagnostics import diagnose_shift

DEFAULT_PROTOCOL = Path("configs/benchmarks/semantic_risk_control_v5.yaml")
DEFAULT_SOURCE = Path("data/external/timegraph_v5")
DEFAULT_SCORE_MODEL = Path("results/hybrid_selective_policy/model.json")
DEFAULT_RISK_POLICY = Path("results/semantic_risk_policy/model.json")
DEFAULT_OUTPUT = Path("results/timegraph_v5_confirmation/summary.json")
DEFAULT_MARKDOWN = Path("results/timegraph_v5_confirmation/summary.md")


@dataclass(frozen=True)
class RoleData:
    """Generated claims and their declared hierarchy for one locked role."""

    role: str
    records: tuple[dict[str, Any], ...]
    scores: np.ndarray
    labels: np.ndarray
    admissible: np.ndarray
    features: np.ndarray
    top_level_ids: np.ndarray
    subgroup_ids: np.ndarray
    strata: np.ndarray
    generated_scenarios: int
    generated_values: int
    aggregate_data_hash: str


def _sha256(path: Path) -> str:
    return f"sha256:{hashlib.sha256(path.read_bytes()).hexdigest()}"


def _load_official_source(
    path: Path,
    *,
    expected_hash: str,
) -> MappingProxyType[str, Any]:
    """Execute hash-verified TimeGraph source after non-numerical sanitization."""

    observed = _sha256(path)
    if observed != expected_hash:
        raise RuntimeError(f"source hash mismatch for {path}: {observed}")
    kept: list[str] = []
    removed: list[str] = []
    for line in path.read_text().splitlines():
        stripped = line.strip()
        if line.startswith('if __name__ == "__main__":'):
            removed.append("main_and_notebook_export_block")
            break
        if stripped.startswith("!pip install"):
            removed.append("colab_shell_magic")
            continue
        if stripped == "from tigramite import plotting as tp":
            removed.append("unused_tigramite_plotting_import")
            continue
        if stripped == "import matplotlib.pyplot as plt":
            removed.append("unused_matplotlib_plotting_import")
            continue
        kept.append(line)
    expected = [
        "colab_shell_magic",
        "main_and_notebook_export_block",
        "unused_matplotlib_plotting_import",
        "unused_tigramite_plotting_import",
    ]
    if sorted(removed) != expected:
        raise RuntimeError(f"unexpected TimeGraph compatibility transformation: {removed}")
    namespace: dict[str, Any] = {"__name__": f"timegraph_{path.stem}_v5"}
    # Execution is limited to the exact, hash-verified upstream source frozen above.
    exec(compile("\n".join(kept), str(path), "exec"), namespace)  # noqa: S102
    required = {"LinearTimeSeriesGenerator", "get_linear_equations"}
    if not required.issubset(namespace):
        raise RuntimeError(f"official generator API missing from {path}")
    return MappingProxyType(namespace)


def _derived_seed(seed: int, generator: str, noise: str, max_lag: int) -> int:
    digest = hashlib.sha256(
        f"mouseclaimbench-v5:{seed}:{generator}:{noise}:{max_lag}".encode()
    ).hexdigest()
    return int(digest[:8], 16)


def _observed_edges(module: MappingProxyType[str, Any], n_vars: int, max_lag: int) -> set[tuple[str, str]]:
    links = module["extract_linear_links"](module["get_linear_equations"](n_vars, max_lag))
    return {
        (str(source), str(target))
        for source, _lag, target in links
        if str(source).startswith("X") and str(target).startswith("X")
    }


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


def _role_seeds(protocol: dict[str, Any], role: str) -> range:
    spec = protocol["confirmatory_population"]["split_seeds"][role]
    start = int(spec["start"])
    return range(start, start + int(spec["count"]))


def _role_data(
    role: str,
    *,
    protocol: dict[str, Any],
    source_root: Path,
    score_model: dict[str, Any],
    variable_claims: tuple[str, ...],
) -> RoleData:
    population = protocol["confirmatory_population"]
    source_files = population["source_files"]
    modules = {
        generator: _load_official_source(
            source_root / f"Codes/{generator}.py",
            expected_hash=str(source_files[f"Codes/{generator}.py"]),
        )
        for generator in population["factors"]["generator"]
    }
    factors = population["factors"]
    n_vars = int(factors["n_vars"])
    n_points = int(factors["n_points"])
    noise_scale = float(factors["noise_scale"])
    if role == "ood_stress":
        stress = population["ood_stress_factors"]
        n_points = int(stress["n_points"])
        noise_scale = float(stress["noise_scale"])
    claim_names = tuple(str(value) for value in score_model["claim_names"])
    records: list[dict[str, Any]] = []
    digest = hashlib.sha256()
    scenarios = values_generated = 0
    pair_spec = population["pair_sampling"]
    for seed in _role_seeds(protocol, role):
        top_id = f"{role}/seed-{seed}"
        for generator in factors["generator"]:
            module = modules[str(generator)]
            for noise in factors["noise"]:
                for max_lag_value in factors["max_lag"]:
                    max_lag = int(max_lag_value)
                    stratum = f"{generator}/{noise}/lag-{max_lag}"
                    subgroup = f"{top_id}/{stratum}"
                    random_state = _derived_seed(seed, str(generator), str(noise), max_lag)
                    instance = module["LinearTimeSeriesGenerator"](
                        noise_type=str(noise),
                        noise_scale=noise_scale,
                        df=int(factors["student_t_df"]),
                        random_state=random_state,
                    )
                    frame = instance.generate_multivariate_ts(n_points, n_vars, max_lag)
                    observed = tuple(f"X{index}" for index in range(1, n_vars + 1))
                    numeric = np.ascontiguousarray(frame.loc[:, observed].to_numpy(dtype="<f8"))
                    if not np.all(np.isfinite(numeric)):
                        raise RuntimeError(f"non-finite official output in {subgroup}")
                    digest.update(subgroup.encode())
                    digest.update(numeric.tobytes())
                    scenarios += 1
                    values_generated += numeric.size
                    edges = _observed_edges(module, n_vars, max_lag)
                    available = direct_and_control_pairs(observed, edges, namespace=subgroup)
                    selected = _select_pairs(
                        available,
                        namespace=f"mouseclaimbench-v5:{subgroup}",
                        direct_count=int(pair_spec["direct_pairs_per_scenario"]),
                        control_count=int(pair_spec["control_pairs_per_scenario"]),
                    )
                    if len(selected) != 2:
                        raise RuntimeError(f"pair contract failed in {subgroup}")
                    for source, target, direct_edge in selected:
                        record = _pair_record(
                            frame,
                            dataset="timegraph-v5",
                            chamber=str(generator),
                            experiment=subgroup,
                            source=source,
                            target=target,
                            direct_edge=direct_edge,
                            eligible=observed,
                            claim_names=claim_names,
                        )
                        if record is None:
                            raise RuntimeError(f"pair adapter failed in {subgroup}")
                        records.append(
                            {
                                **record,
                                "top_level_id": top_id,
                                "subgroup_id": subgroup,
                                "stratum": stratum,
                                "generator_seed": random_state,
                            }
                        )
    features = np.vstack([row["features"] for row in records])
    labels = np.vstack([row["labels"] for row in records]).astype(bool)
    probabilities = predict_probabilities(score_model["model_sets"]["full"], features, claim_names)
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
        top_level_ids=np.asarray([row["top_level_id"] for row in records]),
        subgroup_ids=np.asarray([row["subgroup_id"] for row in records]),
        strata=np.asarray([row["stratum"] for row in records]),
        generated_scenarios=scenarios,
        generated_values=values_generated,
        aggregate_data_hash=f"sha256:{digest.hexdigest()}",
    )


def _limits(protocol: dict[str, Any]) -> dict[str, float | int]:
    contract = protocol["confirmatory_population"]["inferential_contract"]
    return {
        "target_risk": float(contract["target_seed_bundle_failure_probability"]),
        "minimum_coverage": float(contract["minimum_authorized_seed_bundle_coverage"]),
        "minimum_positive_recovery": float(contract["minimum_positive_recovery"]),
        "minimum_independent_units": 29,
        "confidence": float(contract["confidence_level"]),
    }


def _evaluate_decisions(
    decisions: np.ndarray,
    data: RoleData,
    limits: dict[str, float | int],
    *,
    threshold: float,
) -> HierarchicalCertificate:
    return evaluate_hierarchical_decisions(
        decisions,
        data.labels,
        data.admissible,
        data.top_level_ids,
        data.subgroup_ids,
        data.strata,
        threshold=threshold,
        **limits,
    )


def _comparators(
    data: RoleData,
    *,
    threshold: float | None,
    confidence_threshold: float | None,
    frozen_policy: SemanticRiskPolicy,
    limits: dict[str, float | int],
) -> dict[str, Any]:
    zeros = np.zeros_like(data.admissible, dtype=bool)
    decisions = {
        "abstain_all": zeros,
        "fixed_probability_0_5": data.scores >= 0.5,
        "confidence_only_target_calibrated": (
            data.scores >= confidence_threshold if confidence_threshold is not None else zeros
        ),
        "evidence_contract_only": data.admissible,
        "frozen_v3_semantic_ltt": authorize_with_policy(
            frozen_policy, data.scores, data.admissible
        ).astype(bool),
        "v5_hierarchical_nondegenerate": (
            data.admissible & (data.scores >= threshold) if threshold is not None else zeros
        ),
    }
    thresholds = {
        "abstain_all": 2.0,
        "fixed_probability_0_5": 0.5,
        "confidence_only_target_calibrated": confidence_threshold or -1.0,
        "evidence_contract_only": 0.0,
        "frozen_v3_semantic_ltt": -1.0,
        "v5_hierarchical_nondegenerate": threshold or -1.0,
    }
    return {
        name: _evaluate_decisions(value, data, limits, threshold=thresholds[name]).as_dict()
        for name, value in decisions.items()
    }


def _top_feature_means(data: RoleData) -> np.ndarray:
    return np.vstack(
        [
            data.features[data.top_level_ids == top].mean(axis=0)
            for top in np.unique(data.top_level_ids)
        ]
    )


def _leave_one_stratum_out(
    data: RoleData,
    *,
    threshold: float,
    limits: dict[str, float | int],
) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for excluded in sorted(np.unique(data.strata)):
        keep = data.strata != excluded
        result = evaluate_hierarchical_policy(
            data.scores[keep],
            data.labels[keep],
            data.admissible[keep],
            data.top_level_ids[keep],
            data.subgroup_ids[keep],
            data.strata[keep],
            threshold=threshold,
            **limits,
        )
        output[str(excluded)] = {
            "certified": result.certified,
            "risk_upper_bound": result.certificate.risk_upper_bound,
            "coverage_lower_bound": result.certificate.coverage_lower_bound,
            "positive_recovery_lower_bound": result.certificate.positive_recovery_lower_bound,
        }
    return output


def _role_manifest(data: RoleData) -> dict[str, Any]:
    return {
        "seed_bundles": len(np.unique(data.top_level_ids)),
        "generator_scenarios": data.generated_scenarios,
        "pair_records": len(data.records),
        "claim_candidates": int(data.scores.size),
        "generated_values": data.generated_values,
        "aggregate_data_hash": data.aggregate_data_hash,
        "strata": sorted(np.unique(data.strata).tolist()),
    }


def _peak_rss_mb() -> float:
    value = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return value / (1024.0 * 1024.0) if sys.platform == "darwin" else value / 1024.0


def run(
    *,
    protocol_path: Path = DEFAULT_PROTOCOL,
    source_root: Path = DEFAULT_SOURCE,
    score_model_path: Path = DEFAULT_SCORE_MODEL,
    risk_policy_path: Path = DEFAULT_RISK_POLICY,
    output: Path = DEFAULT_OUTPUT,
    markdown: Path = DEFAULT_MARKDOWN,
) -> Path:
    """Run calibration, locked confirmation, and diagnostic OOD stress."""

    started = time.perf_counter()
    protocol = yaml.safe_load(protocol_path.read_text())
    if protocol["status"] != "frozen_before_v5_outcome_generation":
        raise ValueError("v5 protocol is not frozen")
    score_model = json.loads(score_model_path.read_text())
    frozen = json.loads(risk_policy_path.read_text())
    variable_claims = tuple(str(value) for value in frozen["variable_claims"])
    frozen_policy = SemanticRiskPolicy.from_dict(frozen["semantic_policy"])
    limits = _limits(protocol)

    calibration = _role_data(
        "target_calibration",
        protocol=protocol,
        source_root=source_root,
        score_model=score_model,
        variable_claims=variable_claims,
    )
    complete = calibrate_hierarchical_policy(
        calibration.scores,
        calibration.labels,
        calibration.admissible,
        calibration.top_level_ids,
        calibration.subgroup_ids,
        calibration.strata,
        **limits,
    )
    confidence_only = calibrate_hierarchical_policy(
        calibration.scores,
        calibration.labels,
        np.ones_like(calibration.admissible),
        calibration.top_level_ids,
        calibration.subgroup_ids,
        calibration.strata,
        **limits,
    )
    threshold = complete.certificate.threshold if complete is not None else None
    confidence_threshold = (
        confidence_only.certificate.threshold if confidence_only is not None else None
    )

    risk_lock = _role_data(
        "risk_lock",
        protocol=protocol,
        source_root=source_root,
        score_model=score_model,
        variable_claims=variable_claims,
    )
    risk_comparators = _comparators(
        risk_lock,
        threshold=threshold,
        confidence_threshold=confidence_threshold,
        frozen_policy=frozen_policy,
        limits=limits,
    )
    risk_primary = risk_comparators["v5_hierarchical_nondegenerate"]
    risk_passed = complete is not None and bool(risk_primary["certified"])

    final_payload: dict[str, Any] = {"opened": False, "reason": "risk_lock_did_not_pass"}
    final_data: RoleData | None = None
    if risk_passed:
        final_data = _role_data(
            "final",
            protocol=protocol,
            source_root=source_root,
            score_model=score_model,
            variable_claims=variable_claims,
        )
        final_comparators = _comparators(
            final_data,
            threshold=threshold,
            confidence_threshold=confidence_threshold,
            frozen_policy=frozen_policy,
            limits=limits,
        )
        final_primary = final_comparators["v5_hierarchical_nondegenerate"]
        final_payload = {
            "opened": True,
            **_role_manifest(final_data),
            "comparators": final_comparators,
            "shift_from_calibration": diagnose_shift(
                _top_feature_means(calibration),
                _top_feature_means(final_data),
                feature_names=tuple(score_model["feature_names"]),
            ),
            "leave_one_stratum_out": _leave_one_stratum_out(
                final_data, threshold=float(threshold), limits=limits
            ),
            "externally_supported_within_declared_synthetic_population": bool(
                final_primary["certified"]
            ),
        }

    stress = _role_data(
        "ood_stress",
        protocol=protocol,
        source_root=source_root,
        score_model=score_model,
        variable_claims=variable_claims,
    )
    stress_comparators = _comparators(
        stress,
        threshold=threshold,
        confidence_threshold=confidence_threshold,
        frozen_policy=frozen_policy,
        limits=limits,
    )
    elapsed = time.perf_counter() - started
    final_supported = final_payload.get(
        "externally_supported_within_declared_synthetic_population"
    ) is True
    result = {
        "version": __version__,
        "git_revision": code_revision(),
        "analysis": "timegraph_v5_prospective_hierarchical_confirmation",
        "protocol": str(protocol_path),
        "protocol_status": protocol["status"],
        "source_revision": protocol["confirmatory_population"]["source_revision"],
        "source_hashes": protocol["confirmatory_population"]["source_files"],
        "source_transformations": protocol["confirmatory_population"][
            "source_transformations"
        ],
        "inferential_unit": "seed_bundle",
        "lower_level_inference_permitted": False,
        "variable_claims": list(variable_claims),
        "target_limits": limits,
        "calibration": {
            **_role_manifest(calibration),
            "complete_policy": complete.as_dict() if complete else None,
            "confidence_only_policy": confidence_only.as_dict() if confidence_only else None,
        },
        "risk_lock": {
            **_role_manifest(risk_lock),
            "comparators": risk_comparators,
            "shift_from_calibration": diagnose_shift(
                _top_feature_means(calibration),
                _top_feature_means(risk_lock),
                feature_names=tuple(score_model["feature_names"]),
            ),
            "passed": risk_passed,
        },
        "final_evaluation": final_payload,
        "ood_stress": {
            **_role_manifest(stress),
            "role": "diagnostic_only_not_part_of_the_confirmatory_certificate",
            "comparators": stress_comparators,
            "shift_from_calibration": diagnose_shift(
                _top_feature_means(calibration),
                _top_feature_means(stress),
                feature_names=tuple(score_model["feature_names"]),
            ),
        },
        "efficiency": {
            "wall_time_seconds": elapsed,
            "peak_rss_megabytes": _peak_rss_mb(),
            "generated_scenarios": (
                calibration.generated_scenarios
                + risk_lock.generated_scenarios
                + stress.generated_scenarios
                + (final_data.generated_scenarios if final_data else 0)
            ),
        },
        "decision": (
            "v5_declared_synthetic_population_supported"
            if final_supported
            else "v5_declared_synthetic_population_not_supported"
        ),
        "claim_boundary": (
            "A positive result is limited to the frozen TimeGraph generator mixture. "
            "It is not a real-domain, biological, or distribution-free guarantee."
        ),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    _write_markdown(result, markdown)
    return output


def _write_markdown(payload: dict[str, Any], path: Path) -> None:
    risk = payload["risk_lock"]["comparators"]["v5_hierarchical_nondegenerate"]
    final = payload["final_evaluation"]
    lines = [
        "# TimeGraph v5 prospective hierarchical confirmation",
        "",
        f"- Decision: `{payload['decision']}`",
        f"- Inferential unit: `{payload['inferential_unit']}`",
        f"- Risk lock passed: `{str(payload['risk_lock']['passed']).lower()}`",
        f"- Risk-lock risk upper bound: `{risk['risk_upper_bound']:.6f}`",
        f"- Risk-lock coverage lower bound: `{risk['coverage_lower_bound']:.6f}`",
        f"- Final opened: `{str(final['opened']).lower()}`",
        "",
        "## Comparator status on risk lock",
        "",
        "| Comparator | Certified | Risk UCB | Coverage LCB | Recovery LCB |",
        "|---|---:|---:|---:|---:|",
    ]
    for name, row in payload["risk_lock"]["comparators"].items():
        lines.append(
            f"| `{name}` | {str(row['certified']).lower()} | "
            f"{row['risk_upper_bound']:.4f} | {row['coverage_lower_bound']:.4f} | "
            f"{row['positive_recovery_lower_bound']:.4f} |"
        )
    lines.extend(("", "## Interpretation", "", payload["claim_boundary"], ""))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--score-model", type=Path, default=DEFAULT_SCORE_MODEL)
    parser.add_argument("--risk-policy", type=Path, default=DEFAULT_RISK_POLICY)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--markdown", type=Path, default=DEFAULT_MARKDOWN)
    args = parser.parse_args()
    print(
        run(
            protocol_path=args.protocol,
            source_root=args.source_root,
            score_model_path=args.score_model,
            risk_policy_path=args.risk_policy,
            output=args.output,
            markdown=args.markdown,
        ).resolve()
    )


if __name__ == "__main__":
    main()
