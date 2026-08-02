"""Independent confirmation of the v5-selected topology-specific authorizer."""

from __future__ import annotations

import argparse
import json
import resource
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from mousebrainbench import __version__
from mousebrainbench.artifacts import code_revision
from mousebrainbench.benchmarks.timegraph_v5_confirmation import (
    RoleData,
    _leave_one_stratum_out,
    _role_data,
    _role_manifest,
    _top_feature_means,
)
from mousebrainbench.validation.hierarchical_risk_control import (
    evaluate_hierarchical_decisions,
)
from mousebrainbench.validation.shift_diagnostics import diagnose_shift

DEFAULT_PROTOCOL = Path("configs/benchmarks/semantic_risk_control_v5_1.yaml")
DEFAULT_SOURCE = Path("data/external/timegraph_v5")
DEFAULT_SCORE_MODEL = Path("results/hybrid_selective_policy/model.json")
DEFAULT_RISK_POLICY = Path("results/semantic_risk_policy/model.json")
DEFAULT_OUTPUT = Path("results/timegraph_v5_1_topology_confirmation/summary.json")
DEFAULT_MARKDOWN = Path("results/timegraph_v5_1_topology_confirmation/summary.md")


def _limits(protocol: dict[str, Any]) -> dict[str, float | int]:
    contract = protocol["confirmatory_population"]["inferential_contract"]
    return {
        "target_risk": float(contract["target_seed_bundle_failure_probability"]),
        "minimum_coverage": float(contract["minimum_authorized_seed_bundle_coverage"]),
        "minimum_positive_recovery": float(contract["minimum_positive_recovery"]),
        "minimum_independent_units": int(contract["minimum_independent_units"]),
        "confidence": float(contract["confidence_level"]),
    }


def _frozen_v3_threshold(risk_policy: dict[str, Any], claim: str) -> float:
    for certificate in risk_policy["semantic_policy"]["certificates"]:
        if certificate["claim"] == claim:
            value = certificate["threshold"]
            if value is None:
                raise RuntimeError(f"frozen v3 claim {claim} has no threshold")
            return float(value)
    raise KeyError(f"claim {claim} is absent from the frozen v3 policy")


def _certificate(
    decisions: np.ndarray,
    data: RoleData,
    limits: dict[str, float | int],
    *,
    threshold: float,
) -> dict[str, Any]:
    return evaluate_hierarchical_decisions(
        decisions,
        data.labels,
        data.admissible,
        data.top_level_ids,
        data.subgroup_ids,
        data.strata,
        threshold=threshold,
        **limits,
    ).as_dict()


def _comparators(
    data: RoleData,
    *,
    fixed_threshold: float,
    frozen_v3_threshold: float,
    limits: dict[str, float | int],
) -> dict[str, Any]:
    decisions = {
        "abstain_all": np.zeros_like(data.admissible, dtype=bool),
        "fixed_probability_0_5": data.scores >= 0.5,
        "evidence_contract_only": data.admissible,
        "frozen_v3_topology_threshold": data.admissible
        & (data.scores >= frozen_v3_threshold),
        "v5_1_fixed_hierarchical_threshold": data.admissible
        & (data.scores >= fixed_threshold),
    }
    thresholds = {
        "abstain_all": 2.0,
        "fixed_probability_0_5": 0.5,
        "evidence_contract_only": 0.0,
        "frozen_v3_topology_threshold": frozen_v3_threshold,
        "v5_1_fixed_hierarchical_threshold": fixed_threshold,
    }
    return {
        name: _certificate(value, data, limits, threshold=thresholds[name])
        for name, value in decisions.items()
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
    """Apply the fixed topology threshold to two new locked seed populations."""

    started = time.perf_counter()
    protocol = yaml.safe_load(protocol_path.read_text())
    expected_status = "frozen_after_v5_development_before_v5_1_outcome_generation"
    if protocol["status"] != expected_status:
        raise ValueError("v5.1 protocol is not outcome-frozen")
    score_model = json.loads(score_model_path.read_text())
    risk_policy = json.loads(risk_policy_path.read_text())
    disclosure = protocol["post_development_selection_disclosure"]
    claim = str(disclosure["selected_claim"])
    threshold = float(disclosure["selected_threshold"])
    v3_threshold = _frozen_v3_threshold(risk_policy, claim)
    limits = _limits(protocol)

    risk_lock = _role_data(
        "risk_lock",
        protocol=protocol,
        source_root=source_root,
        score_model=score_model,
        variable_claims=(claim,),
    )
    risk_comparators = _comparators(
        risk_lock,
        fixed_threshold=threshold,
        frozen_v3_threshold=v3_threshold,
        limits=limits,
    )
    risk_primary = risk_comparators["v5_1_fixed_hierarchical_threshold"]
    risk_passed = bool(risk_primary["certified"])

    final_data: RoleData | None = None
    final_payload: dict[str, Any] = {"opened": False, "reason": "risk_lock_did_not_pass"}
    if risk_passed:
        final_data = _role_data(
            "final",
            protocol=protocol,
            source_root=source_root,
            score_model=score_model,
            variable_claims=(claim,),
        )
        final_comparators = _comparators(
            final_data,
            fixed_threshold=threshold,
            frozen_v3_threshold=v3_threshold,
            limits=limits,
        )
        primary = final_comparators["v5_1_fixed_hierarchical_threshold"]
        final_payload = {
            "opened": True,
            **_role_manifest(final_data),
            "comparators": final_comparators,
            "shift_from_risk_lock": diagnose_shift(
                _top_feature_means(risk_lock),
                _top_feature_means(final_data),
                feature_names=tuple(score_model["feature_names"]),
            ),
            "leave_one_stratum_out": _leave_one_stratum_out(
                final_data, threshold=threshold, limits=limits
            ),
            "supported": bool(primary["certified"]),
        }

    stress = _role_data(
        "ood_stress",
        protocol=protocol,
        source_root=source_root,
        score_model=score_model,
        variable_claims=(claim,),
    )
    stress_comparators = _comparators(
        stress,
        fixed_threshold=threshold,
        frozen_v3_threshold=v3_threshold,
        limits=limits,
    )
    final_supported = final_payload.get("supported") is True
    elapsed = time.perf_counter() - started
    result = {
        "version": __version__,
        "git_revision": code_revision(),
        "analysis": "timegraph_v5_1_topology_specific_hierarchical_confirmation",
        "protocol": str(protocol_path),
        "protocol_status": protocol["status"],
        "post_development_selection_disclosed": bool(
            disclosure["selected_after_observing_v5_calibration"]
        ),
        "claim": claim,
        "fixed_threshold": threshold,
        "threshold_refitted": False,
        "inferential_unit": "seed_bundle",
        "lower_level_inference_permitted": False,
        "target_limits": limits,
        "risk_lock": {
            **_role_manifest(risk_lock),
            "comparators": risk_comparators,
            "passed": risk_passed,
        },
        "final_evaluation": final_payload,
        "ood_stress": {
            **_role_manifest(stress),
            "role": "diagnostic_only_not_part_of_the_confirmatory_certificate",
            "comparators": stress_comparators,
            "shift_from_risk_lock": diagnose_shift(
                _top_feature_means(risk_lock),
                _top_feature_means(stress),
                feature_names=tuple(score_model["feature_names"]),
            ),
        },
        "efficiency": {
            "wall_time_seconds": elapsed,
            "peak_rss_megabytes": _peak_rss_mb(),
            "generated_scenarios": (
                risk_lock.generated_scenarios
                + stress.generated_scenarios
                + (final_data.generated_scenarios if final_data else 0)
            ),
        },
        "decision": (
            "v5_1_topology_specific_contract_supported"
            if final_supported
            else "v5_1_topology_specific_contract_not_supported"
        ),
        "claim_boundary": (
            "The result concerns only topology-specific authorizations in the "
            "frozen TimeGraph mixture. It does not repair the failed global v5 policy."
        ),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    _write_markdown(result, markdown)
    return output


def _write_markdown(payload: dict[str, Any], path: Path) -> None:
    risk_rows = payload["risk_lock"]["comparators"]
    primary = risk_rows["v5_1_fixed_hierarchical_threshold"]
    lines = [
        "# TimeGraph v5.1 topology-specific confirmation",
        "",
        f"- Decision: `{payload['decision']}`",
        f"- Claim: `{payload['claim']}`",
        f"- Fixed threshold: `{payload['fixed_threshold']:.12f}`",
        f"- Risk lock passed: `{str(payload['risk_lock']['passed']).lower()}`",
        f"- Risk-lock failures: `{primary['failing_experiments']}`",
        f"- Risk-lock risk upper bound: `{primary['risk_upper_bound']:.6f}`",
        f"- Risk-lock coverage lower bound: `{primary['coverage_lower_bound']:.6f}`",
        f"- Final opened: `{str(payload['final_evaluation']['opened']).lower()}`",
        "",
        "| Comparator | Certified | Risk UCB | Coverage LCB | Recovery LCB |",
        "|---|---:|---:|---:|---:|",
    ]
    for name, row in risk_rows.items():
        lines.append(
            f"| `{name}` | {str(row['certified']).lower()} | "
            f"{row['risk_upper_bound']:.4f} | {row['coverage_lower_bound']:.4f} | "
            f"{row['positive_recovery_lower_bound']:.4f} |"
        )
    lines.extend(("", "## Boundary", "", payload["claim_boundary"], ""))
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
