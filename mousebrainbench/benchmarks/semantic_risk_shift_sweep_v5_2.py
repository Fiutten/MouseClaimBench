"""Prospective severity sweep for fixed topology-specific authorization."""

from __future__ import annotations

import argparse
import json
import resource
import sys
import time
from itertools import pairwise
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from mousebrainbench import __version__
from mousebrainbench.artifacts import code_revision
from mousebrainbench.benchmarks.timegraph_v5_confirmation import (
    RoleData,
    _role_data,
    _role_manifest,
    _top_feature_means,
)
from mousebrainbench.validation.hierarchical_risk_control import (
    evaluate_hierarchical_policy,
)
from mousebrainbench.validation.shift_diagnostics import diagnose_shift

DEFAULT_PROTOCOL = Path("configs/benchmarks/semantic_risk_shift_sweep_v5_2.yaml")
DEFAULT_SOURCE = Path("data/external/timegraph_v5")
DEFAULT_SCORE_MODEL = Path("results/hybrid_selective_policy/model.json")
DEFAULT_OUTPUT = Path("results/semantic_risk_shift_sweep_v5_2/summary.json")
DEFAULT_MARKDOWN = Path("results/semantic_risk_shift_sweep_v5_2/summary.md")


def _limits(protocol: dict[str, Any]) -> dict[str, float | int]:
    contract = protocol["simultaneous_contract"]
    return {
        "target_risk": float(contract["target_seed_bundle_failure_probability"]),
        "minimum_coverage": float(contract["minimum_authorized_seed_bundle_coverage"]),
        "minimum_positive_recovery": float(contract["minimum_positive_recovery"]),
        "minimum_independent_units": int(contract["minimum_independent_units"]),
        "confidence": float(contract["per_level_confidence"]),
    }


def _bundle_states(data: RoleData, threshold: float) -> dict[str, dict[str, bool]]:
    decisions = data.admissible & (data.scores >= threshold)
    output: dict[str, dict[str, bool]] = {}
    for top_id in np.unique(data.top_level_ids):
        selected = data.top_level_ids == top_id
        chosen = decisions[selected]
        truth = data.labels[selected]
        output[str(top_id).split("/", maxsplit=1)[-1]] = {
            "authorized": bool(np.any(chosen)),
            "failed": bool(np.any(chosen & ~truth)),
        }
    return output


def paired_bundle_transition(
    previous: dict[str, dict[str, bool]],
    current: dict[str, dict[str, bool]],
) -> dict[str, Any]:
    """Describe authorization and failure transitions for paired seeds."""

    shared = sorted(set(previous) & set(current))
    authorization = {"off_to_off": 0, "off_to_on": 0, "on_to_off": 0, "on_to_on": 0}
    failure = {"off_to_off": 0, "off_to_on": 0, "on_to_off": 0, "on_to_on": 0}
    for seed in shared:
        for key, target in (("authorized", authorization), ("failed", failure)):
            before = previous[seed][key]
            after = current[seed][key]
            target[f"{'on' if before else 'off'}_to_{'on' if after else 'off'}"] += 1
    return {
        "shared_seed_bundles": len(shared),
        "authorization": authorization,
        "failure": failure,
        "inferential_status": "paired_descriptive_only",
    }


def _peak_rss_mb() -> float:
    value = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return value / (1024.0 * 1024.0) if sys.platform == "darwin" else value / 1024.0


def run(
    *,
    protocol_path: Path = DEFAULT_PROTOCOL,
    source_root: Path = DEFAULT_SOURCE,
    score_model_path: Path = DEFAULT_SCORE_MODEL,
    output: Path = DEFAULT_OUTPUT,
    markdown: Path = DEFAULT_MARKDOWN,
) -> Path:
    """Run all frozen severity levels without adapting the claim threshold."""

    started = time.perf_counter()
    protocol = yaml.safe_load(protocol_path.read_text())
    if protocol["status"] != "frozen_before_shift_sweep_generation":
        raise ValueError("shift sweep protocol is not outcome-frozen")
    score_model = json.loads(score_model_path.read_text())
    claim = str(protocol["claim"])
    threshold = float(protocol["fixed_threshold"])
    limits = _limits(protocol)
    reference = _role_data(
        "reference",
        protocol=protocol,
        source_root=source_root,
        score_model=score_model,
        variable_claims=(claim,),
    )
    reference_features = _top_feature_means(reference)
    feature_names = tuple(score_model["feature_names"])

    roles = tuple(protocol["confirmatory_population"]["role_factor_overrides"])
    levels: dict[str, Any] = {}
    states: dict[str, dict[str, dict[str, bool]]] = {}
    generated_scenarios = reference.generated_scenarios
    for role in roles:
        data = _role_data(
            role,
            protocol=protocol,
            source_root=source_root,
            score_model=score_model,
            variable_claims=(claim,),
        )
        generated_scenarios += data.generated_scenarios
        certificate = evaluate_hierarchical_policy(
            data.scores,
            data.labels,
            data.admissible,
            data.top_level_ids,
            data.subgroup_ids,
            data.strata,
            threshold=threshold,
            **limits,
        )
        shift = diagnose_shift(
            reference_features,
            _top_feature_means(data),
            feature_names=feature_names,
            alpha=float(protocol["shift_diagnostic"]["alpha"]),
            permutations=int(protocol["shift_diagnostic"]["permutations"]),
        )
        levels[role] = {
            **_role_manifest(data),
            "factors": protocol["confirmatory_population"]["role_factor_overrides"][role],
            "certificate": certificate.as_dict(),
            "shift_diagnostic": shift,
        }
        states[role] = _bundle_states(data, threshold)

    transitions = {
        f"{previous}_to_{current}": paired_bundle_transition(states[previous], states[current])
        for previous, current in pairwise(roles)
    }
    warning_roles = [role for role in roles if levels[role]["shift_diagnostic"]["warning"]]
    failure_roles = [role for role in roles if not levels[role]["certificate"]["certified"]]
    first_warning = warning_roles[0] if warning_roles else None
    first_failure = failure_roles[0] if failure_roles else None
    warning_no_later = bool(
        first_failure is not None
        and first_warning is not None
        and roles.index(first_warning) <= roles.index(first_failure)
    )
    elapsed = time.perf_counter() - started
    result = {
        "version": __version__,
        "git_revision": code_revision(),
        "analysis": "semantic_risk_topology_shift_degradation_v5_2",
        "protocol": str(protocol_path),
        "protocol_status": protocol["status"],
        "claim": claim,
        "fixed_threshold": threshold,
        "threshold_refitted": False,
        "inferential_unit": "seed_bundle_within_level",
        "multiplicity": protocol["simultaneous_contract"]["multiplicity_method"],
        "reference": _role_manifest(reference),
        "levels": levels,
        "paired_transitions": transitions,
        "primary_endpoints": {
            "first_level_with_shift_warning": first_warning,
            "first_level_with_certificate_failure": first_failure,
            "warning_occurs_no_later_than_certificate_failure": warning_no_later,
            "detector_validated_for_general_use": False,
        },
        "efficiency": {
            "wall_time_seconds": elapsed,
            "peak_rss_megabytes": _peak_rss_mb(),
            "generated_scenarios": generated_scenarios,
        },
        "decision": (
            "warning_precedes_or_matches_observed_certificate_loss"
            if warning_no_later
            else "warning_does_not_precede_observed_certificate_loss"
        ),
        "interpretation": (
            "This sweep characterizes one frozen degradation path. It does not calibrate "
            "the warning as a universal deployment decision rule."
        ),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    _write_markdown(result, markdown)
    return output


def _write_markdown(payload: dict[str, Any], path: Path) -> None:
    endpoints = payload["primary_endpoints"]
    lines = [
        "# Semantic-risk shift degradation v5.2",
        "",
        f"- Decision: `{payload['decision']}`",
        f"- First warning: `{endpoints['first_level_with_shift_warning']}`",
        f"- First certificate failure: `{endpoints['first_level_with_certificate_failure']}`",
        "- General-purpose detector validated: `false`",
        "",
        "| Level | n | Noise | Failures | Risk UCB | Coverage LCB | Warning | Certified |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for role, level in payload["levels"].items():
        row = level["certificate"]
        lines.append(
            f"| `{role}` | {level['factors']['n_points']} | {level['factors']['noise_scale']} | "
            f"{row['failing_experiments']} | {row['risk_upper_bound']:.4f} | "
            f"{row['coverage_lower_bound']:.4f} | "
            f"{str(level['shift_diagnostic']['warning']).lower()} | "
            f"{str(row['certified']).lower()} |"
        )
    lines.extend(("", payload["interpretation"], ""))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--score-model", type=Path, default=DEFAULT_SCORE_MODEL)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--markdown", type=Path, default=DEFAULT_MARKDOWN)
    args = parser.parse_args()
    print(
        run(
            protocol_path=args.protocol,
            source_root=args.source_root,
            score_model_path=args.score_model,
            output=args.output,
            markdown=args.markdown,
        ).resolve()
    )


if __name__ == "__main__":
    main()
