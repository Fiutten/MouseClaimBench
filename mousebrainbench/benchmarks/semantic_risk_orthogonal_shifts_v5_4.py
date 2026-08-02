"""Evaluate frozen topology authorization across orthogonal TimeGraph shifts."""

from __future__ import annotations

import argparse
import json
import resource
import sys
import time
from itertools import pairwise
from pathlib import Path
from typing import Any

import yaml

from mousebrainbench import __version__
from mousebrainbench.artifacts import code_revision
from mousebrainbench.benchmarks.semantic_risk_shift_sweep_v5_2 import (
    _bundle_states,
    paired_bundle_transition,
)
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

DEFAULT_PROTOCOL = Path("configs/benchmarks/semantic_risk_orthogonal_shifts_v5_4.yaml")
DEFAULT_SOURCE = Path("data/external/timegraph_v5")
DEFAULT_SCORE_MODEL = Path("results/hybrid_selective_policy/model.json")
DEFAULT_OUTPUT = Path("results/semantic_risk_orthogonal_shifts_v5_4/summary.json")
DEFAULT_MARKDOWN = Path("results/semantic_risk_orthogonal_shifts_v5_4/summary.md")


def _limits(protocol: dict[str, Any]) -> dict[str, float | int]:
    contract = protocol["simultaneous_contract"]
    return {
        "target_risk": float(contract["target_seed_bundle_failure_probability"]),
        "minimum_coverage": float(contract["minimum_authorized_seed_bundle_coverage"]),
        "minimum_positive_recovery": float(contract["minimum_positive_recovery"]),
        "minimum_independent_units": int(contract["minimum_independent_units"]),
        "confidence": float(contract["per_level_confidence"]),
    }


def _peak_rss_mb() -> float:
    value = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return value / (1024.0 * 1024.0) if sys.platform == "darwin" else value / 1024.0


def _endpoint(certificate: dict[str, Any]) -> dict[str, Any]:
    lower = certificate["lower_level_summary"]["subgroups"]
    return {
        "failing_seed_bundles": certificate["failing_experiments"],
        "authorized_seed_bundles": certificate["authorized_experiments"],
        "missed_positive_seed_bundles": (
            certificate["eligible_positive_experiments"]
            - certificate["recovered_positive_experiments"]
        ),
        "false_authorization_subgroups_descriptive": lower["failing_units"],
        "certified": certificate["certified"],
    }


def run(
    *,
    protocol_path: Path = DEFAULT_PROTOCOL,
    source_root: Path = DEFAULT_SOURCE,
    score_model_path: Path = DEFAULT_SCORE_MODEL,
    output: Path = DEFAULT_OUTPUT,
    markdown: Path = DEFAULT_MARKDOWN,
) -> Path:
    """Generate every frozen level and evaluate fixed-threshold failure events."""

    started = time.perf_counter()
    protocol = yaml.safe_load(protocol_path.read_text())
    if protocol["status"] != "frozen_before_orthogonal_shift_generation":
        raise ValueError("orthogonal shift protocol is not outcome-frozen")
    score_model = json.loads(score_model_path.read_text())
    claim = str(protocol["claim"])
    threshold = float(protocol["fixed_threshold"])
    limits = _limits(protocol)
    feature_names = tuple(score_model["feature_names"])
    cache: dict[str, RoleData] = {}
    levels: dict[str, Any] = {}
    generated_scenarios = 0
    for family in protocol["families"].values():
        for level in family["levels"]:
            if level in cache:
                continue
            data = _role_data(
                str(level),
                protocol=protocol,
                source_root=source_root,
                score_model=score_model,
                variable_claims=(claim,),
            )
            cache[str(level)] = data
            generated_scenarios += data.generated_scenarios

    family_results: dict[str, Any] = {}
    for family_name, spec in protocol["families"].items():
        reference = cache[str(spec["reference"])]
        reference_features = _top_feature_means(reference)
        family_levels: dict[str, Any] = {}
        states: dict[str, dict[str, dict[str, bool]]] = {}
        for level in spec["levels"]:
            data = cache[str(level)]
            certificate = evaluate_hierarchical_policy(
                data.scores,
                data.labels,
                data.admissible,
                data.top_level_ids,
                data.subgroup_ids,
                data.strata,
                threshold=threshold,
                **limits,
            ).as_dict()
            shift = diagnose_shift(
                reference_features,
                _top_feature_means(data),
                feature_names=feature_names,
                alpha=float(protocol["shift_diagnostic"]["per_level_alpha"]),
                permutations=int(protocol["shift_diagnostic"]["permutations"]),
            )
            family_levels[str(level)] = {
                **_role_manifest(data),
                "factors": protocol["confirmatory_population"]["role_factor_overrides"][level],
                "certificate": certificate,
                "endpoint": _endpoint(certificate),
                "shift_diagnostic": shift,
            }
            states[str(level)] = _bundle_states(data, threshold)
            levels[str(level)] = family_levels[str(level)]
        ordered = [str(value) for value in spec["levels"]]
        transitions = {
            f"{left}_to_{right}": paired_bundle_transition(states[left], states[right])
            for left, right in pairwise(ordered)
        }
        warnings = [
            level for level in ordered if family_levels[level]["shift_diagnostic"]["warning"]
        ]
        failures = [
            level for level in ordered if not family_levels[level]["certificate"]["certified"]
        ]
        false_alarm_levels = [
            level
            for level in ordered
            if family_levels[level]["certificate"]["failing_experiments"] > 0
        ]
        missed_levels = [
            level
            for level in ordered
            if family_levels[level]["endpoint"]["missed_positive_seed_bundles"] > 0
        ]
        family_results[str(family_name)] = {
            "reference": spec["reference"],
            "interpretation": spec["interpretation"],
            "levels": family_levels,
            "paired_transitions": transitions,
            "first_shift_warning": warnings[0] if warnings else None,
            "first_certificate_failure": failures[0] if failures else None,
            "first_false_authorization_level": false_alarm_levels[0]
            if false_alarm_levels
            else None,
            "first_missed_positive_level": missed_levels[0] if missed_levels else None,
        }

    result = {
        "version": __version__,
        "git_revision": code_revision(),
        "analysis": "semantic_risk_orthogonal_shifts_v5_4",
        "protocol": str(protocol_path),
        "protocol_status": protocol["status"],
        "claim": claim,
        "fixed_threshold": threshold,
        "threshold_refitted": False,
        "inferential_unit": "seed_bundle_within_level",
        "simultaneous_contract": protocol["simultaneous_contract"],
        "families": family_results,
        "all_levels_certified": all(
            level["certificate"]["certified"] for level in levels.values()
        ),
        "levels_with_false_authorization": [
            name
            for name, level in levels.items()
            if level["certificate"]["failing_experiments"] > 0
        ],
        "levels_with_missed_positives": [
            name
            for name, level in levels.items()
            if level["endpoint"]["missed_positive_seed_bundles"] > 0
        ],
        "claim_boundary": protocol["claim_boundary"],
        "efficiency": {
            "wall_time_seconds": time.perf_counter() - started,
            "peak_rss_megabytes": _peak_rss_mb(),
            "generated_scenarios": generated_scenarios,
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    _write_markdown(result, markdown)
    return output


def _write_markdown(payload: dict[str, Any], path: Path) -> None:
    lines = [
        "# Orthogonal semantic-risk shifts v5.4",
        "",
        f"- All 14 levels certified: `{str(payload['all_levels_certified']).lower()}`",
        f"- False-authorization levels: `{payload['levels_with_false_authorization']}`",
        f"- Missed-positive levels: `{payload['levels_with_missed_positives']}`",
        "",
        "| Family | Level | Failures | Misses | Risk UCB | Coverage LCB | Shift warning | Certified |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for family_name, family in payload["families"].items():
        for level_name, level in family["levels"].items():
            certificate = level["certificate"]
            lines.append(
                f"| `{family_name}` | `{level_name}` | "
                f"{certificate['failing_experiments']} | "
                f"{level['endpoint']['missed_positive_seed_bundles']} | "
                f"{certificate['risk_upper_bound']:.4f} | "
                f"{certificate['coverage_lower_bound']:.4f} | "
                f"{str(level['shift_diagnostic']['warning']).lower()} | "
                f"{str(certificate['certified']).lower()} |"
            )
    lines.extend(("", payload["claim_boundary"], ""))
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
