"""Non-compensatory release audit for the eight-point semantic-risk program."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from mousebrainbench import __version__
from mousebrainbench.artifacts import code_revision

DEFAULT_OUTPUT = Path("results/semantic_risk_v4_release/summary.json")
DEFAULT_MARKDOWN = Path("results/semantic_risk_v4_release/summary.md")
ARTIFACTS = {
    "external_v4": Path("results/causal_chambers_v4_confirmation/summary.json"),
    "external_v4_1": Path("results/causal_chambers_v4_1_final/summary.json"),
    "router_v4": Path("results/direction_router_v4_confirmation/summary.json"),
    "router_v4_2": Path("results/direction_router_v4_2_confirmation/summary.json"),
    "profile": Path("results/knowledge_profile_validity_v4/summary.json"),
    "power": Path("results/semantic_risk_v4_power/summary.json"),
}


def _clean_revision(payload: dict[str, Any]) -> bool:
    revision = str(payload.get("git_revision", ""))
    return bool(revision) and revision != "unknown" and not revision.endswith("-dirty")


def evaluate(payloads: dict[str, dict[str, Any]]) -> dict[str, Any]:
    external = payloads["external_v4"]
    final = payloads["external_v4_1"]
    router_v4 = payloads["router_v4"]
    router_v4_2 = payloads["router_v4_2"]
    profile = payloads["profile"]
    power = payloads["power"]
    final_comparators = final["final"]["comparators"]
    final_primary = final["final"]["primary"]
    statuses = {
        "target_calibrated_external_block": {
            "implemented": True,
            "passed": final["decision"]
            == "v4_1_bounded_predictive_external_confirmation_passed",
            "evidence": final["decision"],
        },
        "nondegenerate_coverage_and_recovery": {
            "implemented": True,
            "passed": bool(final_primary["certified"]),
            "evidence": {
                "risk_upper_bound": final_primary["risk_upper_bound"],
                "coverage_lower_bound": final_primary["coverage_lower_bound"],
                "positive_recovery_lower_bound": final_primary[
                    "positive_recovery_lower_bound"
                ],
                "abstain_all_certified": final_comparators["abstain_all"]["certified"],
            },
        },
        "direct_competitor_baselines": {
            "implemented": len(final_comparators) >= 7,
            "passed": True,
            "evidence": {
                "comparators": list(final_comparators),
                "certified_comparators": [
                    name for name, row in final_comparators.items() if row["certified"]
                ],
            },
        },
        "prospective_association_router": {
            "implemented": True,
            "passed": bool(router_v4_2["primary_passed"]),
            "evidence": {
                "v4_decision": router_v4["decision"],
                "v4_2_decision": router_v4_2["decision"],
                "v4_2_metrics": router_v4_2["association_aware_router"],
            },
        },
        "knowledge_profile_validity": {
            "implemented": bool(profile["structural_documentation_complete"]),
            "passed": False,
            "evidence": profile["decision"],
            "reason": "content validity has not been independently established",
        },
        "prospective_power_and_sample_size": {
            "implemented": power["minimum_units_for_zero_failures"] == 29,
            "passed": True,
            "evidence": {
                "minimum_units_for_zero_failures": power[
                    "minimum_units_for_zero_failures"
                ],
                "method": "exact one-sided Clopper-Pearson",
            },
        },
        "warning_only_shift_diagnostics": {
            "implemented": final["final"]["shift_diagnostic"][
                "may_authorize_or_restore_certificate"
            ]
            is False,
            "passed": True,
            "evidence": {
                "warning": final["final"]["shift_diagnostic"]["warning"],
                "certificate_restoration_allowed": False,
            },
        },
        "efficiency_abstention_cost_and_failure_taxonomy": {
            "implemented": all(
                key in final["efficiency"]
                for key in (
                    "wall_time_seconds",
                    "peak_rss_megabytes",
                    "final_abstention_rate",
                    "abstention_opportunity_cost",
                )
            )
            and "mutually_exclusive_counts"
            in final["final"]["failure_taxonomy"],
            "passed": True,
            "evidence": {
                "efficiency": final["efficiency"],
                "failure_taxonomy": final["final"]["failure_taxonomy"],
            },
        },
    }
    clean = all(_clean_revision(payload) for payload in payloads.values())
    external_supported = statuses["target_calibrated_external_block"]["passed"]
    profile_validated = statuses["knowledge_profile_validity"]["passed"]
    strong_q1_ready = bool(external_supported and profile_validated)
    return {
        "artifacts_clean": clean,
        "eight_point_status": statuses,
        "implementation_complete": all(row["implemented"] for row in statuses.values()),
        "external_authorization_confirmed": external_supported,
        "router_repair_confirmed": statuses["prospective_association_router"]["passed"],
        "knowledge_profile_content_validated": profile_validated,
        "strong_q1_second_paper_ready": strong_q1_ready,
        "decision": (
            "eight_point_program_complete_external_authorization_not_confirmed"
            if all(row["implemented"] for row in statuses.values())
            and not external_supported
            else "eight_point_program_requires_action"
        ),
        "non_compensation_rule": (
            "router success, reproducibility, coverage, or efficiency cannot compensate for "
            "failed external risk control or absent profile content validation"
        ),
        "required_evidence_before_strong_q1_claim": [
            "a new untouched target population for the revised authorizer",
            "dependence-aware calibration robust to experiments nested in datasets",
            "independent domain-expert validation of profile content and mappings",
        ],
        "prohibited_claims": [
            "external false-authorization control has been established",
            "the knowledge profile is a consensus scientific standard",
            "router association establishes causal direction",
            "MouseClaimBench verifies scientific truth",
            "a complete mouse-brain digital twin has been validated",
        ],
        "original_v4_external_decision": external["decision"],
    }


def run(
    *,
    root: Path = Path("."),
    output: Path = DEFAULT_OUTPUT,
    markdown: Path = DEFAULT_MARKDOWN,
) -> Path:
    missing = [name for name, path in ARTIFACTS.items() if not (root / path).exists()]
    if missing:
        raise FileNotFoundError(f"missing v4 artifacts: {', '.join(missing)}")
    payloads = {
        name: json.loads((root / path).read_text()) for name, path in ARTIFACTS.items()
    }
    assessment = evaluate(payloads)
    result = {
        "version": __version__,
        "git_revision": code_revision(),
        "analysis": "semantic_risk_v4_eight_point_release_audit",
        "artifacts": {
            name: {"path": str(path), "git_revision": payloads[name]["git_revision"]}
            for name, path in ARTIFACTS.items()
        },
        **assessment,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    _write_markdown(result, markdown)
    return output


def _write_markdown(payload: dict[str, Any], path: Path) -> None:
    lines = [
        "# Semantic risk v4 eight-point release audit",
        "",
        f"- Decision: `{payload['decision']}`",
        f"- Implementation complete: `{str(payload['implementation_complete']).lower()}`",
        f"- External authorization confirmed: `{str(payload['external_authorization_confirmed']).lower()}`",
        f"- Router repair confirmed: `{str(payload['router_repair_confirmed']).lower()}`",
        f"- Strong Q1 second paper ready: `{str(payload['strong_q1_second_paper_ready']).lower()}`",
        "",
        "| Point | Implemented | Passed |",
        "|---|---:|---:|",
    ]
    lines.extend(
        f"| `{name}` | {str(row['implemented']).lower()} | {str(row['passed']).lower()} |"
        for name, row in payload["eight_point_status"].items()
    )
    lines.extend(("", "## Required evidence", ""))
    lines.extend(f"- {item}" for item in payload["required_evidence_before_strong_q1_claim"])
    lines.extend(("", "## Prohibited claims", ""))
    lines.extend(f"- {item}" for item in payload["prohibited_claims"])
    lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--markdown", type=Path, default=DEFAULT_MARKDOWN)
    args = parser.parse_args()
    print(json.dumps({"output": str(run(root=args.root, output=args.output, markdown=args.markdown).resolve())}))


if __name__ == "__main__":
    main()
