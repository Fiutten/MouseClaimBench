"""Non-compensatory release audit for the semantic-risk v5 experiments."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from mousebrainbench import __version__
from mousebrainbench.artifacts import code_revision

DEFAULT_OUTPUT = Path("results/semantic_risk_v5_release/summary.json")
DEFAULT_MARKDOWN = Path("results/semantic_risk_v5_release/summary.md")
ARTIFACTS = {
    "global_v5": Path("results/timegraph_v5_confirmation/summary.json"),
    "topology_v5_1": Path("results/timegraph_v5_1_topology_confirmation/summary.json"),
    "shift_v5_2": Path("results/semantic_risk_shift_sweep_v5_2/summary.json"),
    "causalrivers": Path("results/causalrivers_v5_transport/summary.json"),
    "profile": Path("results/knowledge_profile_external_validation_v5/summary.json"),
}


def _clean_revision(payload: dict[str, Any]) -> bool:
    revision = str(payload.get("git_revision", ""))
    return bool(revision) and revision != "unknown" and not revision.endswith("-dirty")


def evaluate(payloads: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Evaluate all evidence blocks without allowing one block to mask another."""

    global_v5 = payloads["global_v5"]
    topology = payloads["topology_v5_1"]
    shift = payloads["shift_v5_2"]
    rivers = payloads["causalrivers"]
    profile = payloads["profile"]

    primary_name = "v5_1_fixed_hierarchical_threshold"
    risk = topology["risk_lock"]["comparators"][primary_name]
    final = topology["final_evaluation"]["comparators"][primary_name]
    ood = topology["ood_stress"]["comparators"][primary_name]
    competing_names = [name for name in topology["risk_lock"]["comparators"] if name != primary_name]
    competitors_fail = all(
        not topology[block]["comparators"][name]["certified"]
        for block in ("risk_lock", "final_evaluation")
        for name in competing_names
    )
    profile_validated = bool(profile["profile_content_validated"])
    external_certificate = bool(rivers["exact_external_certificate_allowed"])
    shift_boundary_observed = bool(
        shift["primary_endpoints"]["warning_occurs_no_later_than_certificate_failure"]
    )
    statuses = {
        "global_six_family_contract": {
            "implemented": True,
            "passed": False,
            "evidence": global_v5["decision"],
            "expected_interpretation": "negative result retained; final block remained closed",
        },
        "topology_hierarchical_confirmation": {
            "implemented": True,
            "passed": bool(risk["certified"] and final["certified"]),
            "evidence": {
                "risk_lock_risk_ucb": risk["risk_upper_bound"],
                "final_risk_ucb": final["risk_upper_bound"],
                "final_coverage_lcb": final["coverage_lower_bound"],
                "final_recovery_lcb": final["positive_recovery_lower_bound"],
            },
        },
        "same_endpoint_baselines": {
            "implemented": len(competing_names) == 4,
            "passed": competitors_fail,
            "evidence": {"comparators": competing_names, "all_failed": competitors_fail},
        },
        "out_of_distribution_boundary": {
            "implemented": True,
            "passed": not bool(ood["certified"]),
            "evidence": {
                "risk_ucb": ood["risk_upper_bound"],
                "shift_warning": topology["ood_stress"]["shift_from_risk_lock"]["warning"],
            },
        },
        "prospective_shift_path": {
            "implemented": len(shift["levels"]) == 6,
            "passed": shift_boundary_observed,
            "evidence": shift["primary_endpoints"],
            "general_detector_validated": False,
        },
        "real_causalrivers_transport": {
            "implemented": len(rivers["blocks"]) == 4,
            "passed": external_certificate,
            "evidence": {
                "independent_top_level_clusters": rivers["independent_top_level_clusters"],
                "exact_external_certificate_allowed": external_certificate,
                "decision": rivers["decision"],
            },
        },
        "independent_profile_content_validation": {
            "implemented": True,
            "passed": profile_validated,
            "evidence": {
                "completed_independent_raters": profile["completed_independent_raters"],
                "decision": profile["decision"],
            },
        },
    }
    core_confirmed = bool(
        statuses["topology_hierarchical_confirmation"]["passed"]
        and statuses["same_endpoint_baselines"]["passed"]
        and statuses["out_of_distribution_boundary"]["passed"]
        and statuses["prospective_shift_path"]["passed"]
    )
    strong_q1_ready = bool(core_confirmed and external_certificate and profile_validated)
    return {
        "artifacts_clean": all(_clean_revision(payload) for payload in payloads.values()),
        "status": statuses,
        "methodological_core_confirmed": core_confirmed,
        "global_population_supported": False,
        "real_domain_external_risk_confirmed": external_certificate,
        "knowledge_profile_content_validated": profile_validated,
        "strong_q1_second_paper_ready": strong_q1_ready,
        "decision": (
            "methodological_core_confirmed_external_and_content_validity_open"
            if core_confirmed and not strong_q1_ready
            else "v5_release_requires_action"
        ),
        "non_compensation_rule": (
            "synthetic confirmation and early shift warning cannot compensate for failed OOD "
            "transport, absent real-domain certification, or pending profile content validity"
        ),
        "supported_claims": [
            "the fixed topology contract met the hierarchical certificate twice in its declared synthetic population",
            "the compared simple baselines did not meet the same certificate",
            "the fixed certificate failed under stronger synthetic shift and transported poorly to CausalRivers",
            "the shift warning preceded certificate loss on one frozen severity path",
        ],
        "required_before_strong_q1_claim": [
            "independent expert content validation of the evidence-to-claim profile",
            "a target with enough independent real top-level clusters for external risk inference",
            "confirmation on a new untouched real population without threshold refitting",
        ],
        "prohibited_claims": [
            "the six-family global authorizer is validated",
            "false-authorization risk is controlled in CausalRivers or mouse-brain data",
            "the shift diagnostic is a calibrated general-purpose detector",
            "the knowledge profile is an expert consensus standard",
            "MouseClaimBench verifies causal truth or validates a complete mouse-brain twin",
        ],
    }


def run(
    *, root: Path = Path("."), output: Path = DEFAULT_OUTPUT, markdown: Path = DEFAULT_MARKDOWN
) -> Path:
    missing = [name for name, path in ARTIFACTS.items() if not (root / path).exists()]
    if missing:
        raise FileNotFoundError(f"missing v5 artifacts: {', '.join(missing)}")
    payloads = {
        name: json.loads((root / path).read_text()) for name, path in ARTIFACTS.items()
    }
    result = {
        "version": __version__,
        "git_revision": code_revision(),
        "analysis": "semantic_risk_v5_noncompensatory_release_audit",
        "artifacts": {
            name: {"path": str(path), "git_revision": payloads[name]["git_revision"]}
            for name, path in ARTIFACTS.items()
        },
        **evaluate(payloads),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    _write_markdown(result, markdown)
    return output


def _write_markdown(payload: dict[str, Any], path: Path) -> None:
    lines = [
        "# Semantic-risk v5 release audit",
        "",
        f"- Decision: `{payload['decision']}`",
        f"- Methodological core confirmed: `{str(payload['methodological_core_confirmed']).lower()}`",
        f"- Real-domain external risk confirmed: `{str(payload['real_domain_external_risk_confirmed']).lower()}`",
        f"- Profile content validated: `{str(payload['knowledge_profile_content_validated']).lower()}`",
        f"- Strong Q1 package ready: `{str(payload['strong_q1_second_paper_ready']).lower()}`",
        "",
        "| Evidence block | Implemented | Passed |",
        "|---|---:|---:|",
    ]
    lines.extend(
        f"| `{name}` | {str(row['implemented']).lower()} | {str(row['passed']).lower()} |"
        for name, row in payload["status"].items()
    )
    lines.extend(("", "## Supported claims", ""))
    lines.extend(f"- {item}" for item in payload["supported_claims"])
    lines.extend(("", "## Required evidence", ""))
    lines.extend(f"- {item}" for item in payload["required_before_strong_q1_claim"])
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
    print(run(root=args.root, output=args.output, markdown=args.markdown).resolve())


if __name__ == "__main__":
    main()
