"""Audit the standards, integrity, and prospective profile-v2 release."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import yaml

from mousebrainbench import __version__
from mousebrainbench.artifacts import code_revision

DEFAULT_PROTOCOL = Path("configs/benchmarks/standards_prospective_release.yaml")
DEFAULT_OUTPUT = Path("results/standards_prospective_release/summary.json")
DEFAULT_MARKDOWN = Path("results/standards_prospective_release/summary.md")


def _load(path: str) -> dict[str, Any]:
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"release artifact is missing: {source}")
    return json.loads(source.read_text())


def _clean_revision(payload: dict[str, Any]) -> bool:
    revision = payload.get("git_revision")
    return (
        isinstance(revision, str)
        and bool(revision)
        and revision != "unknown"
        and not revision.endswith("-dirty")
    )


def evaluate(protocol: dict[str, Any]) -> dict[str, Any]:
    artifacts = {name: _load(path) for name, path in protocol["artifacts"].items()}
    required = protocol["release_requirements"]
    mutation = artifacts["contract_mutation"]
    standards = artifacts["standards"]
    formal = artifacts["formal_properties"]
    attacks = artifacts["provenance_attacks"]
    scaling = artifacts["scalability_ablation"]
    application = artifacts["artifact_application"]
    dandi = artifacts["prospective_dandi"]
    largest_batch = max(
        row["packages"] for row in scaling["scalability"]["batch_scaling"]
    )
    largest_package = max(
        row["artifacts"] for row in scaling["scalability"]["artifact_scaling"]
    )
    leave_one_out = {
        name: row
        for name, row in scaling["ablation"]["systems"].items()
        if name.startswith("without_")
    }
    conditions = {
        "python_contract_exact": (
            mutation.get("cases") == required["contract_cases"]
            and mutation.get("profile_v2", {}).get("false_authorizations") == 0
            and mutation.get("profile_v2", {}).get("false_rejections") == 0
            and mutation.get("profile_v2", {}).get("exact_deficit_rate") == 1.0
        ),
        "independent_asp_exact": (
            mutation.get("asp_conformance", {}).get("cases") == required["asp_cases"]
            and mutation.get("asp_conformance", {}).get("rate") == 1.0
        ),
        "external_shacl_exact": (
            standards.get("cases") == required["shacl_cases"]
            and standards.get("all_endpoints_passed") is True
        ),
        "formal_properties_hold": (
            formal.get("total_checks", 0) >= required["minimum_formal_checks"]
            and formal.get("total_violations") == 0
        ),
        "declared_attacks_blocked": (
            attacks.get("cases") == required["attack_cases"]
            and attacks.get("attacked_cases") == required["attacked_cases"]
            and attacks.get("full_integrity_gate", {}).get("false_authorizations")
            == 0
            and attacks.get("full_integrity_gate", {}).get("false_rejections") == 0
        ),
        "every_integrity_control_is_decisive": bool(leave_one_out)
        and all(row["false_authorizations"] > 0 for row in leave_one_out.values()),
        "scaling_workloads_complete": (
            largest_batch >= required["minimum_batch_packages"]
            and largest_package >= required["minimum_artifacts_per_package"]
            and scaling.get("scalability", {}).get(
                "all_pristine_decisions_authorized"
            )
            is True
        ),
        "bounded_mouse_artifacts_only": (
            application.get("target_authorizations")
            == required["bounded_artifact_authorizations"]
            and application.get("strict_twin_authorizations")
            == required["complete_twin_authorizations"]
        ),
        "prospective_outcomes_retained": (
            dandi.get("positive_authorizations")
            == required["prospective_positive_authorizations"]
            and dandi.get("all_release_conditions_passed") is True
            and {row.get("authorization", {}).get("authorized") for row in dandi["applications"]}
            == {False, True}
        ),
        "all_source_revisions_clean": all(
            _clean_revision(payload) for payload in artifacts.values()
        ),
        "broad_claims_blocked": not any(protocol["claim_policy"].values()),
    }
    ready = all(conditions.values())
    return {
        "artifact_revisions": {
            name: payload.get("git_revision") for name, payload in artifacts.items()
        },
        "conditions": conditions,
        "all_release_conditions_passed": ready,
        "evidence_counts": {
            "contract_cases": mutation.get("cases"),
            "asp_cases": mutation.get("asp_conformance", {}).get("cases"),
            "shacl_cases": standards.get("cases"),
            "formal_checks": formal.get("total_checks"),
            "attack_cases": attacks.get("cases"),
            "attacked_cases": attacks.get("attacked_cases"),
            "largest_batch_packages": largest_batch,
            "largest_artifact_package": largest_package,
            "bounded_artifact_authorizations": application.get("target_authorizations"),
            "prospective_positive_authorizations": dandi.get("positive_authorizations"),
        },
        "claim_policy": protocol["claim_policy"],
        "technically_ready_for_kbs_manuscript": ready,
        "publication_assessment": (
            "bounded_kbs_submission_candidate_not_acceptance_guarantee"
            if ready
            else "standards_prospective_release_incomplete"
        ),
        "remaining_scientific_limits": [
            "the profile is author-defined rather than a consensus taxonomy",
            "formal and SHACL results are profile-relative rather than biological truth",
            "the attack suite covers a declared non-adaptive threat model",
            "MICRONS evidence comes from one cortical volume and remains observational",
            "the positive prospective DANDI result concerns one simple bounded endpoint",
            "human trace utility and independent content validity are not evaluated",
        ],
        "interpretation": protocol["interpretation"],
    }


def _write_markdown(payload: dict[str, Any], path: Path) -> None:
    lines = [
        "# Standards and prospective release",
        "",
        f"- Decision: `{payload['decision']}`",
        f"- KBS manuscript ready: `{payload['technically_ready_for_kbs_manuscript']}`",
        "",
        "| Release condition | Passed |",
        "|---|---:|",
    ]
    lines.extend(
        f"| `{name}` | {passed} |" for name, passed in payload["conditions"].items()
    )
    lines.extend(("", "## Remaining scientific limits", ""))
    lines.extend(f"- {item}" for item in payload["remaining_scientific_limits"])
    lines.extend(("", payload["interpretation"], ""))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines))


def run(
    *,
    protocol_path: Path = DEFAULT_PROTOCOL,
    output: Path = DEFAULT_OUTPUT,
    markdown: Path = DEFAULT_MARKDOWN,
) -> Path:
    protocol = yaml.safe_load(protocol_path.read_text())
    assessment = evaluate(protocol)
    payload = {
        "version": __version__,
        "git_revision": code_revision(),
        "analysis": "standards_prospective_release",
        "protocol": str(protocol_path),
        "protocol_sha256": hashlib.sha256(protocol_path.read_bytes()).hexdigest(),
        **assessment,
        "decision": (
            "standards_prospective_release_complete"
            if assessment["technically_ready_for_kbs_manuscript"]
            else "standards_prospective_release_incomplete"
        ),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n")
    _write_markdown(payload, markdown)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--markdown", type=Path, default=DEFAULT_MARKDOWN)
    args = parser.parse_args()
    print(run(protocol_path=args.protocol, output=args.output, markdown=args.markdown).resolve())


if __name__ == "__main__":
    main()
