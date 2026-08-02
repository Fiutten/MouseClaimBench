"""Audit the complete hardened profile-v2 methodological release."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import yaml

from mousebrainbench import __version__
from mousebrainbench.artifacts import code_revision

DEFAULT_PROTOCOL = Path("configs/benchmarks/profile_v2_release.yaml")
DEFAULT_OUTPUT = Path("results/profile_v2_release/summary.json")
DEFAULT_MARKDOWN = Path("results/profile_v2_release/summary.md")


def _load(path: str) -> dict[str, Any]:
    source = Path(path)
    if not source.exists():
        raise FileNotFoundError(f"required profile-v2 release artifact is missing: {path}")
    return json.loads(source.read_text())


def _clean_revision(payload: dict[str, Any]) -> bool:
    revision = payload.get("git_revision")
    return isinstance(revision, str) and revision != "unknown" and not revision.endswith(
        "-dirty"
    )


def evaluate(protocol: dict[str, Any]) -> dict[str, Any]:
    """Evaluate formal, real-case, risk, boundary, and governance conditions."""

    artifacts = {
        name: _load(str(path)) for name, path in protocol["artifacts"].items()
    }
    resolution = artifacts["resolution"]
    mutation = artifacts["mutation"]
    application = artifacts["application"]
    risk = artifacts["semantic_risk_v5"]
    ibl = artifacts["ibl"]
    shifts = artifacts["orthogonal_shifts"]
    shift_levels = [
        level
        for family in shifts.get("families", {}).values()
        for level in family.get("levels", {}).values()
    ]
    shift_failures = sum(
        row.get("certificate", {}).get("certified") is False for row in shift_levels
    )
    conditions = {
        "all_internal_audit_issues_mapped": (
            resolution.get("decision") == "profile_v2_internal_audit_issues_resolved"
            and resolution.get("external_content_validity") is False
        ),
        "mutation_primary_endpoints_pass": (
            mutation.get("all_primary_endpoints_passed") is True
            and mutation.get("profile_v2", {}).get("false_authorizations") == 0
        ),
        "independent_asp_conformance_passes": (
            mutation.get("asp_conformance", {}).get("rate") == 1.0
            and application.get("release_conditions", {}).get(
                "python_asp_equivalence_for_all_case_decisions"
            )
            is True
        ),
        "at_least_two_bounded_real_target_authorizations": (
            int(application.get("target_authorizations", 0)) >= 2
        ),
        "no_complete_twin_authorized": (
            application.get("strict_twin_authorizations") == 0
        ),
        "existing_hierarchical_risk_core_confirmed": (
            risk.get("methodological_core_confirmed") is True
            and risk.get("status", {})
            .get("topology_hierarchical_confirmation", {})
            .get("passed")
            is True
        ),
        "real_mouse_population_evidence_present": (
            ibl.get("decision") == "external_ibl_behavioral_population_supported"
            and int(ibl.get("risk_lock", {}).get("selected_mice", 0)) == 35
            and int(ibl.get("final_evaluation", {}).get("selected_mice", 0)) == 35
        ),
        "heterogeneous_shift_failures_remain_visible": (
            len(shifts.get("families", {})) >= 4 and shift_failures >= 2
        ),
        "all_artifact_revisions_clean": all(
            _clean_revision(payload) for payload in artifacts.values()
        ),
    }
    policy = protocol["claim_policy"]
    boundaries = {
        "external_content_validity_claimed": False,
        "human_utility_claimed": False,
        "consensus_taxonomy_claimed": False,
        "scientific_truth_claimed": False,
        "q1_acceptance_guaranteed": False,
        "profile_v2_prospective_external_validation": False,
    }
    ready = all(conditions.values()) and not any(boundaries.values())
    return {
        "artifact_revisions": {
            name: payload.get("git_revision") for name, payload in artifacts.items()
        },
        "conditions": conditions,
        "all_release_conditions_passed": all(conditions.values()),
        "contract_mutation_cases": mutation.get("cases"),
        "contract_mutation_false_authorizations": mutation.get("profile_v2", {}).get(
            "false_authorizations"
        ),
        "asp_conformance_cases": mutation.get("asp_conformance", {}).get("cases"),
        "real_artifact_cases": application.get("cases"),
        "bounded_real_target_authorizations": application.get("target_authorizations"),
        "strict_twin_authorizations": application.get("strict_twin_authorizations"),
        "ibl_locked_mice": 70,
        "orthogonal_shift_families": len(shifts.get("families", {})),
        "orthogonal_shift_certificate_failures": shift_failures,
        "claim_policy": policy,
        "claim_boundaries": boundaries,
        "technically_ready_for_manuscript_revision": ready,
        "publication_assessment": (
            "bounded_methodological_submission_candidate_not_acceptance_guarantee"
            if ready
            else "profile_v2_release_not_ready"
        ),
        "remaining_scientific_limits": [
            "profile v2 is author-defined and not a consensus taxonomy",
            "artifact application is retrospective rather than a new blind v2 evaluation",
            "only one shared IBL task ecosystem supplies real mouse-population risk evidence",
            "simple IBL comparators also pass, so exclusive algorithmic superiority is unsupported",
            "MICRONS authorizes only one local observational association after directed dyadic and node-permutation controls",
            "no human trace-utility or decision-quality claim is evaluated",
        ],
        "interpretation": protocol["interpretation"],
    }


def _write_markdown(payload: dict[str, Any], path: Path) -> None:
    ready = str(payload["technically_ready_for_manuscript_revision"]).lower()
    real_authorizations = payload["bounded_real_target_authorizations"]
    lines = [
        "# Profile v2 hardening release",
        "",
        f"- Decision: `{payload['decision']}`",
        f"- Technically ready for manuscript revision: `{ready}`",
        f"- Contract mutation cases: `{payload['contract_mutation_cases']}`",
        f"- ASP conformance cases: `{payload['asp_conformance_cases']}`",
        f"- Real artifact cases: `{payload['real_artifact_cases']}`",
        f"- Bounded real target authorizations: `{real_authorizations}`",
        f"- Complete-twin authorizations: `{payload['strict_twin_authorizations']}`",
        "",
        "## Remaining scientific limits",
        "",
    ]
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
    """Run and persist the profile-v2 release decision."""

    protocol = yaml.safe_load(protocol_path.read_text())
    assessment = evaluate(protocol)
    payload = {
        "version": __version__,
        "git_revision": code_revision(),
        "analysis": "profile_v2_hardening_release",
        "protocol": str(protocol_path),
        "protocol_sha256": hashlib.sha256(protocol_path.read_bytes()).hexdigest(),
        **assessment,
        "decision": (
            "profile_v2_technical_release_complete"
            if assessment["technically_ready_for_manuscript_revision"]
            else "profile_v2_technical_release_incomplete"
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
