"""Release gate for the evidence-constrained knowledge-system package."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from mousebrainbench import __version__
from mousebrainbench.artifacts import code_revision
from mousebrainbench.knowledge import load_default_profile_basis


DEFAULT_OUTPUT = Path("results/knowledge_system_release/summary.json")
DEFAULT_MARKDOWN = Path("results/knowledge_system_release/summary.md")

REQUIREMENTS = {
    "results/knowledge_system_audit/summary.json": (
        "decision",
        "knowledge_system_reproduces_frozen_policy_with_complete_traces",
    ),
    "results/oracle_sem_claim_benchmark/summary.json": (
        "decision",
        "oracle_benchmark_supports_non_compensatory_contract_with_finite_sample_errors",
    ),
    "results/real_case_claim_matrix/summary.json": (
        "decision",
        "artifact_grounded_case_matrix_complete_with_explicit_limits",
    ),
    "results/scifact_claim_verification/summary.json": (
        "decision",
        "scifact_external_claim_audit_ready",
    ),
    "results/tuebingen_causal_direction/summary.json": (
        "decision",
        "tuebingen_external_direction_benchmark_ready",
    ),
    "results/claim_adversarial_v2/summary.json": (
        "validation_role",
        "software_contract_conformance_not_independent_validation",
    ),
    "results/claimbench_component_ablation/summary.json": (
        "decision",
        "claimbench_components_have_nonredundant_value",
    ),
    "results/claimbench_threat_model/summary.json": (
        "decision",
        "claimbench_threat_model_passed_with_boundaries",
    ),
    "results/manuscript_claim_audit/summary.json": (
        "decision",
        "manuscript_claim_audit_passed",
    ),
}


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def run(
    output: Path = DEFAULT_OUTPUT,
    markdown: Path = DEFAULT_MARKDOWN,
    root: Path = Path("."),
) -> Path:
    """Check KBS artifacts without treating a human study as a release dependency."""

    rows: list[dict[str, Any]] = []
    missing: list[str] = []
    failing: list[str] = []
    dirty: list[str] = []
    for relative, (field, expected) in REQUIREMENTS.items():
        path = root / relative
        if not path.exists():
            missing.append(relative)
            rows.append(
                {
                    "artifact": relative,
                    "exists": False,
                    "field": field,
                    "expected": expected,
                    "observed": None,
                    "git_revision": None,
                }
            )
            continue
        artifact = _load(path)
        observed = artifact.get(field)
        revision = artifact.get("git_revision")
        if observed != expected:
            failing.append(relative)
        if not isinstance(revision, str) or revision.endswith("-dirty"):
            dirty.append(relative)
        rows.append(
            {
                "artifact": relative,
                "exists": True,
                "field": field,
                "expected": expected,
                "observed": observed,
                "git_revision": revision,
            }
        )

    audit_path = root / "results/knowledge_system_audit/summary.json"
    audit = _load(audit_path) if audit_path.exists() else {}
    profile = audit.get("knowledge_profile", {})
    basis = load_default_profile_basis()
    basis_complete = (
        basis.get("profile_id") == "mouse_brain_claims"
        and basis.get("version") == "1.1.0"
        and basis.get("status")
        == "author_proposed_literature_grounded_not_externally_validated"
        and basis.get("independent_expert_validation") == "not_performed"
        and len(basis.get("relations", ())) == 22
    )
    profile_structurally_valid = (
        profile.get("profile_id") == "mouse_brain_claims"
        and profile.get("version") == "1.1.0"
        and str(profile.get("source_hash", "")).startswith("sha256:")
        and audit.get("exact_decision_matches") == 40
        and audit.get("explanation_complete_count") == 40
        and basis_complete
    )
    if audit and not profile_structurally_valid:
        failing.append(str(audit_path.relative_to(root)))

    boundaries = [
        {
            "claim": "cross-domain empirical generality",
            "reason": "only the computational mouse-brain profile is evaluated",
        },
        {
            "claim": "external biological replication",
            "reason": "the real cases do not include an independent cross-resource replication",
        },
        {
            "claim": "complete or causal mouse-brain digital twin",
            "reason": "no case supplies both intervention evidence and whole-brain coverage",
        },
        {
            "claim": "improved human decision quality",
            "reason": "the KBS evaluation tests inference behavior, not human outcomes",
        },
        {
            "claim": "expert consensus or external content validity",
            "reason": "the profile is author-proposed and has not been assessed by an independent panel",
        },
    ]
    ready = not missing and not failing and not dirty and profile_structurally_valid
    payload = {
        "version": __version__,
        "git_revision": code_revision(),
        "analysis": "knowledge_system_release_gate_v1",
        "required_artifacts": rows,
        "knowledge_profile": {
            "profile_id": profile.get("profile_id"),
            "version": profile.get("version"),
            "source_hash": profile.get("source_hash"),
            "structurally_valid": profile_structurally_valid,
            "relation_records": len(basis.get("relations", ())),
            "curation_status": basis.get("status"),
            "independent_expert_validation": basis.get("independent_expert_validation"),
        },
        "missing_artifacts": sorted(set(missing)),
        "failing_artifacts": sorted(set(failing)),
        "dirty_artifacts": sorted(set(dirty)),
        "human_study_required_for_stated_scope": False,
        "scientific_claim_boundaries": boundaries,
        "decision": (
            "knowledge_system_method_package_ready_with_declared_boundaries"
            if ready
            else "knowledge_system_release_requires_action"
        ),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2))
    write_markdown(payload, markdown)
    return output


def write_markdown(payload: dict[str, Any], markdown: Path) -> None:
    """Write a concise release report."""

    lines = [
        "# Knowledge-System Release Gate",
        "",
        f"- Decision: `{payload['decision']}`",
        f"- Knowledge profile structurally valid: "
        f"`{payload['knowledge_profile']['structurally_valid']}`",
        f"- Profile curation status: `{payload['knowledge_profile']['curation_status']}`",
        f"- Independent expert validation: "
        f"`{payload['knowledge_profile']['independent_expert_validation']}`",
        f"- Missing artifacts: `{len(payload['missing_artifacts'])}`",
        f"- Failing artifacts: `{len(payload['failing_artifacts'])}`",
        f"- Dirty artifacts: `{len(payload['dirty_artifacts'])}`",
        f"- Human study required for stated scope: "
        f"`{payload['human_study_required_for_stated_scope']}`",
        "",
        "## Scientific claim boundaries",
        "",
    ]
    lines.extend(
        f"- **{row['claim']}**: {row['reason']}"
        for row in payload["scientific_claim_boundaries"]
    )
    lines.append("")
    markdown.parent.mkdir(parents=True, exist_ok=True)
    markdown.write_text("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--markdown", type=Path, default=DEFAULT_MARKDOWN)
    parser.add_argument("--root", type=Path, default=Path("."))
    args = parser.parse_args()
    print(json.dumps({"output": str(run(args.output, args.markdown, args.root).resolve())}))


if __name__ == "__main__":
    main()
