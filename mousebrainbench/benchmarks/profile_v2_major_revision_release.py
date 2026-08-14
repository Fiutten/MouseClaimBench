"""Audit the construct-validity response package for the major revision."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import yaml

from mousebrainbench import __version__
from mousebrainbench.artifacts import code_revision

DEFAULT_PROTOCOL = Path("configs/benchmarks/profile_v2_major_revision_release.yaml")
DEFAULT_OUTPUT = Path("results/profile_v2_major_revision_release/summary.json")
DEFAULT_MARKDOWN = Path("results/profile_v2_major_revision_release/summary.md")


def _load(path: str) -> dict[str, Any]:
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"major-revision artifact is missing: {source}")
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
    required = protocol["requirements"]
    parent = artifacts["parent_submission"]
    trace = artifacts["knowledge_traceability"]
    sensitivity = artifacts["structural_sensitivity"]
    explanations = artifacts["explanation_fidelity"]
    integrity = artifacts["compositional_integrity"]
    conditions = {
        "parent_submission_release_preserved": (
            parent.get("decision") == "standards_prospective_release_complete"
            and parent.get("technically_ready_for_kbs_manuscript") is True
        ),
        "author_policy_traceability_complete": (
            trace.get("claim_to_evidence_relations") == required["claim_to_evidence_relations"]
            and trace.get("predicate_contracts") == required["predicate_contracts"]
            and trace.get("required_observation_slots") == required["required_observation_slots"]
            and trace.get("all_conditions_passed") is True
            and trace.get("independent_content_validity") is False
        ),
        "one_edge_policy_perturbation_complete": (
            sensitivity.get("profile_variants") == required["profile_variants"]
            and sensitivity.get("profile_case_evaluations") == required["profile_case_evaluations"]
            and sensitivity.get("completed") is True
        ),
        "counterfactual_explanations_faithful": (
            explanations.get("random_packages") == required["random_explanation_packages"]
            and explanations.get("all_explanation_properties_hold") is True
        ),
        "declared_attack_compositions_complete": (
            integrity.get("in_model_attacked_packages")
            == required["declared_compositional_attacks"]
            and integrity.get("in_model_false_authorizations") == 0
            and integrity.get("in_model_exact_trace_rate") == 1.0
        ),
        "trust_boundary_escape_is_reported": (
            integrity.get("trust_boundary_negative_controls")
            == required["trust_boundary_negative_controls"]
            and integrity.get("trust_boundary_authorizations")
            == required["trust_boundary_negative_controls"]
        ),
        "new_artifact_revisions_clean": all(
            _clean_revision(payload)
            for name, payload in artifacts.items()
            if name != "parent_submission"
        ),
        "prohibited_claims_remain_false": not any(protocol["claim_policy"].values()),
    }
    complete = all(conditions.values())
    return {
        "artifact_revisions": {
            name: payload.get("git_revision") for name, payload in artifacts.items()
        },
        "conditions": conditions,
        "all_release_conditions_passed": complete,
        "claim_policy": protocol["claim_policy"],
        "remaining_limits": [
            "profile v2 remains author-defined and not independently content-validated",
            "counterfactual fidelity does not establish human explanation utility",
            "one-edge relation perturbations check monotonicity but do not validate alternative profiles",
            "coherent metadata or content forgery remains undetectable without an external trust anchor",
            "mouse-brain applications retain their original local and resource-specific boundaries",
        ],
        "interpretation": protocol["interpretation"],
    }


def _write_markdown(payload: dict[str, Any], path: Path) -> None:
    lines = [
        "# Profile v2 major-revision release",
        "",
        f"- Decision: `{payload['decision']}`",
        "",
        "| Release condition | Passed |",
        "|---|---:|",
    ]
    lines.extend(
        f"| `{name}` | {str(passed).lower()} |" for name, passed in payload["conditions"].items()
    )
    lines.extend(("", "## Remaining limits", ""))
    lines.extend(f"- {item}" for item in payload["remaining_limits"])
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
        "analysis": "profile_v2_major_revision_release",
        "protocol": str(protocol_path),
        "protocol_sha256": hashlib.sha256(protocol_path.read_bytes()).hexdigest(),
        **assessment,
        "decision": (
            "profile_v2_major_revision_release_complete"
            if assessment["all_release_conditions_passed"]
            else "profile_v2_major_revision_release_incomplete"
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
