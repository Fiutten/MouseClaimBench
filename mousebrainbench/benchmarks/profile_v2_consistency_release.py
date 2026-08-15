"""Audit the paper-code-result consistency release after the external code review."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import yaml

from mousebrainbench import __version__
from mousebrainbench.artifacts import code_revision

DEFAULT_PROTOCOL = Path("configs/benchmarks/profile_v2_consistency_release.yaml")
DEFAULT_OUTPUT = Path("results/profile_v2_consistency_release/summary.json")
DEFAULT_MARKDOWN = Path("results/profile_v2_consistency_release/summary.md")


def _load(path: str) -> dict[str, Any]:
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"consistency-release artifact is missing: {source}")
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
    parent = artifacts["parent_second_review"]
    contract = artifacts["contract_mutation"]
    structural = artifacts["structural_conformance"]
    original = artifacts["provenance_attacks"]
    compositional = artifacts["compositional_integrity"]
    regression = artifacts["integrity_regression"]
    final_gate = artifacts["final_gate"]
    scaling = artifacts["scalability"]
    dandi = artifacts["prospective_dandi"]
    artifact_points = [
        row["artifacts"] for row in scaling["scalability"]["artifact_scaling"]
    ]
    conditions = {
        "parent_release_preserved": (
            parent.get("decision") == "profile_v2_second_review_release_complete"
            and parent.get("all_release_conditions_passed") is True
        ),
        "domain_contract_exact": (
            contract.get("cases") == required["contract_cases"]
            and contract.get("profile_v2", {}).get("false_authorizations") == 0
            and contract.get("profile_v2", {}).get("false_rejections") == 0
            and contract.get("profile_v2", {}).get("exact_deficit_rate") == 1.0
            and contract.get("asp_conformance", {}).get("rate") == 1.0
        ),
        "structural_contract_exact": (
            structural.get("cases") == required["structural_cases"]
            and structural.get("all_endpoints_passed") is True
            and structural.get("shacl", {}).get("exact_structural_deficit_rate") == 1.0
            and structural.get("shacl", {}).get("structurally_valid_domain_refusals", 0) > 0
        ),
        "original_integrity_benchmark_preserved": (
            original.get("cases") == required["original_attack_cases"]
            and original.get("all_endpoints_passed") is True
        ),
        "compositional_integrity_exact": (
            compositional.get("in_model_attacked_packages")
            == required["compositional_attacks"]
            and compositional.get("in_model_false_authorizations") == 0
            and compositional.get("in_model_exact_trace_rate") == 1.0
        ),
        "extended_integrity_regression_exact": (
            regression.get("cases") == required["integrity_regression_cases"]
            and regression.get("all_endpoints_passed") is True
        ),
        "three_gate_composition_exact": (
            final_gate.get("logical_combinations")
            == required["final_gate_combinations"]
            and final_gate.get("all_endpoints_passed") is True
        ),
        "scalability_table_complete": artifact_points
        == required["artifact_scaling_points"],
        "dandi_outcomes_unchanged": (
            dandi.get("decision") == "prospective_dandi_profile_v2_1_complete"
            and dandi.get("positive_authorizations") == 1
        ),
        "new_artifact_revisions_clean": all(
            _clean_revision(payload)
            for name, payload in artifacts.items()
            if name
            in {
                "contract_mutation",
                "structural_conformance",
                "provenance_attacks",
                "compositional_integrity",
                "integrity_regression",
                "final_gate",
            }
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
            "profile v2 remains author-defined and lacks independent content validation",
            "SHACL validates graph structure rather than scientific truth",
            "integrity tests remain conditional on declared invariants",
            "coherent forgery remains outside the trust model",
            "no new biological dataset or outcome is introduced by this revision",
        ],
        "interpretation": protocol["interpretation"],
    }


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
        "analysis": "profile_v2_paper_code_result_consistency_release",
        "protocol": str(protocol_path),
        "protocol_sha256": hashlib.sha256(protocol_path.read_bytes()).hexdigest(),
        **assessment,
        "decision": (
            "profile_v2_consistency_release_complete"
            if assessment["all_release_conditions_passed"]
            else "profile_v2_consistency_release_incomplete"
        ),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n")
    lines = [
        "# Profile v2 paper-code-result consistency release",
        "",
        f"- Decision: `{payload['decision']}`",
        "",
        "| Condition | Passed |",
        "|---|---:|",
        *(f"| `{name}` | {str(value).lower()} |" for name, value in payload["conditions"].items()),
        "",
        "## Remaining limits",
        "",
        *(f"- {item}" for item in payload["remaining_limits"]),
        "",
        payload["interpretation"],
        "",
    ]
    markdown.parent.mkdir(parents=True, exist_ok=True)
    markdown.write_text("\n".join(lines))
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
