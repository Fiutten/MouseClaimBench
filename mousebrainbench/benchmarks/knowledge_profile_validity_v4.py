"""Audit the evidential status of the author-proposed v4 knowledge profile."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from mousebrainbench import __version__
from mousebrainbench.artifacts import code_revision

DEFAULT_BASIS = Path("mousebrainbench/knowledge/profiles/mouse_brain_claims_v1_basis.yaml")
DEFAULT_PROTOCOL = Path("configs/benchmarks/semantic_risk_control_v4.yaml")
DEFAULT_OUTPUT = Path("results/knowledge_profile_validity_v4/summary.json")


def evaluate(basis: dict, protocol: dict) -> dict:
    relations = list(basis.get("relations", ()))
    required = {"rationale", "scope", "exceptions", "alternatives_rejected", "source_ids"}
    incomplete = [
        f"{row.get('claim')}:{row.get('evidence_block')}"
        for row in relations
        if not required.issubset(row) or not row.get("source_ids")
    ]
    external = list(protocol["profile_validity"]["external_basis"])
    author_extensions = set(protocol["profile_validity"]["author_extensions"])
    relation_blocks = {str(row["evidence_block"]) for row in relations}
    conditions = {
        "profile_marked_author_proposed": str(basis.get("status", "")).startswith(
            "author_proposed"
        ),
        "independent_validation_not_claimed": basis.get("independent_expert_validation")
        == "not_performed",
        "every_relation_has_scope_rationale_exceptions_and_sources": not incomplete,
        "external_standards_have_bounded_roles": all(
            item.get("role") and (item.get("url") or item.get("doi")) for item in external
        ),
        "author_extensions_are_declared": author_extensions.issubset(relation_blocks),
        "consensus_mapping_not_claimed": protocol["profile_validity"]
        ["consensus_standard_claimed"]
        is False,
    }
    return {
        "conditions": conditions,
        "incomplete_relations": incomplete,
        "relations": len(relations),
        "external_bases": external,
        "author_extensions": sorted(author_extensions),
        "profile_content_validated": False,
        "structural_documentation_complete": all(conditions.values()),
        "decision": (
            "author_proposed_profile_transparently_documented_not_content_validated"
            if all(conditions.values())
            else "profile_documentation_incomplete"
        ),
    }


def run(
    *,
    basis: Path = DEFAULT_BASIS,
    protocol: Path = DEFAULT_PROTOCOL,
    output: Path = DEFAULT_OUTPUT,
) -> Path:
    assessment = evaluate(yaml.safe_load(basis.read_text()), yaml.safe_load(protocol.read_text()))
    result = {
        "version": __version__,
        "git_revision": code_revision(),
        "analysis": "knowledge_profile_validity_v4",
        "basis": str(basis),
        "protocol": str(protocol),
        **assessment,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--basis", type=Path, default=DEFAULT_BASIS)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    print(json.dumps({"output": str(run(basis=args.basis, protocol=args.protocol, output=args.output).resolve())}))


if __name__ == "__main__":
    main()
