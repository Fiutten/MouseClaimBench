"""Audit profile-v2 knowledge acquisition and predicate traceability.

This audit answers a narrower question than expert content validation. It checks
that every executable claim-to-evidence relation has a stable identifier and a
claim-specific necessity rationale, and that every evidence block declares the
boundary between its upstream scientific predicate and the authorization
engine. It deliberately keeps external consensus false.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

import yaml

from mousebrainbench import __version__
from mousebrainbench.artifacts import code_revision
from mousebrainbench.knowledge import (
    load_authorization_profile_v2,
    load_authorization_profile_v2_basis,
)

DEFAULT_PROTOCOL = Path("configs/validation/profile_v2_traceability.yaml")
DEFAULT_OUTPUT = Path("results/profile_v2_traceability/summary.json")
DEFAULT_MARKDOWN = Path("results/profile_v2_traceability/summary.md")
DEFAULT_RELATIONS = Path("results/profile_v2_traceability/relations.csv")
DEFAULT_PREDICATES = Path("results/profile_v2_traceability/predicates.csv")


def _bibliography_ids(text: str) -> set[str]:
    return set(re.findall(r"@\w+\s*\{\s*([^,\s]+)", text))


def evaluate(protocol: dict[str, Any], bibliography: str) -> dict[str, Any]:
    """Check exact traceability coverage without inferring scientific consensus."""

    profile = load_authorization_profile_v2()
    basis = load_authorization_profile_v2_basis()
    relations = basis["relations"]
    justifications = basis["relation_justifications"]
    predicates = basis["predicate_contracts"]
    acquisition = basis["knowledge_acquisition"]

    relation_keys = {(str(row["claim"]), str(row["evidence_block"])) for row in relations}
    justification_keys = {(str(row["claim"]), str(row["evidence_block"])) for row in justifications}
    executable_keys = {
        (requirement.claim, block)
        for requirement in profile.requirements
        for block in requirement.required_blocks
    }
    source_ids = {source for row in relations for source in row["source_ids"]} | {
        source for row in predicates.values() for source in row["source_ids"]
    }
    unresolved_sources = sorted(source_ids - _bibliography_ids(bibliography))
    required_steps = set(protocol["required_acquisition_steps"])
    observed_steps = set(acquisition["steps"])
    relation_ids = [str(row["relation_id"]) for row in justifications]
    rationales = [str(row["necessity_rationale"]).strip() for row in justifications]
    block_ids = {item.name for item in profile.evidence_blocks}
    predicate_ids = [str(row["predicate_id"]) for row in predicates.values()]
    observation_slots = sum(
        len(item.required_observations_when_passed) for item in profile.evidence_blocks
    )
    conditions = {
        "acquisition_steps_complete": required_steps <= observed_steps,
        "relations_exactly_cover_profile": relation_keys == executable_keys,
        "justifications_exactly_cover_relations": justification_keys == executable_keys,
        "relation_identifiers_unique": len(relation_ids) == len(set(relation_ids)),
        "necessity_rationales_present": all(rationales),
        "necessity_rationales_relation_specific": (len(set(rationales)) == len(rationales)),
        "predicate_contracts_exactly_cover_blocks": set(predicates) == block_ids,
        "predicate_identifiers_unique": len(predicate_ids) == len(set(predicate_ids)),
        "predicate_execution_boundary_declared": all(
            row.get("evaluation_owner") and row.get("decision_rule_scope")
            for row in predicates.values()
        ),
        "all_sources_resolve_to_bibliography": not unresolved_sources,
        "external_consensus_not_claimed": (
            acquisition["independent_content_validation"] == "not_performed"
            and basis["independent_expert_validation"]
            == "not_performed_and_not_claimed_as_consensus"
        ),
    }
    return {
        "profile_id": profile.profile_id,
        "profile_version": profile.version,
        "profile_hash": profile.source_hash,
        "claims": len(profile.requirements),
        "evidence_blocks": len(profile.evidence_blocks),
        "claim_to_evidence_relations": len(executable_keys),
        "claim_specific_justifications": len(justifications),
        "unique_necessity_rationales": len(set(rationales)),
        "predicate_contracts": len(predicates),
        "required_observation_slots": observation_slots,
        "declared_source_ids": len(source_ids),
        "unresolved_source_ids": unresolved_sources,
        "predicate_execution_owners": dict(
            sorted(Counter(row["evaluation_owner"] for row in predicates.values()).items())
        ),
        "conditions": conditions,
        "all_conditions_passed": all(conditions.values()),
        "independent_content_validity": False,
        "human_validation": False,
        "claim_boundary": protocol["claim_boundary"],
    }


def _write_relations(path: Path) -> None:
    basis = load_authorization_profile_v2_basis()
    semantic = {(row["claim"], row["evidence_block"]): row for row in basis["relations"]}
    fieldnames = (
        "relation_id",
        "claim",
        "evidence_block",
        "knowledge_status",
        "consensus_status",
        "necessity_rationale",
        "block_rationale",
        "scope",
        "exceptions",
        "source_ids",
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in basis["relation_justifications"]:
            source = semantic[(row["claim"], row["evidence_block"])]
            writer.writerow(
                {
                    **{name: row.get(name, "") for name in fieldnames},
                    "block_rationale": source["rationale"],
                    "scope": source["scope"],
                    "exceptions": source["exceptions"],
                    "source_ids": "|".join(source["source_ids"]),
                }
            )


def _write_predicates(path: Path) -> None:
    profile = load_authorization_profile_v2()
    basis = load_authorization_profile_v2_basis()
    required_fields = {
        item.name: "|".join(item.required_observations_when_passed)
        for item in profile.evidence_blocks
    }
    fieldnames = (
        "predicate_id",
        "evidence_block",
        "evaluation_owner",
        "decision_rule_scope",
        "required_observations",
        "rationale",
        "scope",
        "exceptions",
        "source_ids",
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for name, row in basis["predicate_contracts"].items():
            writer.writerow(
                {
                    "predicate_id": row["predicate_id"],
                    "evidence_block": name,
                    "evaluation_owner": row["evaluation_owner"],
                    "decision_rule_scope": row["decision_rule_scope"],
                    "required_observations": required_fields[name],
                    "rationale": row["rationale"],
                    "scope": row["scope"],
                    "exceptions": row["exceptions"],
                    "source_ids": "|".join(row["source_ids"]),
                }
            )


def _write_markdown(payload: dict[str, Any], path: Path) -> None:
    lines = [
        "# Profile v2 knowledge traceability",
        "",
        f"- Decision: `{payload['decision']}`",
        f"- Claims: `{payload['claims']}`",
        f"- Evidence blocks and predicate contracts: `{payload['evidence_blocks']}`",
        f"- Claim-to-evidence relations: `{payload['claim_to_evidence_relations']}`",
        f"- Relation-specific rationales: `{payload['unique_necessity_rationales']}`",
        f"- Required observation slots: `{payload['required_observation_slots']}`",
        f"- Bibliographic source identifiers: `{payload['declared_source_ids']}`",
        "- Independent content validity: `false`",
        "",
        "| Condition | Passed |",
        "|---|---:|",
    ]
    lines.extend(
        f"| `{name}` | {str(passed).lower()} |" for name, passed in payload["conditions"].items()
    )
    lines.extend(("", payload["claim_boundary"], ""))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines))


def run(
    *,
    protocol_path: Path = DEFAULT_PROTOCOL,
    output: Path = DEFAULT_OUTPUT,
    markdown: Path = DEFAULT_MARKDOWN,
    relations: Path = DEFAULT_RELATIONS,
    predicates: Path = DEFAULT_PREDICATES,
) -> Path:
    protocol = yaml.safe_load(protocol_path.read_text())
    bibliography_path = Path(protocol["bibliography"])
    assessment = evaluate(protocol, bibliography_path.read_text())
    payload = {
        "version": __version__,
        "git_revision": code_revision(),
        "analysis": "profile_v2_knowledge_traceability_audit",
        "protocol": str(protocol_path),
        "protocol_sha256": hashlib.sha256(protocol_path.read_bytes()).hexdigest(),
        "basis_sha256": hashlib.sha256(Path(protocol["basis"]).read_bytes()).hexdigest(),
        **assessment,
        "decision": (
            "profile_v2_traceability_complete_author_policy_only"
            if assessment["all_conditions_passed"]
            else "profile_v2_traceability_incomplete"
        ),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n")
    _write_relations(relations)
    _write_predicates(predicates)
    _write_markdown(payload, markdown)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--markdown", type=Path, default=DEFAULT_MARKDOWN)
    parser.add_argument("--relations", type=Path, default=DEFAULT_RELATIONS)
    parser.add_argument("--predicates", type=Path, default=DEFAULT_PREDICATES)
    args = parser.parse_args()
    print(
        run(
            protocol_path=args.protocol,
            output=args.output,
            markdown=args.markdown,
            relations=args.relations,
            predicates=args.predicates,
        ).resolve()
    )


if __name__ == "__main__":
    main()
