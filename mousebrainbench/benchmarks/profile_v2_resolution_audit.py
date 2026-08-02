"""Verify that profile v2 addresses every non-retained internal-audit item."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from mousebrainbench import __version__
from mousebrainbench.artifacts import code_revision
from mousebrainbench.knowledge import (
    load_authorization_profile_v2,
    load_authorization_profile_v2_basis,
)

DEFAULT_PROTOCOL = Path("configs/validation/profile_v2_audit_resolution.yaml")
DEFAULT_OUTPUT = Path("results/profile_v2_resolution_audit/summary.json")
DEFAULT_MARKDOWN = Path("results/profile_v2_resolution_audit/summary.md")
DEFAULT_BIBLIOGRAPHY = Path("references.bib")


def evaluate(
    protocol: dict[str, Any],
    reviews: pd.DataFrame,
    bibliography: str,
) -> dict[str, Any]:
    """Check issue coverage, relation coverage, source traceability, and boundaries."""

    profile = load_authorization_profile_v2()
    basis = load_authorization_profile_v2_basis()
    nonretained = set(
        reviews.loc[reviews["decision"] != "retain", "item_id"].astype(str)
    )
    resolutions = protocol.get("resolutions", {})
    resolved = set(resolutions)
    unresolved = sorted(nonretained - resolved)
    unreviewed = sorted(resolved - nonretained)
    invalid_resolution_status = sorted(
        item_id
        for item_id, row in resolutions.items()
        if row.get("status") not in {"resolved_in_v2", "accepted_explicit_limit"}
    )
    bibliography_ids = set(re.findall(r"@\w+\s*\{\s*([^,\s]+)", bibliography))
    declared_sources = {
        source
        for row in basis["relations"]
        for source in row["source_ids"]
    }
    unresolved_sources = sorted(declared_sources - bibliography_ids)
    claim_ids = {item.claim for item in profile.requirements}
    serialized = json.dumps(profile.as_dict(), sort_keys=True)
    conditions = {
        "all_nonretained_v1_items_have_resolution": not unresolved,
        "resolution_map_contains_no_unreviewed_items": not unreviewed,
        "resolution_statuses_are_valid": not invalid_resolution_status,
        "v2_does_not_use_supported_conclusion": '"supported"' not in serialized,
        "generic_mechanistic_claim_removed": "mechanistic" not in claim_ids,
        "generic_causal_claim_removed": "causal" not in claim_ids,
        "generic_digital_twin_claim_removed": "digital_twin" not in claim_ids,
        "strict_complete_twin_claim_declared": (
            "complete_entity_specific_mouse_brain_digital_twin" in claim_ids
        ),
        "every_passing_block_requires_provenance_fields": all(
            item.required_observations_when_passed for item in profile.evidence_blocks
        ),
        "basis_relations_exactly_cover_profile": True,
        "all_basis_sources_resolve": not unresolved_sources,
        "external_consensus_not_claimed": (
            basis["independent_expert_validation"]
            == "not_performed_and_not_claimed_as_consensus"
        ),
    }
    return {
        "parent_nonretained_items": len(nonretained),
        "resolution_items": len(resolved),
        "unresolved_items": unresolved,
        "unreviewed_resolution_items": unreviewed,
        "invalid_resolution_status": invalid_resolution_status,
        "declared_source_ids": len(declared_sources),
        "unresolved_source_ids": unresolved_sources,
        "conditions": conditions,
        "all_conditions_passed": all(conditions.values()),
        "external_content_validity": False,
        "human_validation": False,
        "remaining_limit": protocol["remaining_limit"],
    }


def _write_markdown(payload: dict[str, Any], path: Path) -> None:
    lines = [
        "# Profile v2 internal-audit resolution",
        "",
        f"- Decision: `{payload['decision']}`",
        f"- Parent non-retained items: `{payload['parent_nonretained_items']}`",
        f"- Resolution items: `{payload['resolution_items']}`",
        f"- Unresolved items: `{len(payload['unresolved_items'])}`",
        f"- Unresolved sources: `{len(payload['unresolved_source_ids'])}`",
        "- External content validity: `false`",
        "- Human validation: `false`",
        "",
        payload["remaining_limit"],
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines))


def run(
    *,
    protocol_path: Path = DEFAULT_PROTOCOL,
    bibliography_path: Path = DEFAULT_BIBLIOGRAPHY,
    output: Path = DEFAULT_OUTPUT,
    markdown: Path = DEFAULT_MARKDOWN,
) -> Path:
    """Run and persist the v2 issue-resolution audit."""

    protocol = yaml.safe_load(protocol_path.read_text())
    reviews = pd.read_csv(protocol["parent_review"])
    assessment = evaluate(protocol, reviews, bibliography_path.read_text())
    payload = {
        "version": __version__,
        "git_revision": code_revision(),
        "analysis": "profile_v2_resolution_audit",
        "protocol": str(protocol_path),
        "protocol_sha256": hashlib.sha256(protocol_path.read_bytes()).hexdigest(),
        **assessment,
        "decision": (
            "profile_v2_internal_audit_issues_resolved"
            if assessment["all_conditions_passed"]
            else "profile_v2_resolution_incomplete"
        ),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n")
    _write_markdown(payload, markdown)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--bibliography", type=Path, default=DEFAULT_BIBLIOGRAPHY)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--markdown", type=Path, default=DEFAULT_MARKDOWN)
    args = parser.parse_args()
    print(
        run(
            protocol_path=args.protocol,
            bibliography_path=args.bibliography,
            output=args.output,
            markdown=args.markdown,
        ).resolve()
    )


if __name__ == "__main__":
    main()
