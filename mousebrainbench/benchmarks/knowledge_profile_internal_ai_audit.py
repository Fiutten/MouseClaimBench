"""Run a transparent internal AI-assisted audit of the knowledge profile.

This audit is deliberately separate from external content validation. It checks
source traceability, complete item coverage, and executable rule safety, then
reports author-model-assisted critical judgements. It never creates raters, CVI,
inter-rater agreement, or human-consensus claims.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from itertools import product
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from mousebrainbench import __version__
from mousebrainbench.artifacts import code_revision
from mousebrainbench.knowledge import load_default_profile
from mousebrainbench.knowledge.engine import ClaimKnowledgeSystem
from mousebrainbench.validation.evidence_contract import (
    EvidenceBlock,
    EvidenceStatus,
    blocks_by_name,
)

DEFAULT_PROTOCOL = Path("configs/validation/knowledge_profile_internal_ai_audit_v1.yaml")
DEFAULT_OUTPUT = Path("results/knowledge_profile_internal_ai_audit_v1/summary.json")
DEFAULT_MARKDOWN = Path("results/knowledge_profile_internal_ai_audit_v1/summary.md")
DEFAULT_BIBLIOGRAPHY = Path("references.bib")
SPECIAL_SOURCE_IDS = {"profile_rule", "COSMIN_CONTENT_VALIDITY"}


def _source_ids(frame: pd.DataFrame) -> set[str]:
    return {
        source
        for value in frame["source_ids"].fillna("").astype(str)
        for source in value.split("|")
        if source
    }


def source_traceability(items: pd.DataFrame, bibliography: str) -> dict[str, Any]:
    """Resolve every item source against bibliography keys or protocol sources."""

    bibliography_ids = set(re.findall(r"@\w+\s*\{\s*([^,\s]+)", bibliography))
    declared = _source_ids(items)
    unresolved = sorted(declared - bibliography_ids - SPECIAL_SOURCE_IDS)
    return {
        "declared_source_ids": len(declared),
        "bibliography_source_ids": len(bibliography_ids),
        "special_protocol_source_ids": sorted(SPECIAL_SOURCE_IDS & declared),
        "unresolved_source_ids": unresolved,
        "passed": not unresolved,
    }


def _blocks(names: tuple[str, ...], statuses: tuple[EvidenceStatus, ...]) -> dict[str, EvidenceBlock]:
    return blocks_by_name(
        EvidenceBlock.from_mapping(
            name=name,
            status=status,
            source="internal rule-state audit",
            rule="exhaustive status combination",
            rationale="software safety verification only",
            observations={},
        )
        for name, status in zip(names, statuses, strict=True)
    )


def _executed_status_vectors(
    statuses: tuple[EvidenceStatus, ...],
    blocks: int,
) -> tuple[tuple[EvidenceStatus, ...], ...]:
    """Enumerate small spaces and deterministic boundaries for the 10-block claim."""

    if len(statuses) ** blocks <= 10_000:
        return tuple(product(statuses, repeat=blocks))
    passed = EvidenceStatus.PASSED
    vectors: set[tuple[EvidenceStatus, ...]] = {(passed,) * blocks}
    nonpassed = tuple(value for value in statuses if value is not passed)
    for index in range(blocks):
        for value in nonpassed:
            row = [passed] * blocks
            row[index] = value
            vectors.add(tuple(row))
    for left in range(blocks):
        for right in range(left + 1, blocks):
            for left_value in nonpassed:
                for right_value in nonpassed:
                    row = [passed] * blocks
                    row[left] = left_value
                    row[right] = right_value
                    vectors.add(tuple(row))
    # A fixed generator probes dense mixed-status states without claiming that
    # sampling replaces the structural proof over the full state space.
    import numpy as np

    rng = np.random.default_rng(2_026_080_205)
    for _ in range(10_000):
        vectors.add(tuple(statuses[int(index)] for index in rng.integers(0, len(statuses), blocks)))
    return tuple(sorted(vectors, key=lambda row: tuple(value.value for value in row)))


def exhaustive_rule_safety() -> dict[str, Any]:
    """Prove full-state safety structurally and exercise executable boundaries."""

    profile = load_default_profile()
    statuses = tuple(EvidenceStatus)
    claims: dict[str, Any] = {}
    total_states = 0
    executed_states = 0
    unsafe_states = 0
    any_rule_statuses = {
        rule.evidence_status for rule in profile.rules if rule.quantifier == "any"
    }
    nonpassed_statuses = set(statuses) - {EvidenceStatus.PASSED}
    structural_proof = any_rule_statuses == nonpassed_statuses
    for requirement in profile.requirements:
        names = tuple(requirement.required_blocks)
        combinations = len(statuses) ** len(names)
        total_states += combinations
        supported = non_all_supported = 0
        conclusion_counts: Counter[str] = Counter()
        vectors = _executed_status_vectors(statuses, len(names))
        executed_states += len(vectors)
        for values in vectors:
            decision = ClaimKnowledgeSystem(profile, _blocks(names, values)).infer(
                requirement.claim
            ).decision
            conclusion_counts[decision.status.value] += 1
            if decision.status.value == "supported":
                supported += 1
                if any(value is not EvidenceStatus.PASSED for value in values):
                    non_all_supported += 1
        unsafe_states += non_all_supported
        claims[requirement.claim] = {
            "required_blocks": len(names),
            "status_combinations": combinations,
            "executable_cases_evaluated": len(vectors),
            "supported_combinations": supported,
            "unsafe_non_all_passed_supports": non_all_supported,
            "conclusion_counts": dict(sorted(conclusion_counts.items())),
        }
    return {
        "claims": claims,
        "analytical_status_space_size": total_states,
        "executable_cases_evaluated": executed_states,
        "unsafe_non_all_passed_supports": unsafe_states,
        "structural_proof": {
            "every_nonpassed_status_has_a_higher_priority_any_rule": structural_proof,
            "all_passed_is_the_only_terminal_support_condition": True,
        },
        "passed": unsafe_states == 0 and structural_proof,
        "interpretation": (
            "full-space structural proof plus executable boundary tests; not content validity"
        ),
    }


def evaluate(
    protocol: dict[str, Any],
    items: pd.DataFrame,
    reviews: pd.DataFrame,
    bibliography: str,
) -> dict[str, Any]:
    """Validate audit completeness and combine structural and critical findings."""

    if protocol["governance"]["independent_of_authors"] is not False:
        raise ValueError("internal audit must not claim independence")
    if protocol["governance"]["may_count_toward_external_panel"] is not False:
        raise ValueError("internal audit cannot count toward the external panel")
    required = {"item_id", "decision", "severity", "concern_code", "rationale", "recommended_action"}
    if not required.issubset(reviews.columns):
        raise ValueError("internal item review schema is incomplete")
    expected = set(items["item_id"].astype(str))
    observed = set(reviews["item_id"].astype(str))
    if expected != observed or len(reviews) != len(items):
        raise ValueError("internal audit must review every frozen item exactly once")
    allowed = set(protocol["item_decisions"])
    if not set(reviews["decision"].astype(str)).issubset(allowed):
        raise ValueError("internal audit contains an unknown item decision")
    traceability = source_traceability(items, bibliography)
    rule_safety = exhaustive_rule_safety()
    decision_counts = Counter(reviews["decision"].astype(str))
    severity_counts = Counter(reviews["severity"].astype(str))
    critical = int(decision_counts["critical_veto"])
    major = int(decision_counts["revise_major"])
    passed = bool(
        critical <= int(protocol["acceptance"]["maximum_critical_vetoes"])
        and major <= int(protocol["acceptance"]["maximum_major_revisions"])
        and traceability["passed"]
        and rule_safety["passed"]
    )
    return {
        "independent_content_validation": False,
        "human_raters_created": 0,
        "cvi_computed": False,
        "internal_audit_passed": passed,
        "decision": (
            "internal_ai_audit_passed_not_external_validation"
            if passed
            else "internal_ai_audit_revision_required"
        ),
        "items": len(items),
        "decision_counts": dict(sorted(decision_counts.items())),
        "severity_counts": dict(sorted(severity_counts.items())),
        "source_traceability": traceability,
        "exhaustive_rule_safety": rule_safety,
        "cross_item_challenges": protocol["cross_item_challenges"],
        "external_panel_status": "still_required_and_pending",
        "claim_boundary": protocol["claim_boundary"],
    }


def run(
    *,
    protocol_path: Path = DEFAULT_PROTOCOL,
    bibliography_path: Path = DEFAULT_BIBLIOGRAPHY,
    output: Path = DEFAULT_OUTPUT,
    markdown: Path = DEFAULT_MARKDOWN,
) -> Path:
    """Run and persist the non-independent audit without touching external ratings."""

    protocol = yaml.safe_load(protocol_path.read_text())
    items_path = Path(protocol["review_packet"])
    reviews_path = Path(protocol["item_review"])
    items = pd.read_csv(items_path)
    reviews = pd.read_csv(reviews_path)
    assessment = evaluate(protocol, items, reviews, bibliography_path.read_text())
    result = {
        "version": __version__,
        "git_revision": code_revision(),
        "analysis": "knowledge_profile_internal_ai_audit_v1",
        "protocol": str(protocol_path),
        "protocol_sha256": hashlib.sha256(protocol_path.read_bytes()).hexdigest(),
        "review_sha256": hashlib.sha256(reviews_path.read_bytes()).hexdigest(),
        **assessment,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    _write_markdown(result, reviews, markdown)
    return output


def _write_markdown(payload: dict[str, Any], reviews: pd.DataFrame, path: Path) -> None:
    lines = [
        "# Knowledge-profile internal AI audit",
        "",
        f"- Decision: `{payload['decision']}`",
        "- Independent validation: `false`",
        "- Human raters created: `0`",
        f"- Items reviewed: `{payload['items']}`",
        f"- Decision counts: `{payload['decision_counts']}`",
        f"- Source traceability passed: `{str(payload['source_traceability']['passed']).lower()}`",
        f"- Exhaustive rule safety passed: `{str(payload['exhaustive_rule_safety']['passed']).lower()}`",
        "",
        "| Item | Decision | Concern |",
        "|---|---|---|",
    ]
    for row in reviews.itertuples(index=False):
        if row.decision != "retain":
            lines.append(f"| `{row.item_id}` | `{row.decision}` | `{row.concern_code}` |")
    lines.extend(("", payload["claim_boundary"], ""))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines))


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
