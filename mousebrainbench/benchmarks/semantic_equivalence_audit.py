"""Audit semantic equivalence between the Python and independent ASP engines.

The audit is deliberately narrower than a formal verification proof. It
exhausts every evidence assignment for claims with at most four required
blocks and exercises homogeneous, single-fault, and pairwise-priority
boundaries for larger contracts. The ASP program is evaluated by Potassco
clingo and compared with the authoritative Python knowledge engine.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from collections.abc import Iterable
from itertools import combinations, product
from pathlib import Path
from typing import Any

from mousebrainbench import __version__
from mousebrainbench.artifacts import code_revision
from mousebrainbench.knowledge import (
    ClaimKnowledgeSystem,
    infer_with_clingo,
    load_default_profile,
)
from mousebrainbench.validation.evidence_contract import EvidenceBlock, EvidenceStatus

DEFAULT_OUTPUT = Path("results/semantic_equivalence_audit/summary.json")
DEFAULT_MARKDOWN = Path("results/semantic_equivalence_audit/summary.md")
EXHAUSTIVE_BLOCK_LIMIT = 4


def _block(name: str, status: EvidenceStatus) -> EvidenceBlock:
    return EvidenceBlock.from_mapping(
        name=name,
        status=status,
        source="semantic-equivalence-audit",
        rule="enumerated evidence-state assignment",
        rationale="exercises independent executable claim semantics",
    )


def _large_contract_assignments(block_names: tuple[str, ...]) -> Iterable[tuple[EvidenceStatus, ...]]:
    """Yield deterministic cases that exercise every rule and priority boundary."""

    statuses = tuple(EvidenceStatus)
    passed = EvidenceStatus.PASSED

    # Homogeneous assignments exercise every terminal disposition directly.
    for status in statuses:
        yield (status,) * len(block_names)

    # Single deviations show that every required block is non-compensatory.
    for block_index in range(len(block_names)):
        for status in statuses:
            assignment = [passed] * len(block_names)
            assignment[block_index] = status
            yield tuple(assignment)

    # Pairwise mixtures exercise rule priority independently of block order.
    for left_index, right_index in combinations(range(len(block_names)), 2):
        for left_status, right_status in product(statuses, repeat=2):
            assignment = [passed] * len(block_names)
            assignment[left_index] = left_status
            assignment[right_index] = right_status
            yield tuple(assignment)


def _assignments(block_names: tuple[str, ...]) -> tuple[str, Iterable[tuple[EvidenceStatus, ...]]]:
    if len(block_names) <= EXHAUSTIVE_BLOCK_LIMIT:
        return "exhaustive", product(EvidenceStatus, repeat=len(block_names))
    return "boundary_and_pairwise_priority", _large_contract_assignments(block_names)


def run(
    output: Path = DEFAULT_OUTPUT,
    markdown: Path = DEFAULT_MARKDOWN,
) -> Path:
    """Compare both inference engines and write a machine-readable audit."""

    profile = load_default_profile()
    claim_rows: list[dict[str, Any]] = []
    mismatches: list[dict[str, Any]] = []
    asp_backends: Counter[str] = Counter()
    unique_program_hashes: set[str] = set()

    for requirement in profile.requirements:
        mode, assignments = _assignments(requirement.required_blocks)
        case_count = 0
        for states in assignments:
            case_count += 1
            blocks = {
                name: _block(name, status)
                for name, status in zip(requirement.required_blocks, states, strict=True)
            }
            python_decision = ClaimKnowledgeSystem(profile, blocks).infer(requirement.claim)
            asp_decision = infer_with_clingo(profile, requirement.claim, blocks)
            asp_backends[asp_decision.backend] += 1
            unique_program_hashes.add(asp_decision.program_hash)
            if python_decision.decision.status is not asp_decision.status:
                mismatches.append(
                    {
                        "claim": requirement.claim,
                        "assignment": {
                            name: status.value
                            for name, status in zip(
                                requirement.required_blocks, states, strict=True
                            )
                        },
                        "python_status": python_decision.decision.status.value,
                        "asp_status": asp_decision.status.value,
                    }
                )
        claim_rows.append(
            {
                "claim": requirement.claim,
                "required_block_count": len(requirement.required_blocks),
                "enumeration_mode": mode,
                "case_count": case_count,
            }
        )

    # Missing blocks and undeclared claims are explicit ontology boundaries.
    boundary_rows = []
    for claim in ("predictive", "conscious"):
        python_status = ClaimKnowledgeSystem(profile, {}).infer(claim).decision.status
        asp_status = infer_with_clingo(profile, claim, {}).status
        boundary_rows.append(
            {
                "claim": claim,
                "python_status": python_status.value,
                "asp_status": asp_status.value,
                "equivalent": python_status is asp_status,
            }
        )

    total_cases = sum(row["case_count"] for row in claim_rows) + len(boundary_rows)
    passed = not mismatches and all(row["equivalent"] for row in boundary_rows)
    payload = {
        "version": __version__,
        "git_revision": code_revision(),
        "analysis": "independent_asp_semantic_equivalence_audit_v1",
        "knowledge_profile_id": profile.profile_id,
        "knowledge_profile_version": profile.version,
        "knowledge_profile_hash": profile.source_hash,
        "asp_backends": dict(sorted(asp_backends.items())),
        "claim_audits": claim_rows,
        "boundary_audits": boundary_rows,
        "evaluated_case_count": total_cases,
        "unique_asp_program_count": len(unique_program_hashes),
        "mismatch_count": len(mismatches),
        "mismatches": mismatches,
        "decision": "semantic_equivalence_observed" if passed else "semantic_equivalence_failed",
        "limits": [
            "This is an executable equivalence audit, not a mathematical proof of correctness.",
            "Claims with at most four required blocks are exhausted over all five evidence states.",
            "Larger contracts use homogeneous, single-block, and pairwise-priority boundary cases.",
            "Agreement between engines does not establish that the curated scientific policy is true.",
        ],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n")
    _write_markdown(payload, markdown)
    return output


def _write_markdown(payload: dict[str, Any], path: Path) -> None:
    lines = [
        "# Independent ASP Semantic Equivalence Audit",
        "",
        f"- Decision: `{payload['decision']}`",
        f"- Evaluated assignments and boundaries: `{payload['evaluated_case_count']}`",
        f"- Mismatches: `{payload['mismatch_count']}`",
        f"- ASP backend: `{payload['asp_backends']}`",
        f"- Knowledge profile hash: `{payload['knowledge_profile_hash']}`",
        "",
        "## Claim coverage",
        "",
    ]
    lines.extend(
        f"- `{row['claim']}`: `{row['case_count']}` cases "
        f"(`{row['enumeration_mode']}`)"
        for row in payload["claim_audits"]
    )
    lines.extend(("", "## Limits", ""))
    lines.extend(f"- {limit}" for limit in payload["limits"])
    lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--markdown", type=Path, default=DEFAULT_MARKDOWN)
    args = parser.parse_args()
    print(json.dumps({"output": str(run(args.output, args.markdown).resolve())}))


if __name__ == "__main__":
    main()
