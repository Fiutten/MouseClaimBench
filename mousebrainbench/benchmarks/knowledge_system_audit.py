"""Audit the knowledge system against author-generated migration decisions.

The audit is a conformance test, not new biological evidence. It verifies that
the versioned knowledge profile reproduces decisions created by the authors
during system migration while adding an explicit fired rule, evidence
provenance, and a machine-readable inference trace to every decision. These
reference decisions are not independent scientific labels.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from mousebrainbench import __version__
from mousebrainbench.artifacts import code_revision
from mousebrainbench.benchmarks.real_case_claim_matrix import build_cases
from mousebrainbench.knowledge import ClaimKnowledgeSystem, load_default_profile
from mousebrainbench.validation.evidence_contract import blocks_by_name


DEFAULT_OUTPUT = Path("results/knowledge_system_audit/summary.json")
DEFAULT_MARKDOWN = Path("results/knowledge_system_audit/summary.md")
FROZEN_MATRIX = Path("results/real_case_claim_matrix/summary.json")


def _decision_key(row: dict[str, Any]) -> tuple[str, str]:
    return str(row["case"]), str(row["claim"])


def _comparable_decision(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "claim": row["claim"],
        "status": row["status"],
        "required_blocks": row["required_blocks"],
        "block_statuses": row["block_statuses"],
        "rationale": row["rationale"],
    }


def _graph_schema(graph: dict[str, Any]) -> dict[str, Any]:
    """Remove case facts while retaining the ontology and relation topology."""

    return {
        "nodes": [(node["id"], node["type"]) for node in graph["nodes"]],
        "edges": graph["edges"],
    }


def run(
    output: Path = DEFAULT_OUTPUT,
    markdown: Path = DEFAULT_MARKDOWN,
    root: Path = Path("."),
) -> Path:
    """Run exact decision-conformance and explanation-completeness checks."""

    frozen_path = root / FROZEN_MATRIX
    if not frozen_path.exists():
        raise FileNotFoundError(f"frozen real-case matrix is missing: {frozen_path}")
    frozen_payload = json.loads(frozen_path.read_text())
    frozen_rows = {
        _decision_key(row): _comparable_decision(row)
        for row in frozen_payload["claim_decisions"]
    }

    profile = load_default_profile()
    inference_rows: list[dict[str, Any]] = []
    conformance_rows: list[dict[str, Any]] = []
    graphs: list[dict[str, Any]] = []
    for case in build_cases(root):
        system = ClaimKnowledgeSystem(profile, blocks_by_name(case.blocks))
        graphs.append(system.knowledge_graph())
        for inference in system.infer_all():
            row = {"case": case.name, **inference.as_dict()}
            inference_rows.append(row)
            key = (case.name, inference.decision.claim)
            actual = _comparable_decision(inference.decision.as_dict())
            expected = frozen_rows.get(key)
            conformance_rows.append(
                {
                    "case": case.name,
                    "claim": inference.decision.claim,
                    "exact_match": actual == expected,
                    "expected": expected,
                    "actual": actual,
                }
            )

    exact_matches = sum(row["exact_match"] for row in conformance_rows)
    explanation_complete = sum(
        bool(row["fired_rule"])
        and bool(row["steps"])
        and bool(row["profile_hash"])
        and len(row["evidence_facts"]) == len(row["decision"]["required_blocks"])
        for row in inference_rows
    )
    rule_counts = Counter(row["fired_rule"] for row in inference_rows)
    status_counts = Counter(row["decision"]["status"] for row in inference_rows)
    expected_count = len(frozen_rows)
    observed_count = len(inference_rows)
    graph = graphs[0]

    passed = (
        expected_count == observed_count == 40
        and exact_matches == observed_count
        and explanation_complete == observed_count
        and all(_graph_schema(graphs[0]) == _graph_schema(candidate) for candidate in graphs[1:])
    )
    payload = {
        "version": __version__,
        "git_revision": code_revision(),
        "analysis": "knowledge_system_conformance_audit_v1",
        "knowledge_profile": profile.as_dict(),
        "frozen_reference": str(FROZEN_MATRIX),
        "case_count": len(graphs),
        "decision_count": observed_count,
        "exact_decision_matches": exact_matches,
        "explanation_complete_count": explanation_complete,
        "status_counts": dict(sorted(status_counts.items())),
        "fired_rule_counts": dict(sorted(rule_counts.items())),
        "knowledge_graph": {
            "node_count": len(graph["nodes"]),
            "edge_count": len(graph["edges"]),
            "node_type_counts": dict(
                sorted(Counter(node["type"] for node in graph["nodes"]).items())
            ),
            "relation_counts": dict(
                sorted(Counter(edge["relation"] for edge in graph["edges"]).items())
            ),
        },
        "conformance_failures": [row for row in conformance_rows if not row["exact_match"]],
        "limits": [
            "Conformance to author-generated migration decisions is not evidence that the policy is biologically true.",
            "The migration decisions were not produced blindly or by independent experts.",
            "The audit evaluates only the computational mouse-brain knowledge profile.",
            "Architectural profile extensibility is not empirical validation in another domain.",
            "The real cases do not establish causal, whole-brain, or digital-twin claims.",
        ],
        "decision": (
            "knowledge_system_reproduces_frozen_policy_with_complete_traces"
            if passed
            else "knowledge_system_conformance_failed"
        ),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2))
    write_markdown(payload, markdown)
    return output


def write_markdown(payload: dict[str, Any], markdown: Path) -> None:
    """Write the reviewer-facing conformance summary."""

    lines = [
        "# Knowledge-System Conformance Audit",
        "",
        f"- Decision: `{payload['decision']}`",
        f"- Knowledge profile: `{payload['knowledge_profile']['profile_id']}` "
        f"v`{payload['knowledge_profile']['version']}`",
        f"- Profile hash: `{payload['knowledge_profile']['source_hash']}`",
        f"- Exact migration-decision matches: `{payload['exact_decision_matches']}/"
        f"{payload['decision_count']}`",
        f"- Complete inference traces: `{payload['explanation_complete_count']}/"
        f"{payload['decision_count']}`",
        f"- Fired rules: `{payload['fired_rule_counts']}`",
        f"- Knowledge graph: `{payload['knowledge_graph']['node_count']}` nodes, "
        f"`{payload['knowledge_graph']['edge_count']}` edges",
        "",
        "## Limits",
        "",
    ]
    lines.extend(f"- {limit}" for limit in payload["limits"])
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
