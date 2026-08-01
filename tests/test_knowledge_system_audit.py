import json
from pathlib import Path

from mousebrainbench.benchmarks.knowledge_system_audit import run


def test_knowledge_system_audit_matches_all_frozen_real_case_decisions(
    tmp_path: Path,
) -> None:
    output = tmp_path / "summary.json"
    markdown = tmp_path / "summary.md"

    run(output=output, markdown=markdown, root=Path("."))
    payload = json.loads(output.read_text())

    assert payload["case_count"] == 4
    assert payload["decision_count"] == 40
    assert payload["exact_decision_matches"] == 40
    assert payload["explanation_complete_count"] == 40
    assert payload["conformance_failures"] == []
    assert payload["decision"] == (
        "knowledge_system_reproduces_frozen_policy_with_complete_traces"
    )
    assert payload["fired_rule_counts"] == {
        "all_requirements_satisfied": 10,
        "failed_block_veto": 7,
        "missing_evidence_uncertainty": 7,
        "protocol_scope_boundary": 16,
    }
    assert markdown.exists()


def test_knowledge_system_audit_exports_a_typed_knowledge_graph(tmp_path: Path) -> None:
    output = tmp_path / "summary.json"

    run(output=output, markdown=tmp_path / "summary.md", root=Path("."))
    graph = json.loads(output.read_text())["knowledge_graph"]

    assert graph["node_type_counts"] == {
        "claim": 10,
        "decision_status": 5,
        "evidence_block": 10,
        "evidence_status": 5,
        "inference_rule": 5,
        "knowledge_profile": 1,
    }
    assert graph["relation_counts"]["requires"] == 20
    assert graph["relation_counts"]["concludes"] == 5
