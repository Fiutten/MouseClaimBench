import json
from pathlib import Path

from mousebrainbench.benchmarks.semantic_equivalence_audit import run


def test_semantic_equivalence_audit_has_no_engine_mismatch(tmp_path: Path) -> None:
    output = tmp_path / "summary.json"
    run(output=output, markdown=tmp_path / "summary.md")
    payload = json.loads(output.read_text())

    assert payload["decision"] == "semantic_equivalence_observed"
    assert payload["mismatch_count"] == 0
    assert payload["evaluated_case_count"] >= 1800
    assert next(
        row for row in payload["claim_audits"] if row["claim"] == "mechanistic"
    )["case_count"] == 625
