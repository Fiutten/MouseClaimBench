import json

from mousebrainbench.benchmarks.hybrid_selective_outcome_audit import run


def test_outcome_audit_preserves_negative_primary(tmp_path):
    output = run(output=tmp_path / "audit.json", markdown=tmp_path / "audit.md")
    payload = json.loads(output.read_text())
    assert payload["source_primary_passed"] is False
    assert payload["confirmation_reexecuted"] is False
    assert payload["model_refitted"] is False
    assert payload["thresholds_changed"] is False
    assert payload["publication_assessment"]["strong_new_q1_claim_supported"] is False
    assert len(payload["per_variable_claim"]) == 6
    assert len(payload["direction_failure"]["by_regime"]) == 10
