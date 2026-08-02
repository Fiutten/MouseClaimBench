import json

from mousebrainbench.benchmarks.hybrid_selective_release_audit import run


def test_release_audit_preserves_negative_publication_boundary(tmp_path):
    output = run(output=tmp_path / "summary.json", markdown=tmp_path / "summary.md")
    payload = json.loads(output.read_text())
    assert payload["passed"] is True
    assert payload["q1_ready"] is False
    assert all(payload["checks"].values())
    assert payload["decision"] == "hybrid_v2_reproducible_negative_release"
