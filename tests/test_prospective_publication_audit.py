import json

from mousebrainbench.benchmarks.prospective_publication_audit import run


def test_publication_audit_blocks_superiority_claim_without_hiding_positive_microns(
    tmp_path,
) -> None:
    output = run(
        output=tmp_path / "summary.json",
        markdown=tmp_path / "summary.md",
    )
    payload = json.loads(output.read_text())

    assert payload["protocol_integrity"] is True
    assert payload["artifacts_clean"] is True
    assert payload["synthetic_primary_endpoint_passed"] is False
    assert payload["microns_network_endpoint_passed"] is True
    assert payload["dyadic_calibration_passed"] is True
    assert payload["strong_q1_superiority_ready"] is False
    assert payload["decision"] == (
        "strong_q1_superiority_claim_blocked_by_prospective_evidence"
    )
    assert any(
        row["risk"].startswith("The directional diagnostic")
        for row in payload["reviewer_risks"]
    )

