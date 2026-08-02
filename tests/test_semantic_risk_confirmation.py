import json

from mousebrainbench.benchmarks.semantic_risk_confirmation import run


def test_smoke_confirmation_uses_frozen_models_and_enforces_semantics(tmp_path) -> None:
    output = tmp_path / "summary.json"
    run(
        output=output,
        markdown=tmp_path / "summary.md",
        cases_path=tmp_path / "cases.npz",
        workers=1,
        test_mode=True,
    )
    payload = json.loads(output.read_text())
    semantic = next(
        row
        for row in payload["aggregate_by_policy"]
        if row["policy"] == "semantic_MAPIE_risk_control"
    )

    assert payload["cases"] == 16
    assert payload["score_model_refitted"] is False
    assert payload["risk_policy_recalibrated"] is False
    assert payload["scale_matches_frozen_protocol"] is False
    assert semantic["semantic_support_violations"] == 0
    assert payload["case_artifact_sha256"].startswith("sha256:")
