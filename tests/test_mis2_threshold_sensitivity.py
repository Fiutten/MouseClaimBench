import json

from mousebrainbench.benchmarks.mis2_threshold_sensitivity import run_sensitivity


def test_mis2_threshold_sensitivity_maps_safe_and_conservative_regions(tmp_path) -> None:
    output = run_sensitivity(
        output=tmp_path / "summary.json",
        markdown=tmp_path / "summary.md",
        seeds=(200, 201),
    )
    payload = json.loads(output.read_text())

    assert payload["decision"] == "mis2_sensitivity_supports_conservative_gate"
    assert payload["phase_counts"]["dangerous"] == 0
    assert payload["phase_counts"]["unstable"] == 0
    assert payload["phase_counts"]["safe"] > 0
    assert payload["phase_counts"]["conservative"] > 0

    nominal_low_noise = [
        row
        for row in payload["rows"]
        if row["profile"] == "nominal" and row["noise"] == 0.08 and row["n_sessions"] == 24
    ][0]
    assert nominal_low_noise["phase"] == "safe"
    assert nominal_low_noise["false_positive_rate"] == 0

    nominal_high_noise = [
        row
        for row in payload["rows"]
        if row["profile"] == "nominal" and row["noise"] == 0.60 and row["n_sessions"] == 6
    ][0]
    assert nominal_high_noise["phase"] in {"safe", "conservative"}
    assert nominal_high_noise["false_positive_rate"] == 0
