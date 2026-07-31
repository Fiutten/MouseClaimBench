import json

from mousebrainbench.benchmarks.mis2_synthetic_calibration import run_calibration


def test_mis2_synthetic_calibration_rejects_nonmechanistic_cases(tmp_path) -> None:
    output = run_calibration(
        output=tmp_path / "summary.json",
        markdown=tmp_path / "summary.md",
        seeds=(100, 101, 102),
    )
    payload = json.loads(output.read_text())
    summary = payload["summary"]

    assert payload["decision"] == "mis2_nominal_synthetic_suite_passed"
    assert summary["false_positive_count"] == 0
    assert summary["false_positive_rate"] == 0

    by_scenario = {row["scenario"]: row for row in summary["by_scenario"]}
    assert by_scenario["clean_directed_truth"]["mis_pass_rate"] == 1.0
    assert 0.0 <= by_scenario["low_snr_directed_truth"]["mis_pass_rate"] < 1.0
    assert by_scenario["common_drive_high_reproducibility"]["mis_pass_rate"] == 0.0
    assert by_scenario["topology_without_direction"]["block_pass_rates"]["topology_specificity"] == 1.0
    assert by_scenario["topology_without_direction"]["block_pass_rates"]["directed_identifiability"] == 0.0
    assert by_scenario["direction_without_topology"]["block_pass_rates"]["directed_identifiability"] == 1.0
    assert by_scenario["direction_without_topology"]["block_pass_rates"]["topology_specificity"] == 0.0
