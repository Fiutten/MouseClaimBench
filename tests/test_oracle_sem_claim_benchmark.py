import json

from mousebrainbench.benchmarks.oracle_sem_claim_benchmark import run


def test_oracle_benchmark_uses_dgp_labels_and_reports_standard_error_rates(tmp_path) -> None:
    output = run(
        output=tmp_path / "summary.json",
        markdown=tmp_path / "summary.md",
        seeds=3,
        n_per_cohort=300,
    )
    payload = json.loads(output.read_text())

    assert payload["reference_label_source"].startswith("declared structural equations")
    assert payload["evidence_rule_source"].startswith("finite-sample diagnostics")
    assert payload["num_cases"] == 15
    for row in payload["aggregate_by_policy"]:
        assert "false_positive_rate" in row
        assert len(row["false_positive_rate_wilson_95"]) == 2
        assert "false_negative_rate" in row
        assert len(row["false_negative_rate_wilson_95"]) == 2
        assert "overclaiming_risk_index" not in row
    assert payload["case_level_policy_comparison"]["non_tied_cases"] > 0
    assert payload["case_level_policy_comparison"]["unit"].startswith("independently generated")


def test_oracle_benchmark_is_deterministic_for_fixed_seed_range(tmp_path) -> None:
    first = run(
        output=tmp_path / "first.json",
        markdown=tmp_path / "first.md",
        seeds=2,
        n_per_cohort=250,
    )
    second = run(
        output=tmp_path / "second.json",
        markdown=tmp_path / "second.md",
        seeds=2,
        n_per_cohort=250,
    )
    first_payload = json.loads(first.read_text())
    second_payload = json.loads(second.read_text())

    assert first_payload["aggregate_by_policy"] == second_payload["aggregate_by_policy"]
    assert first_payload["by_regime"] == second_payload["by_regime"]
    assert first_payload["case_level_policy_comparison"] == second_payload[
        "case_level_policy_comparison"
    ]
