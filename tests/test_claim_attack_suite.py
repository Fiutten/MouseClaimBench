import json

from mousebrainbench.benchmarks.claim_adversarial import run as run_adversarial
from mousebrainbench.benchmarks.claim_attack_suite import run


def test_claim_attack_suite_reports_known_nonblocking_limits(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / "results/claim_adversarial_benchmark").mkdir(parents=True)
    run_adversarial(
        output=tmp_path / "results/claim_adversarial_benchmark/summary.json",
        markdown=tmp_path / "results/claim_adversarial_benchmark/summary.md",
    )
    (tmp_path / "results/mis2_threshold_sensitivity").mkdir(parents=True)
    (tmp_path / "results/mis2_threshold_sensitivity/summary.json").write_text(
        json.dumps({"phase_counts": {"dangerous": 0, "unstable": 0}})
    )
    (tmp_path / "results/publication_freeze").mkdir(parents=True)
    (tmp_path / "results/publication_freeze/summary.json").write_text(
        json.dumps({"claims_allowed": ["bounded claim"], "claims_blocked": ["blocked claim"]})
    )
    (tmp_path / "results/digital_twin_claim_audit").mkdir(parents=True)
    (tmp_path / "results/digital_twin_claim_audit/summary.json").write_text(
        json.dumps({"blocked_claims": ["whole_brain_digital_twin"]})
    )
    (tmp_path / "results/microns_q1_package").mkdir(parents=True)
    (tmp_path / "results/microns_q1_package/summary.json").write_text(
        json.dumps({"q1_package_ready": True})
    )
    (tmp_path / "results/microns_primary_robustness").mkdir(parents=True)
    (tmp_path / "results/microns_primary_robustness/summary.json").write_text(
        json.dumps({"all_cohorts_robust": True})
    )
    (tmp_path / "results/sensorium_official_baseline_audit").mkdir(parents=True)
    (tmp_path / "results/sensorium_official_baseline_audit/summary.json").write_text(
        json.dumps({"official_q1_baseline_qualified": False})
    )
    (tmp_path / "results").mkdir(exist_ok=True)
    (tmp_path / "results/allen_vbn_mechanistic_identifiability_score.json").write_text(
        json.dumps({"decision": "reproducible_target_without_mechanistic_identifiability"})
    )

    output = run(output=tmp_path / "summary.json", markdown=tmp_path / "summary.md")
    payload = json.loads(output.read_text())

    assert payload["decision"] == "claim_attack_suite_passed_with_known_limits"
    assert [risk["level"] for risk in payload["risks"]] == ["medium"]
