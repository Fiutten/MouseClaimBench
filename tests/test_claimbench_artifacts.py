import json
import subprocess
from pathlib import Path

from mousebrainbench.benchmarks.claim_ledger import run as run_claim_ledger
from mousebrainbench.benchmarks.claimbench_release import run as run_release
from mousebrainbench.benchmarks.real_case_claim_matrix import run as run_real_cases


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload))


def _minimal_artifacts(root: Path) -> None:
    _write_json(
        root / "results/allen_vbn_mechanistic_identifiability_score.json",
        {"decision": "reproducible_target_without_mechanistic_identifiability", "git_revision": "abc"},
    )
    _write_json(
        root / "results/sensorium_static_model_comparator/summary.json",
        {
            "comparison": "sensorium2022_static_cross_dataset_comparator",
            "topographic_constraint": "results/sensorium_topographic_constraint/summary_static_test.json",
            "git_revision": "abc",
        },
    )
    _write_json(
        root / "results/dynamic_sensorium_model_comparator/summary.json",
        {"comparison": "dynamic_sensorium_predictive_model_comparator", "git_revision": "abc"},
    )
    _write_json(
        root / "results/microns_primary_robustness/summary.json",
        {"decision": "microns_primary_endpoint_survives_harder_controls", "git_revision": "abc"},
    )
    _write_json(
        root / "results/sensorium_official_baseline_audit/summary.json",
        {
            "decision": "official_sensorium_bounded_trained_baseline_available_not_q1_qualified",
            "git_revision": "abc",
        },
    )
    _write_json(
        root / "results/claim_adversarial_benchmark/summary.json",
        {"decision": "claim_gate_blocks_broad_adversarial_overclaims", "git_revision": "abc"},
    )
    _write_json(
        root / "results/claim_attack_suite/summary.json",
        {"decision": "claim_attack_suite_passed_with_known_limits", "git_revision": "abc"},
    )
    _write_json(
        root / "results/mis2_synthetic_calibration/summary.json",
        {"decision": "mis2_nominal_synthetic_suite_passed", "git_revision": "abc"},
    )
    _write_json(
        root / "results/mis2_threshold_sensitivity/summary.json",
        {"decision": "mis2_sensitivity_supports_conservative_gate", "git_revision": "abc"},
    )
    (root / "docs").mkdir(exist_ok=True)
    (root / "docs/CLAIMBENCH_SOTA_AND_VIABILITY.md").write_text("SOTA")


def test_real_case_claim_matrix_is_conservative(tmp_path) -> None:
    _minimal_artifacts(tmp_path)

    output = run_real_cases(
        output=tmp_path / "results/real_case_claim_matrix/summary.json",
        markdown=tmp_path / "results/real_case_claim_matrix/summary.md",
        root=tmp_path,
    )
    payload = json.loads(output.read_text())
    aggregate = {row["evaluator"]: row for row in payload["aggregate_by_evaluator"]}

    assert payload["decision"] == "real_case_claim_gate_consistent"
    assert aggregate["claim_gate"]["fp"] == 0
    assert aggregate["correlation_only"]["fp"] > 0


def test_claim_ledger_tracks_supported_and_blocked_wording(tmp_path) -> None:
    _minimal_artifacts(tmp_path)

    output = run_claim_ledger(
        output=tmp_path / "results/claim_ledger/summary.json",
        markdown=tmp_path / "results/claim_ledger/claim_audit_report.md",
        root=tmp_path,
    )
    payload = json.loads(output.read_text())

    assert payload["decision"] == "claim_ledger_supported"
    assert payload["unsupported_claims"] == 0
    assert all(entry["blocked_wording"] for entry in payload["entries"])


def test_claimbench_release_detects_clean_required_artifacts(tmp_path, monkeypatch) -> None:
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    monkeypatch.setenv("MOUSEBRAINBENCH_GIT_REVISION", head)
    _minimal_artifacts(tmp_path)
    run_real_cases(
        output=tmp_path / "results/real_case_claim_matrix/summary.json",
        markdown=tmp_path / "results/real_case_claim_matrix/summary.md",
        root=tmp_path,
    )
    run_claim_ledger(
        output=tmp_path / "results/claim_ledger/summary.json",
        markdown=tmp_path / "results/claim_ledger/claim_audit_report.md",
        root=tmp_path,
    )

    output = run_release(
        output=tmp_path / "results/claimbench_release/summary.json",
        markdown=tmp_path / "results/claimbench_release/summary.md",
        root=tmp_path,
    )
    payload = json.loads(output.read_text())

    assert payload["decision"] == "claimbench_release_ready"
    assert payload["missing_artifacts"] == []
    assert payload["dirty_artifacts"] == []
