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
        {
            "decision": "reproducible_target_without_mechanistic_identifiability",
            "git_revision": "abc",
            "mis": {
                "blocks": [
                    {"name": "reproducibility", "passed": True, "score": 1.0, "criteria": []},
                    {
                        "name": "topology_specificity",
                        "passed": False,
                        "score": 0.0,
                        "criteria": [],
                    },
                    {
                        "name": "directed_identifiability",
                        "passed": False,
                        "score": 0.0,
                        "criteria": [],
                    },
                ]
            },
        },
    )
    _write_json(
        root / "results/sensorium_static_model_comparator/summary.json",
        {
            "comparison": "sensorium2022_static_cross_dataset_comparator",
            "pretraining_test_repeated": {
                "median_best_predictive_correlation": 0.34,
                "median_best_minus_mean": 0.09,
                "median_best_minus_scrambled": 0.07,
            },
            "topographic_constraint": {
                "decision": "structural_constraint_supported",
                "passed_count": 5,
                "n_datasets": 5,
                "median_observed_spearman": 0.18,
                "median_effect_over_null": 0.18,
            },
            "git_revision": "abc",
        },
    )
    _write_json(
        root / "results/dynamic_sensorium_model_comparator/summary.json",
        {
            "comparison": "dynamic_sensorium_predictive_model_comparator",
            "git_revision": "abc",
            "cohorts": [
                {
                    "cohort": "cohort_a",
                    "n_mice": 5,
                    "reliability_estimable_count": 0,
                    "pairwise": {
                        "mean_response_vs_temporal_svd": {
                            "n_paired": 5,
                            "right_wins": 4,
                            "median_delta": 0.03,
                        }
                    },
                },
                {
                    "cohort": "cohort_b",
                    "n_mice": 5,
                    "reliability_estimable_count": 0,
                    "pairwise": {
                        "mean_response_vs_temporal_svd": {
                            "n_paired": 5,
                            "right_wins": 5,
                            "median_delta": 0.04,
                        }
                    },
                },
            ],
        },
    )
    _write_json(
        root / "results/microns_primary_robustness/summary.json",
        {
            "decision": "microns_primary_endpoint_survives_harder_controls",
            "all_cohorts_robust": True,
            "git_revision": "abc",
        },
    )
    _write_json(
        root / "results/microns_q1_package/summary.json",
        {
            "git_revision": "abc",
            "primary_endpoint": "all_pairs/readout_location",
            "q1_package_ready": True,
            "cohorts": [
                {
                    "cohort": name,
                    "n_units": 1000,
                    "n_connected_edge_pairs": 2000,
                    "primary_test": {
                        "distance_matched_delta": 0.02,
                        "distance_matched_q_one_sided": 0.01,
                        "degree_matched_delta": 0.03,
                        "degree_matched_q_one_sided": 0.01,
                        "confirmed_positive_after_fdr": True,
                    },
                    "unit_bootstrap": {
                        "distance_matched_delta": {"ci95_low": 0.01, "ci95_high": 0.03},
                        "degree_matched_delta": {"ci95_low": 0.02, "ci95_high": 0.04},
                    },
                }
                for name in ("discovery", "holdout_a", "holdout_b")
            ],
        },
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
    cases = {row["case"]: row for row in payload["cases"]}

    assert payload["decision"] == "artifact_grounded_case_matrix_complete_with_explicit_limits"
    assert payload["forbidden_supported_claims"] == []
    assert cases["allen_vbn_identifiability_negative"]["supported_claims"] == [
        "computationally_reproducible",
        "internally_reproduced",
    ]
    assert "structure_function" in cases["microns_local_structure_function"][
        "supported_claims"
    ]


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
