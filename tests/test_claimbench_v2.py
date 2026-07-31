import json
import subprocess
from pathlib import Path

from mousebrainbench.benchmarks.claim_adversarial_v2 import build_cases, run as run_adversarial_v2
from mousebrainbench.benchmarks.claim_threshold_sensitivity_v2 import run as run_sensitivity_v2
from mousebrainbench.benchmarks.claimbench_component_ablation import run as run_ablation
from mousebrainbench.benchmarks.claimbench_threat_model import run as run_threat
from mousebrainbench.benchmarks.claimbench_v2_release import run as run_release_v2
from mousebrainbench.benchmarks.claimbench_unified_report import run as run_unified
from mousebrainbench.benchmarks.cost_fidelity_claim_frontier import run as run_frontier
from mousebrainbench.benchmarks.external_causal_claim_validation import run as run_external_causal
from mousebrainbench.benchmarks.external_benchmark_registry import run as run_registry
from mousebrainbench.benchmarks.manuscript_claim_auditor import run as run_manuscript_audit
from mousebrainbench.benchmarks.reviewer_attack_suite_v2 import run as run_reviewer_attack_v2
from mousebrainbench.benchmarks.uncertainty_claim_gate_v2 import run as run_uncertainty


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload))


def test_claim_adversarial_v2_has_broad_stress_suite(tmp_path) -> None:
    output = run_adversarial_v2(
        output=tmp_path / "results/claim_adversarial_v2/summary.json",
        markdown=tmp_path / "results/claim_adversarial_v2/summary.md",
    )
    payload = json.loads(output.read_text())
    aggregate = {row["evaluator"]: row for row in payload["aggregate_by_evaluator"]}

    assert len(build_cases()) >= 100
    assert payload["decision"] == "claimbench_v2_blocks_overclaiming_under_broad_attacks"
    assert aggregate["claim_gate"]["fp"] == 0
    assert aggregate["correlation_only"]["fp"] > aggregate["claim_gate"]["fp"]


def test_threshold_sensitivity_v2_reports_safe_region(tmp_path) -> None:
    output = run_sensitivity_v2(
        output=tmp_path / "results/claim_threshold_sensitivity_v2/summary.json",
        markdown=tmp_path / "results/claim_threshold_sensitivity_v2/summary.md",
    )
    payload = json.loads(output.read_text())

    assert payload["num_threshold_cells"] == 243
    assert payload["decision"] == "claim_thresholds_have_nontrivial_safe_region_with_reportable_limits"
    assert payload["safe_cells"] >= 20
    assert payload["dangerous_cells"] > 0


def test_external_causal_validation_passes_known_truth(tmp_path) -> None:
    output = run_external_causal(
        output=tmp_path / "results/external_causal_claim_validation/summary.json",
        markdown=tmp_path / "results/external_causal_claim_validation/summary.md",
    )
    payload = json.loads(output.read_text())
    aggregate = {row["evaluator"]: row for row in payload["aggregate_by_evaluator"]}

    assert payload["decision"] == "external_causal_validation_passed"
    assert aggregate["claim_gate"]["fp"] == 0
    assert aggregate["claim_gate"]["fn"] == 0


def test_reviewer_attack_and_release_v2_are_reproducible(tmp_path, monkeypatch) -> None:
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    monkeypatch.setenv("MOUSEBRAINBENCH_GIT_REVISION", head)

    run_adversarial_v2(
        output=tmp_path / "results/claim_adversarial_v2/summary.json",
        markdown=tmp_path / "results/claim_adversarial_v2/summary.md",
    )
    run_sensitivity_v2(
        output=tmp_path / "results/claim_threshold_sensitivity_v2/summary.json",
        markdown=tmp_path / "results/claim_threshold_sensitivity_v2/summary.md",
    )
    run_external_causal(
        output=tmp_path / "results/external_causal_claim_validation/summary.json",
        markdown=tmp_path / "results/external_causal_claim_validation/summary.md",
    )
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs/SUBMISSION_BASELINE_AND_V2_SEPARATION.md").write_text("baseline")
    (tmp_path / "docs/CLAIMBENCH_SOTA_AND_VIABILITY.md").write_text("sota")
    (tmp_path / "configs/claims").mkdir(parents=True)
    (tmp_path / "configs/claims/mousebrainbench_claims.yaml").write_text(
        open("configs/claims/mousebrainbench_claims.yaml").read()
    )
    (tmp_path / "README.md").write_text("MouseBrainBench test manuscript.")

    # The reviewer suite also depends on baseline artifacts, so this test checks
    # that missing baseline inputs are reported as non-blocking medium risks.
    attack_output = run_reviewer_attack_v2(
        output=tmp_path / "results/reviewer_attack_suite_v2/summary.json",
        markdown=tmp_path / "results/reviewer_attack_suite_v2/summary.md",
        root=tmp_path,
    )
    attack_payload = json.loads(attack_output.read_text())

    assert attack_payload["decision"] == "reviewer_attack_suite_v2_passed_with_reportable_limits"

    run_manuscript_audit(
        claims=Path("configs/claims/mousebrainbench_claims.yaml"),
        manuscript=(Path("README.md"),),
        output=tmp_path / "results/manuscript_claim_audit/summary.json",
        markdown=tmp_path / "results/manuscript_claim_audit/summary.md",
        root=tmp_path,
    )
    _write_json(
        tmp_path / "results/llm_claim_extraction_audit/summary.json",
        {
            "decision": "llm_claim_extraction_layer_ready_non_authoritative",
            "num_candidates": 3,
            "llm_authoritative": False,
            "llm_api_called": False,
            "git_revision": head,
        },
    )
    run_uncertainty(
        output=tmp_path / "results/uncertainty_claim_gate_v2/summary.json",
        markdown=tmp_path / "results/uncertainty_claim_gate_v2/summary.md",
    )
    run_frontier(
        output=tmp_path / "results/cost_fidelity_claim_frontier/summary.json",
        markdown=tmp_path / "results/cost_fidelity_claim_frontier/summary.md",
        root=tmp_path,
    )
    run_registry(
        output=tmp_path / "results/external_benchmark_registry/summary.json",
        markdown=tmp_path / "results/external_benchmark_registry/summary.md",
        root=tmp_path,
    )
    _write_json(
        tmp_path / "results/scifact_claim_verification/summary.json",
        {
            "decision": "scifact_external_claim_audit_ready",
            "num_claims": 300,
            "retrieval_recall_at_5": 0.89,
            "shortcut_overclaiming_risk": 0.19,
            "retrieval_overclaiming_risk": 0.25,
            "git_revision": head,
        },
    )
    _write_json(
        tmp_path / "results/tuebingen_causal_direction/summary.json",
        {
            "decision": "tuebingen_external_direction_benchmark_ready",
            "num_pairs_loaded": 108,
            "direction_accuracy": 0.48,
            "correlation_only_direction_overclaims": 79,
            "causal_control_claim_allowed": True,
            "causal_performance_claim_allowed": False,
            "git_revision": head,
        },
    )
    run_ablation(
        output=tmp_path / "results/claimbench_component_ablation/summary.json",
        markdown=tmp_path / "results/claimbench_component_ablation/summary.md",
        root=tmp_path,
    )
    _write_json(
        tmp_path / "results/claimbench_v2_release/summary.json",
        {
            "decision": "claimbench_v2_release_ready",
            "missing_artifacts": [],
            "dirty_artifacts": [],
            "failing_artifacts": [],
            "git_revision": head,
        },
    )
    _write_json(
        tmp_path / "results/claimbench_unified_report/summary.json",
        {"decision": "claimbench_v2_methodological_package_ready", "git_revision": head},
    )
    run_threat(
        output=tmp_path / "results/claimbench_threat_model/summary.json",
        markdown=tmp_path / "results/claimbench_threat_model/summary.md",
        root=tmp_path,
    )
    run_unified(
        output=tmp_path / "results/claimbench_unified_report/summary.json",
        markdown=tmp_path / "results/claimbench_unified_report/summary.md",
        root=tmp_path,
    )

    release_output = run_release_v2(
        output=tmp_path / "results/claimbench_v2_release/summary.json",
        markdown=tmp_path / "results/claimbench_v2_release/summary.md",
        root=tmp_path,
    )
    release_payload = json.loads(release_output.read_text())

    assert release_payload["decision"] == "claimbench_v2_release_ready"
    assert release_payload["dirty_artifacts"] == []
