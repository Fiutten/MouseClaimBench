import json
from pathlib import Path

from mousebrainbench.benchmarks.claimbench_component_ablation import run as run_ablation
from mousebrainbench.benchmarks.claimbench_reproduce_package import STAGES
from mousebrainbench.benchmarks.claimbench_threat_model import run as run_threat
from mousebrainbench.benchmarks.claimbench_unified_report import run as run_unified
from mousebrainbench.benchmarks.llm_claim_extraction_audit import run as run_llm_claims


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload))


def _write_minimal_package(root: Path) -> None:
    _write_json(
        root / "results/claim_adversarial_v2/summary.json",
        {
            "decision": "claimbench_v2_blocks_overclaiming_under_broad_attacks",
            "num_cases": 144,
            "aggregate_by_evaluator": [
                {"evaluator": "claim_gate", "fp": 0, "overclaiming_risk_index": 0.0},
                {"evaluator": "correlation_only", "fp": 20, "overclaiming_risk_index": 0.5},
                {"evaluator": "compensatory_score", "fp": 8, "overclaiming_risk_index": 0.1},
                {
                    "evaluator": "ablated_claim_gate_no_topology",
                    "fp": 2,
                    "overclaiming_risk_index": 0.02,
                },
                {
                    "evaluator": "ablated_claim_gate_no_directed",
                    "fp": 2,
                    "overclaiming_risk_index": 0.02,
                },
            ],
            "git_revision": "abc",
        },
    )
    _write_json(
        root / "results/claim_threshold_sensitivity_v2/summary.json",
        {
            "decision": "claim_thresholds_have_nontrivial_safe_region_with_reportable_limits",
            "safe_cells": 108,
            "dangerous_cells": 135,
            "git_revision": "abc",
        },
    )
    _write_json(
        root / "results/uncertainty_claim_gate_v2/summary.json",
        {
            "decision": "uncertainty_gate_blocks_unsupported_support",
            "unsupported_supported": 0,
            "supported_uncertain": 4,
            "git_revision": "abc",
        },
    )
    _write_json(
        root / "results/scifact_claim_verification/summary.json",
        {
            "decision": "scifact_external_claim_audit_ready",
            "num_claims": 300,
            "retrieval_recall_at_5": 0.89,
            "shortcut_overclaiming_risk": 0.19,
            "retrieval_overclaiming_risk": 0.25,
            "git_revision": "abc",
        },
    )
    _write_json(
        root / "results/tuebingen_causal_direction/summary.json",
        {
            "decision": "tuebingen_external_direction_benchmark_ready",
            "num_pairs_loaded": 108,
            "direction_accuracy": 0.48,
            "correlation_only_direction_overclaims": 79,
            "causal_control_claim_allowed": True,
            "causal_performance_claim_allowed": False,
            "git_revision": "abc",
        },
    )
    _write_json(
        root / "results/manuscript_claim_audit/summary.json",
        {
            "decision": "manuscript_claim_audit_passed",
            "existing_manuscript_inputs": ["paper/main.tex"],
            "active_risk_pattern_hits": [],
            "git_revision": "abc",
        },
    )
    _write_json(
        root / "results/llm_claim_extraction_audit/summary.json",
        {
            "decision": "llm_claim_extraction_layer_ready_non_authoritative",
            "num_candidates": 4,
            "llm_authoritative": False,
            "llm_api_called": False,
            "git_revision": "abc",
        },
    )
    _write_json(
        root / "results/reviewer_attack_suite_v2/summary.json",
        {
            "decision": "reviewer_attack_suite_v2_passed_with_reportable_limits",
            "risks": [{"level": "medium"}],
            "git_revision": "abc",
        },
    )
    _write_json(
        root / "results/claimbench_v2_release/summary.json",
        {
            "decision": "claimbench_v2_release_ready",
            "missing_artifacts": [],
            "dirty_artifacts": [],
            "failing_artifacts": [],
            "git_revision": "abc",
        },
    )


def test_component_ablation_reports_nonredundant_value(tmp_path) -> None:
    _write_minimal_package(tmp_path)
    output = run_ablation(
        output=tmp_path / "results/claimbench_component_ablation/summary.json",
        markdown=tmp_path / "results/claimbench_component_ablation/summary.md",
        root=tmp_path,
    )
    payload = json.loads(output.read_text())

    assert payload["decision"] == "claimbench_components_have_nonredundant_value"
    assert payload["high_or_critical_components"] >= 5
    assert any(row["component"] == "causal_abstention" for row in payload["rows"])


def test_unified_report_passes_when_all_criteria_are_met(tmp_path) -> None:
    _write_minimal_package(tmp_path)
    run_ablation(
        output=tmp_path / "results/claimbench_component_ablation/summary.json",
        markdown=tmp_path / "results/claimbench_component_ablation/summary.md",
        root=tmp_path,
    )
    _write_json(
        tmp_path / "results/claimbench_unified_report/summary.json",
        {
            "decision": "claimbench_v2_methodological_package_ready",
            "git_revision": "abc",
        },
    )
    run_threat(
        output=tmp_path / "results/claimbench_threat_model/summary.json",
        markdown=tmp_path / "results/claimbench_threat_model/summary.md",
        root=tmp_path,
    )
    output = run_unified(
        output=tmp_path / "results/claimbench_unified_report/summary.json",
        markdown=tmp_path / "results/claimbench_unified_report/summary.md",
        root=tmp_path,
    )
    payload = json.loads(output.read_text())

    assert payload["decision"] == "claimbench_v2_methodological_package_ready"
    assert payload["passed_criteria"] == payload["num_criteria"]
    assert "not a causal-discovery performance method" in payload["publishable_claim"]


def test_threat_model_maps_reviewer_attacks_to_artifacts(tmp_path) -> None:
    _write_minimal_package(tmp_path)
    run_ablation(
        output=tmp_path / "results/claimbench_component_ablation/summary.json",
        markdown=tmp_path / "results/claimbench_component_ablation/summary.md",
        root=tmp_path,
    )
    _write_json(
        tmp_path / "results/claimbench_unified_report/summary.json",
        {
            "decision": "claimbench_v2_methodological_package_ready",
            "git_revision": "abc",
        },
    )
    output = run_threat(
        output=tmp_path / "results/claimbench_threat_model/summary.json",
        markdown=tmp_path / "results/claimbench_threat_model/summary.md",
        root=tmp_path,
    )
    payload = json.loads(output.read_text())

    assert payload["decision"] == "claimbench_threat_model_passed_with_boundaries"
    assert payload["passed_threats"] == payload["num_threats"]
    assert any(row["threat_id"] == "causal_overclaim" for row in payload["rows"])
    assert any(row["threat_id"] == "llm_authority_drift" for row in payload["rows"])


def test_reproduction_runner_declares_ordered_package_stages() -> None:
    stage_names = [stage.name for stage in STAGES]
    data_roots = {stage.name: stage.data_root for stage in STAGES}

    assert stage_names.index("component_ablation") < stage_names.index("threat_model")
    assert stage_names.index("llm_claim_extraction_audit") < stage_names.index("threat_model")
    assert stage_names.index("threat_model") < stage_names.index("unified_report")
    assert stage_names.index("unified_report") < stage_names.index("release_check")
    assert stage_names[-1] == "release_check"
    assert data_roots["scifact_external_claims"] is not None
    assert data_roots["tuebingen_causal_direction"] is not None


def test_llm_claim_extraction_layer_is_non_authoritative(tmp_path) -> None:
    (tmp_path / "paper").mkdir()
    (tmp_path / "paper/main.tex").write_text(
        "MouseBrainBench predicts neural responses but does not validate a causal mechanism. "
        "The framework blocks state-of-the-art predictor claims without matched baselines."
    )
    output = run_llm_claims(
        output=tmp_path / "results/llm_claim_extraction_audit/summary.json",
        markdown=tmp_path / "results/llm_claim_extraction_audit/summary.md",
        root=tmp_path,
    )
    payload = json.loads(output.read_text())

    assert payload["decision"] == "llm_claim_extraction_layer_ready_non_authoritative"
    assert payload["llm_api_called"] is False
    assert payload["llm_authoritative"] is False
    assert payload["num_candidates"] > 0
