import json
import subprocess
from pathlib import Path

from mousebrainbench.benchmarks.cost_fidelity_claim_frontier import run as run_frontier
from mousebrainbench.benchmarks.external_benchmark_registry import run as run_registry
from mousebrainbench.benchmarks.manuscript_claim_auditor import run as run_manuscript_audit
from mousebrainbench.benchmarks.uncertainty_claim_gate_v2 import run as run_uncertainty
from mousebrainbench.claimdsl import audit_claim_specs, load_claim_specs


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload))


def _write_minimal_v2_artifacts(root: Path) -> None:
    _write_json(
        root / "results/claim_adversarial_v2/summary.json",
        {
            "decision": "claimbench_v2_blocks_overclaiming_under_broad_attacks",
            "num_cases": 144,
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
        root / "results/external_causal_claim_validation/summary.json",
        {"decision": "external_causal_validation_passed", "git_revision": "abc"},
    )
    _write_json(
        root / "results/reviewer_attack_suite_v2/summary.json",
        {
            "decision": "reviewer_attack_suite_v2_passed_with_reportable_limits",
            "git_revision": "abc",
        },
    )
    _write_json(
        root / "results/scifact_claim_verification/summary.json",
        {
            "decision": "scifact_external_claim_audit_ready",
            "num_claims": 300,
            "shortcut_false_positives": 37,
            "shortcut_overclaiming_risk": 0.21,
            "git_revision": "abc",
        },
    )
    _write_json(
        root / "results/tuebingen_causal_direction/summary.json",
        {
            "decision": "tuebingen_external_direction_benchmark_ready",
            "num_pairs_loaded": 108,
            "correlation_only_direction_overclaims": 79,
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


def test_claim_dsl_audits_required_artifacts(tmp_path) -> None:
    _write_minimal_v2_artifacts(tmp_path)
    claims_file = Path("configs/claims/mousebrainbench_claims.yaml")

    specs = load_claim_specs(claims_file)
    results = audit_claim_specs(specs, root=tmp_path)

    assert len(results) == 7
    assert all(result.status == "supported" for result in results)


def test_manuscript_claim_auditor_blocks_declared_bad_wording(tmp_path) -> None:
    _write_minimal_v2_artifacts(tmp_path)
    (tmp_path / "configs/claims").mkdir(parents=True)
    claim_text = Path("configs/claims/mousebrainbench_claims.yaml").read_text()
    (tmp_path / "configs/claims/mousebrainbench_claims.yaml").write_text(claim_text)
    manuscript = tmp_path / "paper.tex"
    manuscript.write_text("ClaimBench v2 proves biological truth.")

    output = run_manuscript_audit(
        claims=Path("configs/claims/mousebrainbench_claims.yaml"),
        manuscript=(Path("paper.tex"),),
        output=tmp_path / "results/manuscript_claim_audit/summary.json",
        markdown=tmp_path / "results/manuscript_claim_audit/summary.md",
        root=tmp_path,
    )
    payload = json.loads(output.read_text())

    assert payload["decision"] == "manuscript_claim_audit_blocks_release"
    assert payload["blocked_wording_hits"]


def test_manuscript_claim_auditor_detects_risky_patterns_but_allows_negated_limits(
    tmp_path,
) -> None:
    _write_minimal_v2_artifacts(tmp_path)
    (tmp_path / "configs/claims").mkdir(parents=True)
    claim_text = Path("configs/claims/mousebrainbench_claims.yaml").read_text()
    (tmp_path / "configs/claims/mousebrainbench_claims.yaml").write_text(claim_text)

    risky = tmp_path / "risky.tex"
    risky.write_text("We present a full mouse-brain digital twin with SOTA causal mechanism.")
    risky_output = run_manuscript_audit(
        claims=Path("configs/claims/mousebrainbench_claims.yaml"),
        manuscript=(Path("risky.tex"),),
        output=tmp_path / "results/manuscript_claim_audit/risky.json",
        markdown=tmp_path / "results/manuscript_claim_audit/risky.md",
        root=tmp_path,
    )
    risky_payload = json.loads(risky_output.read_text())
    assert risky_payload["decision"] == "manuscript_claim_audit_blocks_release"
    assert risky_payload["active_risk_pattern_hits"]

    limited = tmp_path / "limited.tex"
    limited.write_text("This is not a full mouse-brain digital twin and not a SOTA claim.")
    limited_output = run_manuscript_audit(
        claims=Path("configs/claims/mousebrainbench_claims.yaml"),
        manuscript=(Path("limited.tex"),),
        output=tmp_path / "results/manuscript_claim_audit/limited.json",
        markdown=tmp_path / "results/manuscript_claim_audit/limited.md",
        root=tmp_path,
    )
    limited_payload = json.loads(limited_output.read_text())
    assert limited_payload["decision"] == "manuscript_claim_audit_passed"
    assert limited_payload["active_risk_pattern_hits"] == []


def test_manuscript_claim_auditor_uses_default_paper_sources(tmp_path) -> None:
    _write_minimal_v2_artifacts(tmp_path)
    (tmp_path / "configs/claims").mkdir(parents=True)
    claim_text = Path("configs/claims/mousebrainbench_claims.yaml").read_text()
    (tmp_path / "configs/claims/mousebrainbench_claims.yaml").write_text(claim_text)
    (tmp_path / "paper/sections").mkdir(parents=True)
    (tmp_path / "paper/main.tex").write_text("Main paper source.")
    (tmp_path / "paper/sections/results.tex").write_text("This is not a SOTA claim.")

    output = run_manuscript_audit(
        claims=Path("configs/claims/mousebrainbench_claims.yaml"),
        output=tmp_path / "results/manuscript_claim_audit/default.json",
        markdown=tmp_path / "results/manuscript_claim_audit/default.md",
        root=tmp_path,
    )
    payload = json.loads(output.read_text())

    assert payload["decision"] == "manuscript_claim_audit_passed"
    assert payload["manuscript_inputs"] == ["paper/main.tex"]


def test_uncertainty_claim_gate_blocks_unsupported_support(tmp_path) -> None:
    output = run_uncertainty(
        output=tmp_path / "results/uncertainty_claim_gate_v2/summary.json",
        markdown=tmp_path / "results/uncertainty_claim_gate_v2/summary.md",
    )
    payload = json.loads(output.read_text())

    assert payload["decision"] == "uncertainty_gate_blocks_unsupported_support"
    assert payload["unsupported_supported"] == 0
    assert payload["status_counts"]["uncertain"] > 0


def test_frontier_and_external_registry_are_reproducible(tmp_path, monkeypatch) -> None:
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    monkeypatch.setenv("MOUSEBRAINBENCH_GIT_REVISION", head)
    _write_minimal_v2_artifacts(tmp_path)
    _write_json(
        tmp_path / "results/sensorium_static_model_comparator/summary.json",
        {"comparison": "sensorium2022_static_cross_dataset_comparator"},
    )
    _write_json(
        tmp_path / "results/dynamic_sensorium_model_comparator/summary.json",
        {"comparison": "dynamic_sensorium_predictive_model_comparator"},
    )
    _write_json(
        tmp_path / "results/microns_primary_robustness/summary.json",
        {"all_cohorts_robust": True},
    )

    frontier = json.loads(
        run_frontier(
            output=tmp_path / "results/cost_fidelity_claim_frontier/summary.json",
            markdown=tmp_path / "results/cost_fidelity_claim_frontier/summary.md",
            root=tmp_path,
        ).read_text()
    )
    registry = json.loads(
        run_registry(
            output=tmp_path / "results/external_benchmark_registry/summary.json",
            markdown=tmp_path / "results/external_benchmark_registry/summary.md",
            root=tmp_path,
        ).read_text()
    )

    assert frontier["decision"] == "cost_fidelity_claim_frontier_built"
    assert frontier["frontier"]
    assert registry["decision"] == "external_benchmarks_registered_with_pending_data"
    assert registry["registered_benchmarks"] == 3
