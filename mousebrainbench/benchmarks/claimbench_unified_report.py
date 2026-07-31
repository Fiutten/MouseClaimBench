"""Unified experimental report for ClaimBench v2.

The report consolidates internal synthetic controls, public external controls,
manuscript audit, component ablation, and release reproducibility into one
decision artifact.  Its purpose is to make the next-paper status explicit
instead of relying on informal interpretation of many JSON files.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from mousebrainbench import __version__
from mousebrainbench.artifacts import code_revision


DEFAULT_OUTPUT = Path("results/claimbench_unified_report/summary.json")
DEFAULT_MARKDOWN = Path("results/claimbench_unified_report/summary.md")


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text()) if path.exists() else {}


def _criterion(
    *,
    name: str,
    passed: bool,
    artifact: str,
    evidence: str,
    interpretation: str,
) -> dict[str, Any]:
    return {
        "name": name,
        "passed": passed,
        "artifact": artifact,
        "evidence": evidence,
        "interpretation": interpretation,
    }


def build_criteria(root: Path = Path(".")) -> list[dict[str, Any]]:
    """Build pass/fail criteria from existing artifacts."""

    adversarial = _load(root / "results/claim_adversarial_v2/summary.json")
    threshold = _load(root / "results/claim_threshold_sensitivity_v2/summary.json")
    uncertainty = _load(root / "results/uncertainty_claim_gate_v2/summary.json")
    scifact = _load(root / "results/scifact_claim_verification/summary.json")
    tuebingen = _load(root / "results/tuebingen_causal_direction/summary.json")
    manuscript = _load(root / "results/manuscript_claim_audit/summary.json")
    llm_claims = _load(root / "results/llm_claim_extraction_audit/summary.json")
    ablation = _load(root / "results/claimbench_component_ablation/summary.json")
    reviewer = _load(root / "results/reviewer_attack_suite_v2/summary.json")
    threat = _load(root / "results/claimbench_threat_model/summary.json")

    gate_row = next(
        (
            row
            for row in adversarial.get("aggregate_by_evaluator", [])
            if row.get("evaluator") == "claim_gate"
        ),
        {},
    )
    return [
        _criterion(
            name="synthetic_known_truth_gate",
            passed=(
                adversarial.get("decision")
                == "claimbench_v2_blocks_overclaiming_under_broad_attacks"
                and int(gate_row.get("fp", 1)) == 0
            ),
            artifact="results/claim_adversarial_v2/summary.json",
            evidence=f"cases={adversarial.get('num_cases')}; claim_gate_fp={gate_row.get('fp')}",
            interpretation="The internal gate blocks unsupported known-truth overclaims.",
        ),
        _criterion(
            name="threshold_limits_reported",
            passed=(
                threshold.get("decision")
                == "claim_thresholds_have_nontrivial_safe_region_with_reportable_limits"
                and int(threshold.get("dangerous_cells", 0)) > 0
            ),
            artifact="results/claim_threshold_sensitivity_v2/summary.json",
            evidence=(
                f"safe={threshold.get('safe_cells')}; "
                f"dangerous={threshold.get('dangerous_cells')}"
            ),
            interpretation="Thresholds have safe regions, but dangerous regions must be reported.",
        ),
        _criterion(
            name="uncertainty_blocks_unsupported_support",
            passed=(
                uncertainty.get("decision") == "uncertainty_gate_blocks_unsupported_support"
                and int(uncertainty.get("unsupported_supported", 1)) == 0
            ),
            artifact="results/uncertainty_claim_gate_v2/summary.json",
            evidence=(
                f"unsupported_supported={uncertainty.get('unsupported_supported')}; "
                f"supported_uncertain={uncertainty.get('supported_uncertain')}"
            ),
            interpretation="Uncertainty is conservative and does not turn unsupported claims into support.",
        ),
        _criterion(
            name="scifact_external_claim_control",
            passed=(
                scifact.get("decision") == "scifact_external_claim_audit_ready"
                and float(scifact.get("retrieval_recall_at_5", 0.0)) >= 0.50
                and float(scifact.get("shortcut_overclaiming_risk", 0.0)) > 0.0
            ),
            artifact="results/scifact_claim_verification/summary.json",
            evidence=(
                f"claims={scifact.get('num_claims')}; "
                f"bm25_recall_at_5={scifact.get('retrieval_recall_at_5')}; "
                f"shortcut_ORI={scifact.get('shortcut_overclaiming_risk')}"
            ),
            interpretation="SciFact supports an external claim-auditing case, not SOTA verification.",
        ),
        _criterion(
            name="tuebingen_causal_overclaim_control",
            passed=(
                tuebingen.get("decision") == "tuebingen_external_direction_benchmark_ready"
                and bool(tuebingen.get("causal_control_claim_allowed", False))
                and not bool(tuebingen.get("causal_performance_claim_allowed", True))
            ),
            artifact="results/tuebingen_causal_direction/summary.json",
            evidence=(
                f"pairs={tuebingen.get('num_pairs_loaded')}; "
                f"direction_accuracy={tuebingen.get('direction_accuracy')}; "
                f"corr_overclaims={tuebingen.get('correlation_only_direction_overclaims')}"
            ),
            interpretation="Tuebingen supports causal overclaim auditing, not causal-discovery performance.",
        ),
        _criterion(
            name="manuscript_claim_audit",
            passed=(
                manuscript.get("decision") == "manuscript_claim_audit_passed"
                and len(manuscript.get("active_risk_pattern_hits", [])) == 0
            ),
            artifact="results/manuscript_claim_audit/summary.json",
            evidence=(
                f"inputs={len(manuscript.get('existing_manuscript_inputs', []))}; "
                f"active_hits={len(manuscript.get('active_risk_pattern_hits', []))}"
            ),
            interpretation="The current manuscript wording passes executable claim-boundary checks.",
        ),
        _criterion(
            name="llm_claim_extraction_boundary",
            passed=(
                llm_claims.get("decision")
                == "llm_claim_extraction_layer_ready_non_authoritative"
                and not bool(llm_claims.get("llm_authoritative", True))
                and not bool(llm_claims.get("llm_api_called", True))
            ),
            artifact="results/llm_claim_extraction_audit/summary.json",
            evidence=(
                f"candidates={llm_claims.get('num_candidates')}; "
                f"authoritative={llm_claims.get('llm_authoritative')}; "
                f"api_called={llm_claims.get('llm_api_called')}"
            ),
            interpretation="LLM assistance is limited to candidate extraction, not claim authorization.",
        ),
        _criterion(
            name="component_ablation_nonredundancy",
            passed=ablation.get("decision") == "claimbench_components_have_nonredundant_value",
            artifact="results/claimbench_component_ablation/summary.json",
            evidence=(
                f"components={ablation.get('num_components')}; "
                f"high_or_critical={ablation.get('high_or_critical_components')}"
            ),
            interpretation="Core components are non-redundant because ablations reintroduce risks.",
        ),
        _criterion(
            name="reviewer_attack_suite",
            passed=reviewer.get("decision") == "reviewer_attack_suite_v2_passed_with_reportable_limits",
            artifact="results/reviewer_attack_suite_v2/summary.json",
            evidence=f"risks={len(reviewer.get('risks', []))}",
            interpretation="Reviewer attacks pass with explicit reportable limits.",
        ),
        _criterion(
            name="reviewer_threat_model",
            passed=threat.get("decision") == "claimbench_threat_model_passed_with_boundaries",
            artifact="results/claimbench_threat_model/summary.json",
            evidence=(
                f"passed={threat.get('passed_threats')}; "
                f"critical_failed={len(threat.get('critical_failed_threats', []))}"
            ),
            interpretation="Known reviewer threats are mapped to artifacts and claim boundaries.",
        ),
    ]


def run(output: Path = DEFAULT_OUTPUT, markdown: Path = DEFAULT_MARKDOWN, root: Path = Path(".")) -> Path:
    """Write the unified ClaimBench v2 report."""

    criteria = build_criteria(root)
    failed = [row for row in criteria if not row["passed"]]
    payload = {
        "version": __version__,
        "git_revision": code_revision(),
        "analysis": "claimbench_unified_report",
        "num_criteria": len(criteria),
        "passed_criteria": len(criteria) - len(failed),
        "failed_criteria": failed,
        "criteria": criteria,
        "publishable_claim": (
            "ClaimBench v2 is a claim-aware auditing framework that separates "
            "prediction, evidence retrieval, causal direction, uncertainty, manuscript "
            "wording, and release reproducibility. It is not a SciFact SOTA system and "
            "not a causal-discovery performance method."
        ),
        "decision": (
            "claimbench_v2_methodological_package_ready"
            if not failed
            else "claimbench_v2_methodological_package_requires_revision"
        ),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2))
    write_markdown(payload, markdown)
    return output


def write_markdown(payload: dict[str, Any], markdown: Path) -> None:
    """Write the unified report in manuscript-friendly form."""

    lines = [
        "# ClaimBench v2 Unified Report",
        "",
        f"- Decision: `{payload['decision']}`",
        f"- Criteria: `{payload['passed_criteria']}/{payload['num_criteria']}` passed",
        "",
        "## Criteria",
        "",
        "| Criterion | Passed | Evidence | Interpretation |",
        "|---|---:|---|---|",
    ]
    for row in payload["criteria"]:
        lines.append(
            f"| `{row['name']}` | `{row['passed']}` | {row['evidence']} | "
            f"{row['interpretation']} |"
        )
    lines.extend(
        [
            "",
            "## Publishable Claim Boundary",
            "",
            str(payload["publishable_claim"]),
            "",
        ]
    )
    markdown.parent.mkdir(parents=True, exist_ok=True)
    markdown.write_text("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--markdown", type=Path, default=DEFAULT_MARKDOWN)
    parser.add_argument("--root", type=Path, default=Path("."))
    args = parser.parse_args()
    print(json.dumps({"output": str(run(args.output, args.markdown, args.root).resolve())}))


if __name__ == "__main__":
    main()
