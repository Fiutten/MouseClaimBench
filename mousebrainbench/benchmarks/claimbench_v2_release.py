"""Release check for post-submission ClaimBench v2 artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from mousebrainbench import __version__
from mousebrainbench.artifacts import code_revision


DEFAULT_OUTPUT = Path("results/claimbench_v2_release/summary.json")
DEFAULT_MARKDOWN = Path("results/claimbench_v2_release/summary.md")

REQUIRED_ARTIFACTS = (
    Path("docs/SUBMISSION_BASELINE_AND_V2_SEPARATION.md"),
    Path("docs/CLAIMBENCH_SOTA_AND_VIABILITY.md"),
    Path("results/claim_adversarial_v2/summary.json"),
    Path("results/claim_threshold_sensitivity_v2/summary.json"),
    Path("results/external_causal_claim_validation/summary.json"),
    Path("results/reviewer_attack_suite_v2/summary.json"),
    Path("results/manuscript_claim_audit/summary.json"),
    Path("results/llm_claim_extraction_audit/summary.json"),
    Path("results/uncertainty_claim_gate_v2/summary.json"),
    Path("results/cost_fidelity_claim_frontier/summary.json"),
    Path("results/external_benchmark_registry/summary.json"),
    Path("results/scifact_claim_verification/summary.json"),
    Path("results/tuebingen_causal_direction/summary.json"),
    Path("results/claimbench_component_ablation/summary.json"),
    Path("results/claimbench_threat_model/summary.json"),
    Path("results/claimbench_unified_report/summary.json"),
)


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text()) if path.exists() else {}


def run(output: Path = DEFAULT_OUTPUT, markdown: Path = DEFAULT_MARKDOWN, root: Path = Path(".")) -> Path:
    """Check v2 artifact readiness without modifying submitted baseline artifacts."""

    rows: list[dict[str, Any]] = []
    missing: list[str] = []
    dirty: list[str] = []
    failing: list[str] = []
    expected_decisions = {
        "results/claim_adversarial_v2/summary.json": (
            "claimbench_v2_blocks_overclaiming_under_broad_attacks"
        ),
        "results/claim_threshold_sensitivity_v2/summary.json": (
            "claim_thresholds_have_nontrivial_safe_region_with_reportable_limits"
        ),
        "results/external_causal_claim_validation/summary.json": (
            "external_causal_validation_passed"
        ),
        "results/reviewer_attack_suite_v2/summary.json": (
            "reviewer_attack_suite_v2_passed_with_reportable_limits"
        ),
        "results/manuscript_claim_audit/summary.json": "manuscript_claim_audit_passed",
        "results/llm_claim_extraction_audit/summary.json": (
            "llm_claim_extraction_layer_ready_non_authoritative"
        ),
        "results/uncertainty_claim_gate_v2/summary.json": (
            "uncertainty_gate_blocks_unsupported_support"
        ),
        "results/cost_fidelity_claim_frontier/summary.json": "cost_fidelity_claim_frontier_built",
        "results/external_benchmark_registry/summary.json": (
            "external_benchmarks_registered_with_pending_data"
        ),
        "results/scifact_claim_verification/summary.json": "scifact_external_claim_audit_ready",
        "results/tuebingen_causal_direction/summary.json": (
            "tuebingen_external_direction_benchmark_ready"
        ),
        "results/claimbench_component_ablation/summary.json": (
            "claimbench_components_have_nonredundant_value"
        ),
        "results/claimbench_threat_model/summary.json": (
            "claimbench_threat_model_passed_with_boundaries"
        ),
        "results/claimbench_unified_report/summary.json": (
            "claimbench_v2_methodological_package_ready"
        ),
    }
    for relative in REQUIRED_ARTIFACTS:
        path = root / relative
        exists = path.exists()
        revision = None
        decision = None
        expected = expected_decisions.get(str(relative))
        if not exists:
            missing.append(str(relative))
        elif path.suffix == ".json":
            data = _load_json(path)
            revision = data.get("git_revision")
            decision = data.get("decision")
            if isinstance(revision, str) and revision.endswith("-dirty"):
                dirty.append(str(relative))
            if expected is not None and decision != expected:
                failing.append(str(relative))
        rows.append(
            {
                "artifact": str(relative),
                "exists": exists,
                "git_revision": revision,
                "decision": decision,
                "expected_decision": expected,
            }
        )

    payload = {
        "version": __version__,
        "git_revision": code_revision(),
        "analysis": "claimbench_v2_release",
        "baseline_commit": "ff0aca9",
        "missing_artifacts": missing,
        "dirty_artifacts": dirty,
        "failing_artifacts": failing,
        "required_artifacts": rows,
        "decision": (
            "claimbench_v2_release_ready"
            if not missing and not dirty and not failing
            else "claimbench_v2_release_requires_cleanup"
        ),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2))
    write_markdown(payload, markdown)
    return output


def write_markdown(payload: dict[str, Any], markdown: Path) -> None:
    """Write v2 release check report."""

    lines = [
        "# ClaimBench v2 Release Check",
        "",
        f"- Decision: `{payload['decision']}`",
        f"- Baseline commit: `{payload['baseline_commit']}`",
        f"- Missing artifacts: `{len(payload['missing_artifacts'])}`",
        f"- Dirty artifacts: `{len(payload['dirty_artifacts'])}`",
        f"- Failing artifacts: `{len(payload['failing_artifacts'])}`",
        "",
        "| Artifact | Exists | Decision | Expected | Git revision |",
        "|---|---:|---|---|---|",
    ]
    for row in payload["required_artifacts"]:
        lines.append(
            f"| `{row['artifact']}` | `{row['exists']}` | `{row['decision']}` | "
            f"`{row['expected_decision']}` | `{row['git_revision']}` |"
        )
    lines.append("")
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
