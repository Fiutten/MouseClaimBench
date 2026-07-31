"""Reproduce the ClaimBench v2 methodological package.

The runner executes the package in a fixed order and writes a manifest with the
decision emitted by every stage.  It is intentionally local and deterministic:
it does not download datasets or install dependencies.  External artifacts are
regenerated only when their local data are already present.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from mousebrainbench import __version__
from mousebrainbench.artifacts import code_revision
from mousebrainbench.benchmarks.claim_adversarial_v2 import run as run_adversarial
from mousebrainbench.benchmarks.claim_threshold_sensitivity_v2 import run as run_thresholds
from mousebrainbench.benchmarks.claimbench_component_ablation import run as run_ablation
from mousebrainbench.benchmarks.claimbench_threat_model import run as run_threat_model
from mousebrainbench.benchmarks.claimbench_unified_report import run as run_unified
from mousebrainbench.benchmarks.claimbench_v2_release import run as run_release
from mousebrainbench.benchmarks.cost_fidelity_claim_frontier import run as run_frontier
from mousebrainbench.benchmarks.external_benchmark_registry import run as run_registry
from mousebrainbench.benchmarks.external_causal_claim_validation import run as run_external_causal
from mousebrainbench.benchmarks.llm_claim_extraction_audit import run as run_llm_claims
from mousebrainbench.benchmarks.manuscript_claim_auditor import run as run_manuscript
from mousebrainbench.benchmarks.reviewer_attack_suite_v2 import run as run_reviewer
from mousebrainbench.benchmarks.scifact_claim_verification import run as run_scifact
from mousebrainbench.benchmarks.tuebingen_causal_direction import run as run_tuebingen
from mousebrainbench.benchmarks.uncertainty_claim_gate_v2 import run as run_uncertainty


DEFAULT_OUTPUT = Path("results/claimbench_reproduction_manifest/summary.json")
DEFAULT_MARKDOWN = Path("results/claimbench_reproduction_manifest/summary.md")


@dataclass(frozen=True)
class Stage:
    """One reproducible package stage."""

    name: str
    artifact: Path
    runner: Callable[..., Path]
    expected_decision: str
    data_root: Path | None = None


STAGES: tuple[Stage, ...] = (
    Stage(
        "adversarial_v2",
        Path("results/claim_adversarial_v2/summary.json"),
        run_adversarial,
        "claimbench_v2_blocks_overclaiming_under_broad_attacks",
    ),
    Stage(
        "threshold_sensitivity_v2",
        Path("results/claim_threshold_sensitivity_v2/summary.json"),
        run_thresholds,
        "claim_thresholds_have_nontrivial_safe_region_with_reportable_limits",
    ),
    Stage(
        "external_causal_synthetic",
        Path("results/external_causal_claim_validation/summary.json"),
        run_external_causal,
        "external_causal_validation_passed",
    ),
    Stage(
        "uncertainty_gate_v2",
        Path("results/uncertainty_claim_gate_v2/summary.json"),
        run_uncertainty,
        "uncertainty_gate_blocks_unsupported_support",
    ),
    Stage(
        "external_benchmark_registry",
        Path("results/external_benchmark_registry/summary.json"),
        run_registry,
        "external_benchmarks_registered_with_pending_data",
    ),
    Stage(
        "scifact_external_claims",
        Path("results/scifact_claim_verification/summary.json"),
        run_scifact,
        "scifact_external_claim_audit_ready",
        Path("data/external/scifact/data"),
    ),
    Stage(
        "tuebingen_causal_direction",
        Path("results/tuebingen_causal_direction/summary.json"),
        run_tuebingen,
        "tuebingen_external_direction_benchmark_ready",
        Path("data/external/tuebingen_cause_effect"),
    ),
    Stage(
        "manuscript_claim_audit",
        Path("results/manuscript_claim_audit/summary.json"),
        run_manuscript,
        "manuscript_claim_audit_passed",
    ),
    Stage(
        "llm_claim_extraction_audit",
        Path("results/llm_claim_extraction_audit/summary.json"),
        run_llm_claims,
        "llm_claim_extraction_layer_ready_non_authoritative",
    ),
    Stage(
        "cost_fidelity_frontier",
        Path("results/cost_fidelity_claim_frontier/summary.json"),
        run_frontier,
        "cost_fidelity_claim_frontier_built",
    ),
    Stage(
        "component_ablation",
        Path("results/claimbench_component_ablation/summary.json"),
        run_ablation,
        "claimbench_components_have_nonredundant_value",
    ),
    Stage(
        "reviewer_attack_v2",
        Path("results/reviewer_attack_suite_v2/summary.json"),
        run_reviewer,
        "reviewer_attack_suite_v2_passed_with_reportable_limits",
    ),
    Stage(
        "threat_model",
        Path("results/claimbench_threat_model/summary.json"),
        run_threat_model,
        "claimbench_threat_model_passed_with_boundaries",
    ),
    Stage(
        "unified_report",
        Path("results/claimbench_unified_report/summary.json"),
        run_unified,
        "claimbench_v2_methodological_package_ready",
    ),
    Stage(
        "release_check",
        Path("results/claimbench_v2_release/summary.json"),
        run_release,
        "claimbench_v2_release_ready",
    ),
)


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text()) if path.exists() else {}


def _run_stage(stage: Stage, root: Path) -> dict[str, Any]:
    output = root / stage.artifact
    markdown = output.with_suffix(".md")
    if stage.data_root is not None:
        stage.runner(root=root / stage.data_root, output=output, markdown=markdown)
    else:
        try:
            stage.runner(output=output, markdown=markdown, root=root)
        except TypeError:
            stage.runner(output=output, markdown=markdown)
    payload = _load(output)
    decision = payload.get("decision")
    return {
        "stage": stage.name,
        "artifact": str(stage.artifact),
        "expected_decision": stage.expected_decision,
        "observed_decision": decision,
        "passed": decision == stage.expected_decision,
        "git_revision": payload.get("git_revision"),
    }


def run(output: Path = DEFAULT_OUTPUT, markdown: Path = DEFAULT_MARKDOWN, root: Path = Path(".")) -> Path:
    """Regenerate the ClaimBench v2 package and write a manifest."""

    rows = [_run_stage(stage, root) for stage in STAGES]
    failed = [row for row in rows if not row["passed"]]
    payload = {
        "version": __version__,
        "git_revision": code_revision(),
        "analysis": "claimbench_reproduction_manifest",
        "num_stages": len(rows),
        "passed_stages": len(rows) - len(failed),
        "failed_stages": failed,
        "stages": rows,
        "decision": (
            "claimbench_reproduction_package_passed"
            if not failed
            else "claimbench_reproduction_package_requires_attention"
        ),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2))
    write_markdown(payload, markdown)
    return output


def write_markdown(payload: dict[str, Any], markdown: Path) -> None:
    """Write the reproduction manifest."""

    lines = [
        "# ClaimBench v2 Reproduction Manifest",
        "",
        f"- Decision: `{payload['decision']}`",
        f"- Stages passed: `{payload['passed_stages']}/{payload['num_stages']}`",
        "",
        "| Stage | Passed | Decision | Artifact |",
        "|---|---:|---|---|",
    ]
    for row in payload["stages"]:
        lines.append(
            f"| `{row['stage']}` | `{row['passed']}` | "
            f"`{row['observed_decision']}` | `{row['artifact']}` |"
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
