"""Release gate for the scientifically hardened ClaimBench v3 package."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from mousebrainbench import __version__
from mousebrainbench.artifacts import code_revision


DEFAULT_OUTPUT = Path("results/claimbench_v3_release/summary.json")
DEFAULT_MARKDOWN = Path("results/claimbench_v3_release/summary.md")

REQUIREMENTS = {
    "results/oracle_sem_claim_benchmark/summary.json": (
        "decision",
        "oracle_benchmark_supports_non_compensatory_contract_with_finite_sample_errors",
    ),
    "results/real_case_claim_matrix/summary.json": (
        "decision",
        "artifact_grounded_case_matrix_complete_with_explicit_limits",
    ),
    "results/human_evaluation_protocol/study_package.json": (
        "decision",
        "human_study_protocol_ready_for_ethics_and_preregistration",
    ),
    "results/scifact_claim_verification/summary.json": (
        "decision",
        "scifact_external_claim_audit_ready",
    ),
    "results/tuebingen_causal_direction/summary.json": (
        "decision",
        "tuebingen_external_direction_benchmark_ready",
    ),
    "results/claim_adversarial_v2/summary.json": (
        "validation_role",
        "software_contract_conformance_not_independent_validation",
    ),
}


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def run(
    output: Path = DEFAULT_OUTPUT,
    markdown: Path = DEFAULT_MARKDOWN,
    root: Path = Path("."),
) -> Path:
    """Check artifact completeness while preserving unresolved scientific limits."""

    rows = []
    missing = []
    failing = []
    dirty = []
    for relative, (field, expected) in REQUIREMENTS.items():
        path = root / relative
        if not path.exists():
            missing.append(relative)
            rows.append(
                {
                    "artifact": relative,
                    "exists": False,
                    "field": field,
                    "expected": expected,
                    "observed": None,
                    "git_revision": None,
                }
            )
            continue
        payload = _load(path)
        observed = payload.get(field)
        revision = payload.get("git_revision")
        if observed != expected:
            failing.append(relative)
        if isinstance(revision, str) and revision.endswith("-dirty"):
            dirty.append(relative)
        rows.append(
            {
                "artifact": relative,
                "exists": True,
                "field": field,
                "expected": expected,
                "observed": observed,
                "git_revision": revision,
            }
        )

    human_payload = (
        _load(root / "results/human_evaluation_protocol/study_package.json")
        if (root / "results/human_evaluation_protocol/study_package.json").exists()
        else {}
    )
    human_not_executed = (
        human_payload.get("study_status") == "not_executed"
        and human_payload.get("results_available") is False
    )
    if not human_not_executed and human_payload:
        failing.append("results/human_evaluation_protocol/study_package.json")

    blockers = [
        {
            "claim": "improved human decision quality",
            "reason": "the expert study is prepared but has not been executed",
        },
        {
            "claim": "external biological replication",
            "reason": "the real cases do not include an independent cross-resource replication",
        },
        {
            "claim": "complete or causal mouse-brain digital twin",
            "reason": "no real case supplies the required whole-brain and intervention blocks",
        },
    ]
    ready = not missing and not failing and not dirty and human_not_executed
    payload = {
        "version": __version__,
        "git_revision": code_revision(),
        "analysis": "claimbench_v3_scientific_hardening_release",
        "required_artifacts": rows,
        "missing_artifacts": sorted(set(missing)),
        "failing_artifacts": sorted(set(failing)),
        "dirty_artifacts": sorted(set(dirty)),
        "human_study_executed": False,
        "scientific_claim_blockers": blockers,
        "decision": (
            "methodological_package_ready_human_effect_unvalidated"
            if ready
            else "claimbench_v3_release_requires_action"
        ),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2))
    write_markdown(payload, markdown)
    return output


def write_markdown(payload: dict[str, Any], markdown: Path) -> None:
    lines = [
        "# ClaimBench v3 Scientific-Hardening Release",
        "",
        f"- Decision: `{payload['decision']}`",
        f"- Missing artifacts: `{len(payload['missing_artifacts'])}`",
        f"- Failing artifacts: `{len(payload['failing_artifacts'])}`",
        f"- Dirty artifacts: `{len(payload['dirty_artifacts'])}`",
        f"- Human study executed: `{payload['human_study_executed']}`",
        "",
        "## Claims that remain blocked",
        "",
    ]
    lines.extend(
        f"- **{row['claim']}**: {row['reason']}"
        for row in payload["scientific_claim_blockers"]
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
