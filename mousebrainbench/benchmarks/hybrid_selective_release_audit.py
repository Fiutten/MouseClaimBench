"""Verify the immutable hybrid v2 artifact chain and publication boundary."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from mousebrainbench import __version__
from mousebrainbench.artifacts import code_revision


DEFAULT_PROTOCOL = Path("configs/benchmarks/hybrid_selective_claim_validation_v2.yaml")
DEFAULT_MATRIX_MANIFEST = Path("results/hybrid_development_features/summary.json")
DEFAULT_MATRIX = Path("results/hybrid_development_features/cases.npz")
DEFAULT_POLICY = Path("results/hybrid_selective_policy/model.json")
DEFAULT_CONFIRMATION = Path("results/hybrid_selective_confirmation/summary.json")
DEFAULT_CASES = Path("results/hybrid_selective_confirmation/cases.npz")
DEFAULT_OUTCOME_AUDIT = Path("results/hybrid_selective_outcome_audit/audit.json")
DEFAULT_OUTPUT = Path("results/hybrid_selective_release_audit/summary.json")
DEFAULT_MARKDOWN = Path("results/hybrid_selective_release_audit/summary.md")


def _sha256(path: Path) -> str:
    return f"sha256:{hashlib.sha256(path.read_bytes()).hexdigest()}"


def run(
    *,
    protocol_path: Path = DEFAULT_PROTOCOL,
    matrix_manifest_path: Path = DEFAULT_MATRIX_MANIFEST,
    matrix_path: Path = DEFAULT_MATRIX,
    policy_path: Path = DEFAULT_POLICY,
    confirmation_path: Path = DEFAULT_CONFIRMATION,
    cases_path: Path = DEFAULT_CASES,
    outcome_audit_path: Path = DEFAULT_OUTCOME_AUDIT,
    output: Path = DEFAULT_OUTPUT,
    markdown: Path = DEFAULT_MARKDOWN,
) -> Path:
    """Fail closed when provenance or the negative publication status changes."""

    manifest = json.loads(matrix_manifest_path.read_text())
    policy = json.loads(policy_path.read_text())
    confirmation = json.loads(confirmation_path.read_text())
    audit = json.loads(outcome_audit_path.read_text())
    checks = {
        "development_matrix_hash_matches_manifest": manifest["matrix_sha256"]
        == _sha256(matrix_path),
        "development_contains_no_v2_cases": manifest["confirmatory_v2_cases_used"] == 0,
        "policy_uses_frozen_protocol": policy["protocol_hash"] == _sha256(protocol_path),
        "policy_uses_frozen_matrix": policy["development_matrix_hash"]
        == _sha256(matrix_path),
        "policy_contains_no_v2_cases": policy["confirmatory_v2_cases_used"] == 0,
        "confirmation_uses_frozen_policy": confirmation["frozen_policy_hash"]
        == _sha256(policy_path),
        "confirmation_case_hash_matches": confirmation["cases_sha256"]
        == _sha256(cases_path),
        "confirmation_scale_is_complete": confirmation["scale_matches_frozen_protocol"],
        "confirmation_was_not_refitted": not confirmation[
            "confirmatory_model_refitting_performed"
        ],
        "primary_endpoint_is_preserved_as_negative": not confirmation[
            "primary_endpoint"
        ]["passed"],
        "outcome_audit_uses_frozen_confirmation": audit["source_summary_hash"]
        == _sha256(confirmation_path),
        "outcome_audit_uses_frozen_cases": audit["source_cases_hash"]
        == _sha256(cases_path),
        "outcome_audit_did_not_refit": not audit["model_refitted"],
        "outcome_audit_did_not_change_thresholds": not audit["thresholds_changed"],
        "strong_q1_claim_remains_blocked": not audit["publication_assessment"][
            "strong_new_q1_claim_supported"
        ],
        "all_source_revisions_are_clean": all(
            not str(payload["git_revision"]).endswith("-dirty")
            for payload in (manifest, policy, confirmation, audit)
        ),
    }
    passed = all(checks.values())
    payload = {
        "version": __version__,
        "git_revision": code_revision(),
        "analysis": "hybrid_selective_v2_release_audit",
        "checks": checks,
        "passed": passed,
        "q1_ready": False,
        "scientific_status": "negative_confirmation_with_partial_engineering_signal",
        "permitted_use": (
            "method-development evidence and reproducible negative-result analysis"
        ),
        "blocked_use": (
            "strong Q1 superiority, universal direction, causal identification, or "
            "independent-confirmation claims"
        ),
        "decision": (
            "hybrid_v2_reproducible_negative_release"
            if passed
            else "hybrid_v2_release_blocked_by_provenance_failure"
        ),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2))
    write_markdown(payload, markdown)
    if not passed:
        failed = [name for name, value in checks.items() if not value]
        raise RuntimeError(f"hybrid release audit failed: {failed}")
    return output


def write_markdown(payload: Mapping[str, Any], markdown: Path) -> None:
    lines = [
        "# Hybrid selective v2 release audit",
        "",
        f"- Decision: `{payload['decision']}`",
        f"- Provenance chain passed: `{payload['passed']}`",
        f"- Q1 ready: `{payload['q1_ready']}`",
        f"- Scientific status: `{payload['scientific_status']}`",
        "",
        "## Checks",
        "",
    ]
    lines.extend(f"- {name}: `{value}`" for name, value in payload["checks"].items())
    lines.extend(
        [
            "",
            "## Boundary",
            "",
            f"Permitted: {payload['permitted_use']}.",
            "",
            f"Blocked: {payload['blocked_use']}.",
            "",
        ]
    )
    markdown.parent.mkdir(parents=True, exist_ok=True)
    markdown.write_text("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--markdown", type=Path, default=DEFAULT_MARKDOWN)
    args = parser.parse_args()
    print(json.dumps({"output": str(run(output=args.output, markdown=args.markdown).resolve())}))


if __name__ == "__main__":
    main()
