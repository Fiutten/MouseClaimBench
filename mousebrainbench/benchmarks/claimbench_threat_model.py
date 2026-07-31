"""Executable reviewer threat model for ClaimBench v2.

The threat model is deliberately explicit.  Each threat states what a skeptical
reviewer could attack, which artifact answers it, and which claim boundary must
remain in place.  This turns reviewer-risk management into a reproducible object
rather than a narrative written after the fact.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from mousebrainbench import __version__
from mousebrainbench.artifacts import code_revision


DEFAULT_OUTPUT = Path("results/claimbench_threat_model/summary.json")
DEFAULT_MARKDOWN = Path("results/claimbench_threat_model/summary.md")


@dataclass(frozen=True)
class Threat:
    """One reviewer-facing methodological threat."""

    threat_id: str
    reviewer_attack: str
    required_artifact: str
    pass_decision: str
    boundary: str
    severity: str


THREATS: tuple[Threat, ...] = (
    Threat(
        threat_id="rules_without_value",
        reviewer_attack="The framework is only a hand-written rule table.",
        required_artifact="results/claimbench_component_ablation/summary.json",
        pass_decision="claimbench_components_have_nonredundant_value",
        boundary="Claim non-redundant audit value, not superior prediction.",
        severity="critical",
    ),
    Threat(
        threat_id="threshold_arbitrariness",
        reviewer_attack="The result depends on arbitrary thresholds.",
        required_artifact="results/claim_threshold_sensitivity_v2/summary.json",
        pass_decision="claim_thresholds_have_nontrivial_safe_region_with_reportable_limits",
        boundary="Report safe and dangerous regions; do not claim universal threshold robustness.",
        severity="high",
    ),
    Threat(
        threat_id="synthetic_overfit",
        reviewer_attack="The gate only works on hand-crafted synthetic cases.",
        required_artifact="results/scifact_claim_verification/summary.json",
        pass_decision="scifact_external_claim_audit_ready",
        boundary="Use SciFact as external claim-auditing evidence, not SOTA verification.",
        severity="high",
    ),
    Threat(
        threat_id="causal_overclaim",
        reviewer_attack="The method mistakes correlation for causal direction.",
        required_artifact="results/tuebingen_causal_direction/summary.json",
        pass_decision="tuebingen_external_direction_benchmark_ready",
        boundary="Use Tuebingen to block causal overclaims, not to claim causal-discovery performance.",
        severity="critical",
    ),
    Threat(
        threat_id="wording_drift",
        reviewer_attack="The manuscript says more than the artifacts support.",
        required_artifact="results/manuscript_claim_audit/summary.json",
        pass_decision="manuscript_claim_audit_passed",
        boundary="Keep manuscript wording tied to executable claim contracts.",
        severity="critical",
    ),
    Threat(
        threat_id="llm_authority_drift",
        reviewer_attack="The LLM becomes an unverified judge of scientific claims.",
        required_artifact="results/llm_claim_extraction_audit/summary.json",
        pass_decision="llm_claim_extraction_layer_ready_non_authoritative",
        boundary="Use LLMs only for candidate extraction and conservative wording support.",
        severity="critical",
    ),
    Threat(
        threat_id="uncertainty_hidden",
        reviewer_attack="Borderline evidence is forced into binary support.",
        required_artifact="results/uncertainty_claim_gate_v2/summary.json",
        pass_decision="uncertainty_gate_blocks_unsupported_support",
        boundary="Uncertain claims remain uncertain; uncertainty cannot become support.",
        severity="high",
    ),
)


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text()) if path.exists() else {}


def _artifact_passed(root: Path, threat: Threat) -> tuple[bool, str | None]:
    path = root / threat.required_artifact
    payload = _load(path)
    decision = payload.get("decision")
    return decision == threat.pass_decision, decision


def run(output: Path = DEFAULT_OUTPUT, markdown: Path = DEFAULT_MARKDOWN, root: Path = Path(".")) -> Path:
    """Evaluate reviewer threats against current artifacts."""

    rows: list[dict[str, Any]] = []
    for threat in THREATS:
        passed, observed = _artifact_passed(root, threat)
        rows.append(
            {
                "threat_id": threat.threat_id,
                "severity": threat.severity,
                "reviewer_attack": threat.reviewer_attack,
                "required_artifact": threat.required_artifact,
                "expected_decision": threat.pass_decision,
                "observed_decision": observed,
                "passed": passed,
                "claim_boundary": threat.boundary,
            }
        )
    failed = [row for row in rows if not row["passed"]]
    critical_failed = [row for row in failed if row["severity"] == "critical"]
    payload = {
        "version": __version__,
        "git_revision": code_revision(),
        "analysis": "claimbench_threat_model",
        "num_threats": len(rows),
        "passed_threats": len(rows) - len(failed),
        "failed_threats": failed,
        "critical_failed_threats": critical_failed,
        "rows": rows,
        "decision": (
            "claimbench_threat_model_passed_with_boundaries"
            if not critical_failed
            else "claimbench_threat_model_blocks_release"
        ),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2))
    write_markdown(payload, markdown)
    return output


def write_markdown(payload: dict[str, Any], markdown: Path) -> None:
    """Write reviewer threat model report."""

    lines = [
        "# ClaimBench v2 Threat Model",
        "",
        f"- Decision: `{payload['decision']}`",
        f"- Threats passed: `{payload['passed_threats']}/{payload['num_threats']}`",
        f"- Critical failed threats: `{len(payload['critical_failed_threats'])}`",
        "",
        "| Threat | Severity | Passed | Artifact | Boundary |",
        "|---|---|---:|---|---|",
    ]
    for row in payload["rows"]:
        lines.append(
            f"| `{row['threat_id']}` | `{row['severity']}` | `{row['passed']}` | "
            f"`{row['required_artifact']}` | {row['claim_boundary']} |"
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
