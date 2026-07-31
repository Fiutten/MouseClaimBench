"""Reviewer attack suite v2 for claim-aware MouseBrainBench studies."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from mousebrainbench import __version__
from mousebrainbench.artifacts import code_revision
from mousebrainbench.benchmarks.claim_adversarial_v2 import run as run_adversarial_v2
from mousebrainbench.benchmarks.claim_threshold_sensitivity_v2 import run as run_sensitivity_v2
from mousebrainbench.benchmarks.external_causal_claim_validation import run as run_external_causal


DEFAULT_OUTPUT = Path("results/reviewer_attack_suite_v2/summary.json")
DEFAULT_MARKDOWN = Path("results/reviewer_attack_suite_v2/summary.md")


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text()) if path.exists() else {}


def _risk(level: str, attack: str, evidence: str, response: str) -> dict[str, str]:
    return {"level": level, "attack": attack, "evidence": evidence, "response": response}


def _ensure_inputs(root: Path = Path(".")) -> dict[str, dict[str, Any]]:
    paths = {
        "adversarial_v2": root / "results/claim_adversarial_v2/summary.json",
        "threshold_sensitivity_v2": root / "results/claim_threshold_sensitivity_v2/summary.json",
        "external_causal": root / "results/external_causal_claim_validation/summary.json",
        "claim_ledger": root / "results/claim_ledger/summary.json",
        "release": root / "results/claimbench_release/summary.json",
        "manuscript_claim_audit": root / "results/manuscript_claim_audit/summary.json",
        "scifact": root / "results/scifact_claim_verification/summary.json",
        "tuebingen": root / "results/tuebingen_causal_direction/summary.json",
    }
    if not paths["adversarial_v2"].exists():
        run_adversarial_v2(
            output=paths["adversarial_v2"],
            markdown=root / "results/claim_adversarial_v2/summary.md",
        )
    if not paths["threshold_sensitivity_v2"].exists():
        run_sensitivity_v2(
            output=paths["threshold_sensitivity_v2"],
            markdown=root / "results/claim_threshold_sensitivity_v2/summary.md",
        )
    if not paths["external_causal"].exists():
        run_external_causal(
            output=paths["external_causal"],
            markdown=root / "results/external_causal_claim_validation/summary.md",
        )
    return {name: _load(path) for name, path in paths.items()}


def run(
    output: Path = DEFAULT_OUTPUT,
    markdown: Path = DEFAULT_MARKDOWN,
    root: Path = Path("."),
) -> Path:
    """Run reviewer attack checks for the v2 extension."""

    inputs = _ensure_inputs(root)
    adversarial = inputs["adversarial_v2"]
    sensitivity = inputs["threshold_sensitivity_v2"]
    external = inputs["external_causal"]
    ledger = inputs["claim_ledger"]
    release = inputs["release"]
    manuscript = inputs["manuscript_claim_audit"]
    scifact = inputs["scifact"]
    tuebingen = inputs["tuebingen"]
    risks: list[dict[str, str]] = []

    gate_v2 = next(
        (
            row
            for row in adversarial.get("aggregate_by_evaluator", [])
            if row.get("evaluator") == "claim_gate"
        ),
        {},
    )
    if gate_v2.get("fp", 1) != 0:
        risks.append(
            _risk(
                "high",
                "The gate overclaims in synthetic attacks.",
                f"ClaimBench v2 gate FP={gate_v2.get('fp')}",
                "Revise gate semantics before using v2 in a revision.",
            )
        )
    if float(gate_v2.get("conservativeness_index", 1.0)) > 0.35:
        risks.append(
            _risk(
                "medium",
                "The gate is too conservative to be useful.",
                f"ClaimBench v2 gate CI={gate_v2.get('conservativeness_index')}",
                "Report the conservativeness tradeoff explicitly.",
            )
        )
    if sensitivity.get("safe_cells", 0) < 20:
        risks.append(
            _risk(
                "high",
                "The thresholds are arbitrary and fragile.",
                f"safe_cells={sensitivity.get('safe_cells')}",
                "Do not claim threshold stability without additional calibration.",
            )
        )
    if sensitivity.get("dangerous_cells", 1) != 0:
        risks.append(
            _risk(
                "medium",
                "Some threshold cells authorize unsupported claims.",
                f"dangerous_cells={sensitivity.get('dangerous_cells')}",
                "Report the dangerous region and keep nominal thresholds fixed.",
            )
        )
    if external.get("decision") != "external_causal_validation_passed":
        risks.append(
            _risk(
                "high",
                "The method is overfit to neuro datasets.",
                f"external decision={external.get('decision')}",
                "Repair external causal validation before claiming generality.",
            )
        )
    if ledger.get("decision") != "claim_ledger_supported":
        risks.append(
            _risk(
                "medium",
                "The manuscript claims are not fully linked to artifacts.",
                f"ledger decision={ledger.get('decision')}",
                "Update the claim ledger or downgrade unsupported wording.",
            )
        )
    if release.get("decision") != "claimbench_release_ready":
        risks.append(
            _risk(
                "medium",
                "The baseline release is not clean.",
                f"release decision={release.get('decision')}",
                "Regenerate release artifacts from a clean commit.",
            )
        )
    if manuscript and manuscript.get("decision") != "manuscript_claim_audit_passed":
        risks.append(
            _risk(
                "high",
                "The manuscript contains unsupported or risky wording.",
                (
                    f"active_risk_patterns="
                    f"{len(manuscript.get('active_risk_pattern_hits', []))}; "
                    f"unsupported={manuscript.get('unsupported_present_claims', [])}"
                ),
                "Downgrade wording or add non-compensatory evidence before submission.",
            )
        )
    if scifact.get("decision") != "scifact_external_claim_audit_ready":
        risks.append(
            _risk(
                "medium",
                "External scientific claim-verification control is missing or insufficient.",
                f"SciFact decision={scifact.get('decision')}",
                "Keep SciFact out of the main contribution or regenerate the adapter.",
            )
        )
    elif float(scifact.get("shortcut_overclaiming_risk", 0.0)) <= 0.0:
        risks.append(
            _risk(
                "medium",
                "SciFact does not expose a measurable lexical overclaiming risk.",
                f"shortcut ORI={scifact.get('shortcut_overclaiming_risk')}",
                "Re-check thresholds or present SciFact only as interoperability evidence.",
            )
        )
    if tuebingen.get("decision") != "tuebingen_external_direction_benchmark_ready":
        risks.append(
            _risk(
                "medium",
                "External causal-direction control is missing or insufficient.",
                f"Tuebingen decision={tuebingen.get('decision')}",
                "Keep causal-direction claims synthetic-only or regenerate public data.",
            )
        )
    elif int(tuebingen.get("correlation_only_direction_overclaims", 0)) <= 0:
        risks.append(
            _risk(
                "medium",
                "Tuebingen does not expose correlation-only directional overclaiming.",
                f"overclaims={tuebingen.get('correlation_only_direction_overclaims')}",
                "Do not use Tuebingen as a reviewer-facing causal overclaiming control.",
            )
        )

    payload = {
        "version": __version__,
        "git_revision": code_revision(),
        "analysis": "reviewer_attack_suite_v2",
        "inputs": {
            "adversarial_v2": "results/claim_adversarial_v2/summary.json",
            "threshold_sensitivity_v2": "results/claim_threshold_sensitivity_v2/summary.json",
            "external_causal": "results/external_causal_claim_validation/summary.json",
            "claim_ledger": "results/claim_ledger/summary.json",
            "release": "results/claimbench_release/summary.json",
            "manuscript_claim_audit": "results/manuscript_claim_audit/summary.json",
            "scifact": "results/scifact_claim_verification/summary.json",
            "tuebingen": "results/tuebingen_causal_direction/summary.json",
        },
        "external_controls": {
            "scifact_decision": scifact.get("decision"),
            "scifact_shortcut_overclaiming_risk": scifact.get("shortcut_overclaiming_risk"),
            "tuebingen_decision": tuebingen.get("decision"),
            "tuebingen_correlation_only_direction_overclaims": tuebingen.get(
                "correlation_only_direction_overclaims"
            ),
        },
        "risks": risks,
        "decision": (
            "reviewer_attack_suite_v2_passed_with_reportable_limits"
            if not any(risk["level"] == "high" for risk in risks)
            else "reviewer_attack_suite_v2_blocks_revision_use"
        ),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2))
    write_markdown(payload, markdown)
    return output


def write_markdown(payload: dict[str, Any], markdown: Path) -> None:
    """Write reviewer-facing attack report."""

    lines = [
        "# Reviewer Attack Suite v2",
        "",
        f"- Decision: `{payload['decision']}`",
        f"- Risks: `{len(payload['risks'])}`",
        "",
        "| Level | Reviewer attack | Evidence | Response |",
        "|---|---|---|---|",
    ]
    if payload["risks"]:
        for risk in payload["risks"]:
            lines.append(
                f"| `{risk['level']}` | {risk['attack']} | {risk['evidence']} | {risk['response']} |"
            )
    else:
        lines.append("| `none` | No blocking reviewer attack detected. | All checks passed. | Continue. |")
    controls = payload.get("external_controls", {})
    lines.extend(
        [
            "",
            "## External Controls",
            "",
            f"- SciFact decision: `{controls.get('scifact_decision')}`",
            f"- SciFact shortcut ORI: `{controls.get('scifact_shortcut_overclaiming_risk')}`",
            f"- Tuebingen decision: `{controls.get('tuebingen_decision')}`",
            "- Tuebingen correlation-only direction overclaims: "
            f"`{controls.get('tuebingen_correlation_only_direction_overclaims')}`",
        ]
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
