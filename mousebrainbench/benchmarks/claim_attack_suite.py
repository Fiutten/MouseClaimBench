"""Executable reviewer-attack suite for MouseBrainBench claims.

The suite aggregates already-generated or lightweight artifacts and emits a
single risk report. It is designed for internal scientific discipline: if a
claim becomes unsupported, the report should expose that before a reviewer does.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from mousebrainbench import __version__
from mousebrainbench.artifacts import code_revision
from mousebrainbench.benchmarks.claim_adversarial import run as run_adversarial


DEFAULT_OUTPUT = Path("results/claim_attack_suite/summary.json")
DEFAULT_MARKDOWN = Path("results/claim_attack_suite/summary.md")


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text()) if path.exists() else {}


def _risk(level: str, item: str, evidence: str, action: str) -> dict[str, str]:
    return {"level": level, "item": item, "evidence": evidence, "action": action}


def run(output: Path = DEFAULT_OUTPUT, markdown: Path = DEFAULT_MARKDOWN) -> Path:
    """Run lightweight claim attacks and summarize current risk."""

    adversarial_path = Path("results/claim_adversarial_benchmark/summary.json")
    if not adversarial_path.exists():
        run_adversarial()
    adversarial = _load(adversarial_path)
    mis2 = _load(Path("results/mis2_threshold_sensitivity/summary.json"))
    freeze = _load(Path("results/publication_freeze/summary.json"))
    digital = _load(Path("results/digital_twin_claim_audit/summary.json"))
    microns = _load(Path("results/microns_q1_package/summary.json"))
    microns_robustness = _load(Path("results/microns_primary_robustness/summary.json"))
    sensorium = _load(Path("results/sensorium_official_baseline_audit/summary.json"))
    allen = _load(Path("results/allen_vbn_mechanistic_identifiability_score.json"))

    risks = []
    gate_row = next(
        (
            row
            for row in adversarial.get("aggregate_by_evaluator", [])
            if row.get("evaluator") == "claim_gate"
        ),
        {},
    )
    if gate_row.get("fp", 1) != 0:
        risks.append(
            _risk(
                "high",
                "claim_gate_false_positives",
                f"claim_gate FP={gate_row.get('fp')}",
                "Revise non-compensatory claim gates before extending claims.",
            )
        )
    if mis2.get("phase_counts", {}).get("dangerous", 1) or mis2.get("phase_counts", {}).get(
        "unstable", 1
    ):
        risks.append(
            _risk(
                "high",
                "mis2_unsafe_region",
                f"phase_counts={mis2.get('phase_counts')}",
                "Do not present MIS 2.0 as stable until unsafe regions are explained.",
            )
        )
    if "whole_brain_digital_twin" not in digital.get("blocked_claims", []):
        risks.append(
            _risk(
                "high",
                "whole_brain_claim_not_blocked",
                "digital twin audit does not block whole-brain wording",
                "Restore explicit whole-brain digital-twin block.",
            )
        )
    if not microns.get("q1_package_ready", False):
        risks.append(
            _risk(
                "medium",
                "microns_package_not_ready",
                "MICRONS Q1 package is not ready",
                "Downgrade MICRONS from primary positive evidence.",
            )
        )
    if microns_robustness and not microns_robustness.get("all_cohorts_robust", False):
        risks.append(
            _risk(
                "high",
                "microns_primary_endpoint_not_robust",
                f"decision={microns_robustness.get('decision')}",
                "Downgrade MICRONS primary endpoint or report it as control-sensitive.",
            )
        )
    if sensorium.get("official_q1_baseline_qualified", False):
        sensorium_action = "Sensorium official baseline may be discussed with exact scope."
    else:
        sensorium_action = "Keep Sensorium as predictive/interoperability evidence, not SOTA."
        risks.append(
            _risk(
                "medium",
                "sensorium_not_q1_qualified",
                "official Sensorium baseline is not Q1-qualified locally",
                sensorium_action,
            )
        )
    allen_decision = allen.get("decision")
    if allen_decision != "reproducible_target_without_mechanistic_identifiability":
        risks.append(
            _risk(
                "medium",
                "allen_negative_control_changed",
                f"Allen decision={allen_decision}",
                "Re-check manuscript framing of Allen as negative mechanistic control.",
            )
        )

    allowed_claims = freeze.get("claims_allowed", [])
    blocked_claims = freeze.get("claims_blocked", [])
    payload = {
        "version": __version__,
        "git_revision": code_revision(),
        "analysis": "claim_attack_suite",
        "inputs": {
            "adversarial": str(adversarial_path),
            "mis2_threshold_sensitivity": "results/mis2_threshold_sensitivity/summary.json",
            "publication_freeze": "results/publication_freeze/summary.json",
            "digital_twin_claim_audit": "results/digital_twin_claim_audit/summary.json",
            "microns_q1_package": "results/microns_q1_package/summary.json",
            "microns_primary_robustness": "results/microns_primary_robustness/summary.json",
            "sensorium_official_baseline_audit": (
                "results/sensorium_official_baseline_audit/summary.json"
            ),
            "allen_vbn_mis": "results/allen_vbn_mechanistic_identifiability_score.json",
        },
        "allowed_claims": [
            *allowed_claims,
            *(
                [
                    "MICRONS primary endpoint survives combined distance/degree matching and within-distance readout shuffling."
                ]
                if microns_robustness.get("all_cohorts_robust", False)
                else []
            ),
        ],
        "blocked_claims": blocked_claims,
        "risks": risks,
        "decision": "claim_attack_suite_passed_with_known_limits"
        if not any(risk["level"] == "high" for risk in risks)
        else "claim_attack_suite_blocks_release",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2))
    write_markdown(payload, markdown)
    return output


def write_markdown(payload: dict[str, Any], markdown: Path) -> None:
    """Write reviewer risk report."""

    lines = [
        "# Claim Attack Suite",
        "",
        f"- Decision: `{payload['decision']}`",
        f"- Risks: `{len(payload['risks'])}`",
        "",
        "## Risks",
        "",
        "| Level | Item | Evidence | Action |",
        "|---|---|---|---|",
    ]
    if payload["risks"]:
        for risk in payload["risks"]:
            lines.append(
                f"| `{risk['level']}` | `{risk['item']}` | {risk['evidence']} | {risk['action']} |"
            )
    else:
        lines.append("| `none` | `none` | No blocking risks detected. | Continue. |")

    lines.extend(["", "## Allowed Claims", ""])
    lines.extend(f"- {claim}" for claim in payload["allowed_claims"])
    lines.extend(["", "## Blocked Claims", ""])
    lines.extend(f"- {claim}" for claim in payload["blocked_claims"])
    lines.append("")
    markdown.parent.mkdir(parents=True, exist_ok=True)
    markdown.write_text("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--markdown", type=Path, default=DEFAULT_MARKDOWN)
    args = parser.parse_args()
    print(json.dumps({"output": str(run(args.output, args.markdown).resolve())}))


if __name__ == "__main__":
    main()
