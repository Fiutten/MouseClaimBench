"""Audit high-level digital-twin claims against explicit evidence gates.

The goal is not to reject ambitious models by default. The goal is to prevent
strong phrases such as "whole-brain digital twin" or "mechanistic explanation"
from being supported only by prediction accuracy or BOLD correlation. Each claim
type has non-interchangeable evidence requirements.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from mousebrainbench import __version__
from mousebrainbench.artifacts import code_revision


DEFAULT_OUTPUT = Path("results/digital_twin_claim_audit/summary.json")
DEFAULT_MARKDOWN = Path("results/digital_twin_claim_audit/summary.md")


@dataclass(frozen=True)
class ClaimGate:
    """One required evidence gate for a digital-twin claim."""

    name: str
    passed: bool
    reason: str

    def as_dict(self) -> dict[str, str | bool]:
        return {"name": self.name, "passed": self.passed, "reason": self.reason}


@dataclass(frozen=True)
class ClaimAudit:
    """Result for one audited claim family."""

    claim: str
    gates: tuple[ClaimGate, ...]
    allowed_wording: str
    blocked_wording: str

    @property
    def passed(self) -> bool:
        return all(gate.passed for gate in self.gates)

    def as_dict(self) -> dict[str, Any]:
        return {
            "claim": self.claim,
            "passed": self.passed,
            "gates": [gate.as_dict() for gate in self.gates],
            "allowed_wording": self.allowed_wording if self.passed else None,
            "blocked_wording": None if self.passed else self.blocked_wording,
        }


def audit_claims(evidence: dict[str, Any]) -> dict[str, Any]:
    """Evaluate digital-twin claim wording from structured evidence flags."""

    audits = (
        ClaimAudit(
            claim="whole_brain_digital_twin",
            gates=(
                ClaimGate(
                    "whole_brain_coverage",
                    bool(evidence.get("whole_brain_coverage", False)),
                    "Model covers the relevant whole-brain anatomical space.",
                ),
                ClaimGate(
                    "independent_whole_brain_validation",
                    bool(evidence.get("independent_whole_brain_validation", False)),
                    "Validation is not restricted to fitted or assimilated regions.",
                ),
                ClaimGate(
                    "reproducible_compute_budget",
                    bool(evidence.get("reproducible_compute_budget", False)),
                    "Compute and artifacts are reproducible by external groups.",
                ),
            ),
            allowed_wording="whole-brain mouse digital-twin model",
            blocked_wording="partial mouse-brain model or benchmark; not a complete whole-brain twin",
        ),
        ClaimAudit(
            claim="single_neuron_resolution_connectivity",
            gates=(
                ClaimGate(
                    "single_neuron_units",
                    bool(evidence.get("single_neuron_units", False)),
                    "The model represents individual neuron-like units.",
                ),
                ClaimGate(
                    "measured_synaptic_connectivity",
                    bool(evidence.get("measured_synaptic_connectivity", False)),
                    "Neuron-to-neuron edges are measured or independently validated, not only inferred.",
                ),
            ),
            allowed_wording="single-neuron connectivity model",
            blocked_wording="single-neuron-scale inferred connectivity, not a measured connectome",
        ),
        ClaimAudit(
            claim="mechanistic_identifiability",
            gates=(
                ClaimGate(
                    "predictive_or_reproducible",
                    bool(evidence.get("predictive_or_reproducible", False)),
                    "The model predicts or reproduces held-out neural observations.",
                ),
                ClaimGate(
                    "matched_nulls_passed",
                    bool(evidence.get("matched_nulls_passed", False)),
                    "The effect survives relevant spatial, degree, topology, or leakage controls.",
                ),
                ClaimGate(
                    "causal_or_interventional_evidence",
                    bool(evidence.get("causal_or_interventional_evidence", False)),
                    "Perturbation, lesion, causal, or held-out structural evidence supports mechanism.",
                ),
            ),
            allowed_wording="mechanistically identifiable model component",
            blocked_wording="predictive or structure-associated result, not causal mechanism",
        ),
        ClaimAudit(
            claim="behavioral_digital_twin",
            gates=(
                ClaimGate(
                    "behavior_above_chance",
                    bool(evidence.get("behavior_above_chance", False)),
                    "Behavioral decoding or control is above chance.",
                ),
                ClaimGate(
                    "behavior_competitive_with_empirical_or_baseline",
                    bool(evidence.get("behavior_competitive_with_empirical_or_baseline", False)),
                    "Behavioral performance is competitive with empirical decoding or strong baselines.",
                ),
                ClaimGate(
                    "held_out_behavior_protocol",
                    bool(evidence.get("held_out_behavior_protocol", False)),
                    "Behavior is evaluated on held-out sessions, animals, or conditions.",
                ),
            ),
            allowed_wording="behaviorally validated digital-twin component",
            blocked_wording="task-related neural dynamics with limited behavioral decoding",
        ),
    )
    passed = [audit.claim for audit in audits if audit.passed]
    blocked = [audit.claim for audit in audits if not audit.passed]
    return {
        "version": __version__,
        "git_revision": code_revision(),
        "analysis": "digital_twin_claim_audit",
        "passed_claims": passed,
        "blocked_claims": blocked,
        "audits": [audit.as_dict() for audit in audits],
        "interpretation": (
            "Digital-twin wording is accepted only when the corresponding evidence "
            "gates pass. Prediction, BOLD correlation, and scale do not substitute "
            "for independent structure, causal, or reproducibility evidence."
        ),
    }


def mousebrainbench_current_evidence() -> dict[str, Any]:
    """Evidence flags supported by the current MouseBrainBench artifact set."""

    stratified_path = Path("results/microns_structure_function_pilot/stratified_summary.json")
    stratified = json.loads(stratified_path.read_text()) if stratified_path.exists() else {}
    holdout_path = Path(
        "results/microns_structure_function_pilot/stratified_holdout_offset1000_summary.json"
    )
    holdout = json.loads(holdout_path.read_text()) if holdout_path.exists() else {}
    publication_path = Path("results/publication_freeze/summary.json")
    publication = json.loads(publication_path.read_text()) if publication_path.exists() else {}
    return {
        "whole_brain_coverage": False,
        "independent_whole_brain_validation": False,
        "reproducible_compute_budget": True,
        "single_neuron_units": True,
        "measured_synaptic_connectivity": bool(
            stratified.get("n_connected_edge_pairs", 0)
            and stratified.get("positive_stratified_structure_function_result", False)
        ),
        "predictive_or_reproducible": bool(publication.get("methodological_paper_ready", False)),
        "matched_nulls_passed": bool(
            stratified.get("positive_stratified_structure_function_result", False)
            and holdout.get("positive_stratified_structure_function_result", False)
        ),
        "causal_or_interventional_evidence": False,
        "behavior_above_chance": False,
        "behavior_competitive_with_empirical_or_baseline": False,
        "held_out_behavior_protocol": False,
    }


def write_outputs(payload: dict[str, Any], output: Path, markdown: Path) -> None:
    """Write JSON and Markdown claim-audit artifacts."""

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2))
    lines = [
        "# Digital-Twin Claim Audit",
        "",
        f"- Passed claims: `{len(payload['passed_claims'])}`",
        f"- Blocked claims: `{len(payload['blocked_claims'])}`",
        "",
        "## Audited Claims",
        "",
        "| Claim | Passed | Allowed wording | Blocked wording |",
        "|---|---:|---|---|",
    ]
    for audit in payload["audits"]:
        lines.append(
            "| "
            f"`{audit['claim']}` | `{audit['passed']}` | "
            f"{audit['allowed_wording'] or ''} | {audit['blocked_wording'] or ''} |"
        )
    lines.extend(["", "## Interpretation", "", str(payload["interpretation"]), ""])
    markdown.parent.mkdir(parents=True, exist_ok=True)
    markdown.write_text("\n".join(lines))


def run(output: Path = DEFAULT_OUTPUT, markdown: Path = DEFAULT_MARKDOWN) -> dict[str, Any]:
    """Audit current MouseBrainBench claims and write tracked artifacts."""

    payload = audit_claims(mousebrainbench_current_evidence())
    write_outputs(payload, output, markdown)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--markdown", type=Path, default=DEFAULT_MARKDOWN)
    args = parser.parse_args()
    payload = run(args.output, args.markdown)
    print(json.dumps({"output": str(args.output), "blocked_claims": payload["blocked_claims"]}))


if __name__ == "__main__":
    main()
