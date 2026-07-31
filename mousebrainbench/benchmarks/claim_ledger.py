"""Build a reproducible claim ledger from MouseBrainBench artifacts."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from mousebrainbench import __version__
from mousebrainbench.artifacts import code_revision


DEFAULT_OUTPUT = Path("results/claim_ledger/summary.json")
DEFAULT_MARKDOWN = Path("results/claim_ledger/claim_audit_report.md")


@dataclass(frozen=True)
class ClaimLedgerEntry:
    """One manuscript-level claim linked to an executable artifact."""

    claim_id: str
    permitted_wording: str
    blocked_wording: tuple[str, ...]
    evidence_artifact: str
    required_decision: str
    status: str
    rationale: str


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def _status(actual: object, expected: object) -> str:
    return "supported" if actual == expected else "not_supported"


def build_entries(root: Path = Path(".")) -> tuple[ClaimLedgerEntry, ...]:
    """Build the current deterministic claim ledger."""

    allen = _load_json(root / "results/allen_vbn_mechanistic_identifiability_score.json")
    sensorium = _load_json(root / "results/sensorium_official_baseline_audit/summary.json")
    dynamic = _load_json(root / "results/dynamic_sensorium_model_comparator/summary.json")
    microns = _load_json(root / "results/microns_primary_robustness/summary.json")
    adversarial = _load_json(root / "results/claim_adversarial_benchmark/summary.json")
    attack = _load_json(root / "results/claim_attack_suite/summary.json")

    return (
        ClaimLedgerEntry(
            claim_id="allen-negative-identifiability",
            permitted_wording="Allen VBN is a real negative mechanistic-identifiability case.",
            blocked_wording=(
                "Allen VBN validates a mechanistic brain model.",
                "Reproducibility alone establishes mechanism.",
            ),
            evidence_artifact="results/allen_vbn_mechanistic_identifiability_score.json",
            required_decision="reproducible_target_without_mechanistic_identifiability",
            status=_status(
                allen.get("decision"),
                "reproducible_target_without_mechanistic_identifiability",
            ),
            rationale="The target is reproducible, but topology and direction are not identified.",
        ),
        ClaimLedgerEntry(
            claim_id="sensorium-predictive-interoperability",
            permitted_wording=(
                "Sensorium/Dynamic Sensorium are predictive and interoperability cases."
            ),
            blocked_wording=(
                "Sensorium proves mechanistic identifiability.",
                "MouseBrainBench is a Sensorium SOTA model.",
            ),
            evidence_artifact="results/sensorium_official_baseline_audit/summary.json",
            required_decision="official_sensorium_bounded_trained_baseline_available_not_q1_qualified",
            status=_status(
                sensorium.get("decision"),
                "official_sensorium_bounded_trained_baseline_available_not_q1_qualified",
            ),
            rationale="The official ecosystem can be audited, but the local run is not a SOTA claim.",
        ),
        ClaimLedgerEntry(
            claim_id="dynamic-sensorium-predictive-only",
            permitted_wording="Dynamic Sensorium is used as a temporal predictive case.",
            blocked_wording=(
                "Dynamic Sensorium validates causal mechanism.",
                "Temporal prediction establishes a digital twin.",
            ),
            evidence_artifact="results/dynamic_sensorium_model_comparator/summary.json",
            required_decision="dynamic_sensorium_predictive_model_comparator",
            status=_status(
                dynamic.get("comparison"),
                "dynamic_sensorium_predictive_model_comparator",
            ),
            rationale="The comparator tracks prediction and temporal modeling, not causal circuits.",
        ),
        ClaimLedgerEntry(
            claim_id="microns-local-structure-function",
            permitted_wording=(
                "MICRONS supports a local observational structure-function case at the fixed endpoint."
            ),
            blocked_wording=(
                "MICRONS proves causality in this study.",
                "MICRONS validates a whole-brain mouse digital twin.",
            ),
            evidence_artifact="results/microns_primary_robustness/summary.json",
            required_decision="microns_primary_endpoint_survives_harder_controls",
            status=_status(
                microns.get("decision"),
                "microns_primary_endpoint_survives_harder_controls",
            ),
            rationale="The endpoint survives harder controls but remains local and observational.",
        ),
        ClaimLedgerEntry(
            claim_id="claim-gate-blocks-overclaiming",
            permitted_wording="The non-compensatory gate blocks broad adversarial overclaims.",
            blocked_wording=(
                "The gate is universally optimal.",
                "The adversarial suite proves biological truth.",
            ),
            evidence_artifact="results/claim_adversarial_benchmark/summary.json",
            required_decision="claim_gate_blocks_broad_adversarial_overclaims",
            status=_status(
                adversarial.get("decision"),
                "claim_gate_blocks_broad_adversarial_overclaims",
            ),
            rationale="Synthetic cases test claim authorization under known truth.",
        ),
        ClaimLedgerEntry(
            claim_id="attack-suite-known-limits",
            permitted_wording="The current release passes known attack checks with declared limits.",
            blocked_wording=(
                "All external Q1 pieces are fully solved.",
                "There are no remaining methodological limitations.",
            ),
            evidence_artifact="results/claim_attack_suite/summary.json",
            required_decision="claim_attack_suite_passed_with_known_limits",
            status=_status(
                attack.get("decision"),
                "claim_attack_suite_passed_with_known_limits",
            ),
            rationale="Known medium-risk limitations are part of the release claim.",
        ),
    )


def run(output: Path = DEFAULT_OUTPUT, markdown: Path = DEFAULT_MARKDOWN, root: Path = Path(".")) -> Path:
    """Write JSON and Markdown claim-ledger artifacts."""

    entries = build_entries(root)
    supported = sum(entry.status == "supported" for entry in entries)
    payload = {
        "version": __version__,
        "git_revision": code_revision(),
        "analysis": "claim_ledger",
        "num_claims": len(entries),
        "supported_claims": supported,
        "unsupported_claims": len(entries) - supported,
        "decision": "claim_ledger_supported" if supported == len(entries) else "claim_ledger_has_gaps",
        "entries": [
            {
                "claim_id": entry.claim_id,
                "permitted_wording": entry.permitted_wording,
                "blocked_wording": list(entry.blocked_wording),
                "evidence_artifact": entry.evidence_artifact,
                "required_decision": entry.required_decision,
                "status": entry.status,
                "rationale": entry.rationale,
            }
            for entry in entries
        ],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2))
    write_markdown(payload, markdown)
    return output


def write_markdown(payload: dict[str, Any], markdown: Path) -> None:
    """Write a manuscript-facing claim audit report."""

    lines = [
        "# Claim Audit Report",
        "",
        f"- Decision: `{payload['decision']}`",
        f"- Supported claims: `{payload['supported_claims']}/{payload['num_claims']}`",
        "",
        "## Ledger",
        "",
        "| Claim ID | Status | Evidence | Permitted wording |",
        "|---|---|---|---|",
    ]
    for entry in payload["entries"]:
        lines.append(
            f"| `{entry['claim_id']}` | `{entry['status']}` | "
            f"`{entry['evidence_artifact']}` | {entry['permitted_wording']} |"
        )
    lines.extend(["", "## Blocked Wording", ""])
    for entry in payload["entries"]:
        lines.append(f"### `{entry['claim_id']}`")
        for wording in entry["blocked_wording"]:
            lines.append(f"- {wording}")
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
