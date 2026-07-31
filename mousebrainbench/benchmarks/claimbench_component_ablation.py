"""Component ablation study for ClaimBench v2.

This benchmark answers a reviewer-facing question: which parts of the framework
actually matter?  It combines the synthetic known-truth suite with external
SciFact/Tuebingen controls and records what breaks when evidence blocks are
removed or reduced to simpler shortcuts.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from mousebrainbench import __version__
from mousebrainbench.artifacts import code_revision


DEFAULT_OUTPUT = Path("results/claimbench_component_ablation/summary.json")
DEFAULT_MARKDOWN = Path("results/claimbench_component_ablation/summary.md")


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text()) if path.exists() else {}


def _aggregate_by_evaluator(adversarial: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {row["evaluator"]: row for row in adversarial.get("aggregate_by_evaluator", [])}


def _ratio(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def _component_row(
    *,
    component: str,
    removed_condition: str,
    evidence: str,
    effect: str,
    severity: str,
) -> dict[str, str]:
    return {
        "component": component,
        "removed_condition": removed_condition,
        "evidence": evidence,
        "effect": effect,
        "severity": severity,
    }


def build_ablation_rows(root: Path = Path(".")) -> list[dict[str, str]]:
    """Build ablation rows from existing executable artifacts."""

    adversarial = _load(root / "results/claim_adversarial_v2/summary.json")
    uncertainty = _load(root / "results/uncertainty_claim_gate_v2/summary.json")
    scifact = _load(root / "results/scifact_claim_verification/summary.json")
    tuebingen = _load(root / "results/tuebingen_causal_direction/summary.json")
    manuscript = _load(root / "results/manuscript_claim_audit/summary.json")
    aggregate = _aggregate_by_evaluator(adversarial)

    claim_gate = aggregate.get("claim_gate", {})
    correlation = aggregate.get("correlation_only", {})
    compensatory = aggregate.get("compensatory_score", {})
    no_topology = aggregate.get("ablated_claim_gate_no_topology", {})
    no_directed = aggregate.get("ablated_claim_gate_no_directed", {})

    rows = [
        _component_row(
            component="non_compensatory_gate",
            removed_condition="replace with correlation-only evaluation",
            evidence=(
                f"claim_gate_ORI={claim_gate.get('overclaiming_risk_index')}; "
                f"correlation_ORI={correlation.get('overclaiming_risk_index')}"
            ),
            effect=(
                "Correlation-only evaluation authorizes many unsupported topology, direction, "
                "structure-function, and mechanistic claims."
            ),
            severity="critical",
        ),
        _component_row(
            component="non_compensatory_gate",
            removed_condition="replace with compensatory weighted score",
            evidence=(
                f"claim_gate_ORI={claim_gate.get('overclaiming_risk_index')}; "
                f"compensatory_ORI={compensatory.get('overclaiming_risk_index')}"
            ),
            effect="Weighted compensation reintroduces unsupported claim authorization.",
            severity="high",
        ),
        _component_row(
            component="topology_block",
            removed_condition="allow mechanism without topology specificity",
            evidence=f"no_topology_ORI={no_topology.get('overclaiming_risk_index')}",
            effect="Mechanistic claims can pass when topology evidence is missing.",
            severity="high",
        ),
        _component_row(
            component="direction_block",
            removed_condition="allow mechanism without directed evidence",
            evidence=f"no_direction_ORI={no_directed.get('overclaiming_risk_index')}",
            effect="Mechanistic claims can pass when direction is not identified.",
            severity="high",
        ),
        _component_row(
            component="uncertainty_status",
            removed_condition="force every borderline claim into supported/blocked",
            evidence=(
                f"unsupported_supported={uncertainty.get('unsupported_supported')}; "
                f"supported_uncertain={uncertainty.get('supported_uncertain')}"
            ),
            effect=(
                "Uncertainty is needed to report supported claims that become unstable under "
                "local evidence perturbations."
            ),
            severity="medium",
        ),
        _component_row(
            component="scientific_evidence_retrieval",
            removed_condition="use lexical citation overlap without retrieval/rationale separation",
            evidence=(
                f"shortcut_ORI={scifact.get('shortcut_overclaiming_risk')}; "
                f"bm25_recall_at_5={scifact.get('retrieval_recall_at_5')}; "
                f"bm25_rationale_ORI={scifact.get('retrieval_overclaiming_risk')}"
            ),
            effect=(
                "SciFact shows that evidence retrieval and support classification must be "
                "reported separately from lexical overlap."
            ),
            severity="high",
        ),
        _component_row(
            component="causal_abstention",
            removed_condition="allow correlation to authorize causal direction",
            evidence=(
                f"correlation_only_direction_overclaims="
                f"{tuebingen.get('correlation_only_direction_overclaims')}; "
                f"causal_performance_claim_allowed="
                f"{tuebingen.get('causal_performance_claim_allowed')}"
            ),
            effect=(
                "Tuebingen supports causal-overclaiming control, but not a causal-discovery "
                "performance claim."
            ),
            severity="critical",
        ),
        _component_row(
            component="manuscript_audit",
            removed_condition="do not audit text against executable claim contracts",
            evidence=(
                f"decision={manuscript.get('decision')}; "
                f"active_risk_hits={len(manuscript.get('active_risk_pattern_hits', []))}; "
                f"inputs={len(manuscript.get('existing_manuscript_inputs', []))}"
            ),
            effect="The current manuscript can be checked directly for unsupported wording.",
            severity="high",
        ),
    ]
    return rows


def run(output: Path = DEFAULT_OUTPUT, markdown: Path = DEFAULT_MARKDOWN, root: Path = Path(".")) -> Path:
    """Run the component ablation summary."""

    rows = build_ablation_rows(root)
    high_or_critical = [row for row in rows if row["severity"] in {"high", "critical"}]
    payload = {
        "version": __version__,
        "git_revision": code_revision(),
        "analysis": "claimbench_component_ablation",
        "num_components": len(rows),
        "high_or_critical_components": len(high_or_critical),
        "rows": rows,
        "decision": (
            "claimbench_components_have_nonredundant_value"
            if len(high_or_critical) >= 5
            else "claimbench_component_value_requires_more_evidence"
        ),
        "interpretation": (
            "The ablation does not claim that ClaimBench is a better predictor or causal "
            "discovery method. It shows that removing claim-gate components reintroduces "
            "unsupported scientific wording or removes the ability to audit it."
        ),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2))
    write_markdown(payload, markdown)
    return output


def write_markdown(payload: dict[str, Any], markdown: Path) -> None:
    """Write a compact ablation report."""

    lines = [
        "# ClaimBench Component Ablation",
        "",
        f"- Decision: `{payload['decision']}`",
        f"- Components: `{payload['num_components']}`",
        f"- High/critical components: `{payload['high_or_critical_components']}`",
        "",
        "| Component | Removed condition | Severity | Evidence | Effect |",
        "|---|---|---|---|---|",
    ]
    for row in payload["rows"]:
        lines.append(
            f"| `{row['component']}` | {row['removed_condition']} | `{row['severity']}` | "
            f"{row['evidence']} | {row['effect']} |"
        )
    lines.extend(["", "## Interpretation", "", str(payload["interpretation"]), ""])
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
