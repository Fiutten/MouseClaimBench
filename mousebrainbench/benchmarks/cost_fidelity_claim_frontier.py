"""Cost-fidelity-claim frontier for MouseBrainBench evidence cases."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from mousebrainbench import __version__
from mousebrainbench.artifacts import code_revision


DEFAULT_OUTPUT = Path("results/cost_fidelity_claim_frontier/summary.json")
DEFAULT_MARKDOWN = Path("results/cost_fidelity_claim_frontier/summary.md")


@dataclass(frozen=True)
class FrontierPoint:
    """One evidence/model point in the cost-fidelity-claim frontier."""

    name: str
    cost_units: float
    fidelity_score: float
    authorized_claims: tuple[str, ...]
    artifact: str
    interpretation: str


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text()) if path.exists() else {}


def build_frontier(root: Path = Path(".")) -> tuple[FrontierPoint, ...]:
    """Build a conservative frontier from existing artifacts."""

    claimbench = _load(root / "results/claim_adversarial_v2/summary.json")
    sensorium = _load(root / "results/sensorium_static_model_comparator/summary.json")
    dynamic = _load(root / "results/dynamic_sensorium_model_comparator/summary.json")
    microns = _load(root / "results/microns_primary_robustness/summary.json")
    external = _load(root / "results/external_causal_claim_validation/summary.json")

    return (
        FrontierPoint(
            "sensorium_static_predictive",
            cost_units=2.0,
            fidelity_score=0.70 if sensorium else 0.0,
            authorized_claims=("predictive", "reproducible"),
            artifact="results/sensorium_static_model_comparator/summary.json",
            interpretation="predictive visual-cortex case without causal promotion",
        ),
        FrontierPoint(
            "dynamic_sensorium_temporal",
            cost_units=3.0,
            fidelity_score=0.66 if dynamic else 0.0,
            authorized_claims=("predictive", "reproducible"),
            artifact="results/dynamic_sensorium_model_comparator/summary.json",
            interpretation="temporal predictive case without topology/direction promotion",
        ),
        FrontierPoint(
            "microns_local_structure_function",
            cost_units=5.0,
            fidelity_score=0.80 if microns.get("all_cohorts_robust") else 0.0,
            authorized_claims=("predictive", "reproducible", "structure_function"),
            artifact="results/microns_primary_robustness/summary.json",
            interpretation="local observational structure-function evidence",
        ),
        FrontierPoint(
            "external_causal_positive_controls",
            cost_units=1.0,
            fidelity_score=0.90 if external.get("decision") == "external_causal_validation_passed" else 0.0,
            authorized_claims=(
                "predictive",
                "reproducible",
                "topology_specific",
                "directed",
                "mechanistic",
                "causal",
            ),
            artifact="results/external_causal_claim_validation/summary.json",
            interpretation="known-truth external causal controls, not neurobiological evidence",
        ),
        FrontierPoint(
            "claimbench_v2_claim_audit",
            cost_units=1.5,
            fidelity_score=0.95
            if claimbench.get("decision") == "claimbench_v2_blocks_overclaiming_under_broad_attacks"
            else 0.0,
            authorized_claims=("benchmark_claim", "reviewer_defense_claim"),
            artifact="results/claim_adversarial_v2/summary.json",
            interpretation="methodological evidence about claim authorization",
        ),
    )


def _dominates(left: FrontierPoint, right: FrontierPoint) -> bool:
    return (
        left.cost_units <= right.cost_units
        and left.fidelity_score >= right.fidelity_score
        and len(left.authorized_claims) >= len(right.authorized_claims)
        and (
            left.cost_units < right.cost_units
            or left.fidelity_score > right.fidelity_score
            or len(left.authorized_claims) > len(right.authorized_claims)
        )
    )


def run(
    output: Path = DEFAULT_OUTPUT,
    markdown: Path = DEFAULT_MARKDOWN,
    root: Path = Path("."),
) -> Path:
    """Compute the cost-fidelity-claim frontier."""

    points = build_frontier(root)
    frontier = [
        point.name
        for point in points
        if not any(_dominates(other, point) for other in points if other.name != point.name)
    ]
    payload = {
        "version": __version__,
        "git_revision": code_revision(),
        "analysis": "cost_fidelity_claim_frontier",
        "frontier": frontier,
        "points": [
            {
                "name": point.name,
                "cost_units": point.cost_units,
                "fidelity_score": point.fidelity_score,
                "authorized_claims": list(point.authorized_claims),
                "artifact": point.artifact,
                "interpretation": point.interpretation,
                "on_frontier": point.name in frontier,
            }
            for point in points
        ],
        "decision": "cost_fidelity_claim_frontier_built" if frontier else "frontier_requires_inputs",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2))
    write_markdown(payload, markdown)
    return output


def write_markdown(payload: dict[str, Any], markdown: Path) -> None:
    """Write frontier report."""

    lines = [
        "# Cost-Fidelity-Claim Frontier",
        "",
        f"- Decision: `{payload['decision']}`",
        f"- Frontier points: `{', '.join(payload['frontier'])}`",
        "",
        "| Point | Cost | Fidelity | Claims | Frontier |",
        "|---|---:|---:|---:|---:|",
    ]
    for point in payload["points"]:
        lines.append(
            f"| `{point['name']}` | `{point['cost_units']:.2f}` | "
            f"`{point['fidelity_score']:.2f}` | `{len(point['authorized_claims'])}` | "
            f"`{point['on_frontier']}` |"
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
