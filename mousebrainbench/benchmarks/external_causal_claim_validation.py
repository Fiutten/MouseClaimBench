"""External synthetic causal validation for claim-aware evaluation.

The benchmark uses small known directed acyclic graphs. It is intentionally not
neuroscience-specific. Its role is to test whether the claim gate generalizes to
structured scientific models where topology, direction, and causal evidence can
be known by construction.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from mousebrainbench import __version__
from mousebrainbench.artifacts import code_revision
from mousebrainbench.validation.claim_evaluation import (
    ClaimEvidence,
    ClaimGateEvaluator,
    CompensatoryScoreEvaluator,
    CorrelationOnlyEvaluator,
    LeaderboardOnlyEvaluator,
    aggregate_claim_confusion,
    claim_confusion_matrix,
)


DEFAULT_OUTPUT = Path("results/external_causal_claim_validation/summary.json")
DEFAULT_MARKDOWN = Path("results/external_causal_claim_validation/summary.md")


@dataclass(frozen=True)
class CausalScenario:
    """Known-truth causal graph scenario."""

    name: str
    graph: tuple[tuple[str, str], ...]
    evidence: ClaimEvidence
    true_claims: tuple[str, ...]
    interpretation: str


def build_scenarios() -> tuple[CausalScenario, ...]:
    """Build deterministic external causal scenarios."""

    return (
        CausalScenario(
            "chain_identified",
            (("x1", "x2"), ("x2", "x3")),
            ClaimEvidence(
                0.86,
                0.90,
                topology_effect=0.10,
                topology_specific=True,
                directed_fraction=0.90,
                causal_evidence=True,
            ),
            ("predictive", "reproducible", "topology_specific", "directed", "mechanistic", "causal"),
            "Correctly identified directed causal chain.",
        ),
        CausalScenario(
            "fork_common_cause_prediction_only",
            (("z", "x1"), ("z", "x2")),
            ClaimEvidence(0.88, 0.89),
            ("predictive", "reproducible"),
            "Shared cause creates prediction without directed mechanism between observed targets.",
        ),
        CausalScenario(
            "collider_direction_ambiguous",
            (("x1", "z"), ("x2", "z")),
            ClaimEvidence(0.74, 0.86, topology_effect=0.08, topology_specific=True),
            ("predictive", "reproducible", "topology_specific"),
            "Topology is present but direction is not recoverable under the observed evidence.",
        ),
        CausalScenario(
            "intervention_without_whole_system_twin",
            (("u", "v"),),
            ClaimEvidence(
                0.82,
                0.88,
                topology_effect=0.09,
                topology_specific=True,
                directed_fraction=0.82,
                causal_evidence=True,
                independent_validation=True,
            ),
            ("predictive", "reproducible", "topology_specific", "directed", "mechanistic", "causal"),
            "Intervention supports causal wording but not complete digital-twin wording.",
        ),
        CausalScenario(
            "complete_twin_positive_control",
            (("x1", "x2"), ("x2", "x3"), ("x1", "x3")),
            ClaimEvidence(
                0.92,
                0.94,
                topology_effect=0.12,
                topology_specific=True,
                directed_fraction=0.90,
                causal_evidence=True,
                whole_brain_coverage=True,
                independent_validation=True,
                reproducible_compute=True,
            ),
            (
                "predictive",
                "reproducible",
                "topology_specific",
                "directed",
                "mechanistic",
                "causal",
                "digital_twin",
            ),
            "Upper-bound complete-system positive control.",
        ),
    )


def run(output: Path = DEFAULT_OUTPUT, markdown: Path = DEFAULT_MARKDOWN) -> Path:
    """Evaluate external causal scenarios."""

    scenarios = build_scenarios()
    evaluators = (
        CorrelationOnlyEvaluator(),
        LeaderboardOnlyEvaluator(),
        CompensatoryScoreEvaluator(),
        ClaimGateEvaluator(),
    )
    truth = {scenario.name: set(scenario.true_claims) for scenario in scenarios}
    decisions_by_evaluator: dict[str, dict[str, set[str]]] = {}
    rows: list[dict[str, Any]] = []
    for evaluator in evaluators:
        evaluator_decisions: dict[str, set[str]] = {}
        for scenario in scenarios:
            decision = evaluator.evaluate(scenario.evidence)
            predicted = set(decision.allowed_claims)
            expected = set(scenario.true_claims)
            evaluator_decisions[scenario.name] = predicted
            rows.append(
                {
                    "scenario": scenario.name,
                    "evaluator": evaluator.name,
                    "allowed_claims": list(decision.allowed_claims),
                    "true_claims": list(scenario.true_claims),
                    "false_positive_claims": sorted(predicted - expected),
                    "false_negative_claims": sorted(expected - predicted),
                    "interpretation": scenario.interpretation,
                }
            )
        decisions_by_evaluator[evaluator.name] = evaluator_decisions
    aggregate = aggregate_claim_confusion(
        claim_confusion_matrix(truth_by_case=truth, decisions_by_evaluator=decisions_by_evaluator)
    )
    gate = next(row for row in aggregate if row["evaluator"] == "claim_gate")
    payload = {
        "version": __version__,
        "git_revision": code_revision(),
        "analysis": "external_causal_claim_validation",
        "num_scenarios": len(scenarios),
        "scenarios": [
            {
                "scenario": scenario.name,
                "graph": [list(edge) for edge in scenario.graph],
                "true_claims": list(scenario.true_claims),
                "interpretation": scenario.interpretation,
            }
            for scenario in scenarios
        ],
        "case_evaluator_rows": rows,
        "aggregate_by_evaluator": aggregate,
        "decision": (
            "external_causal_validation_passed"
            if gate["fp"] == 0 and gate["fn"] == 0
            else "external_causal_validation_requires_revision"
        ),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2))
    write_markdown(payload, markdown)
    return output


def write_markdown(payload: dict[str, Any], markdown: Path) -> None:
    """Write external causal validation report."""

    lines = [
        "# External Causal Claim Validation",
        "",
        f"- Decision: `{payload['decision']}`",
        f"- Scenarios: `{payload['num_scenarios']}`",
        "",
        "| Evaluator | TP | FP | TN | FN | ORI | CI |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in payload["aggregate_by_evaluator"]:
        lines.append(
            f"| `{row['evaluator']}` | `{row['tp']}` | `{row['fp']}` | `{row['tn']}` | "
            f"`{row['fn']}` | `{row['overclaiming_risk_index']:.3f}` | "
            f"`{row['conservativeness_index']:.3f}` |"
        )
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
