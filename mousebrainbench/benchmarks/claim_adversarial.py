"""Adversarial claim benchmark for MouseBrainBench evaluators.

This benchmark constructs known-truth cases where prediction, reproducibility,
topology, direction, causal support, and structure-function evidence disagree.
The target is not to simulate biological truth. The target is to test whether an
evaluator authorizes claims that its evidence does not support.
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
    CLAIM_TYPES,
    AblatedClaimGateEvaluator,
    ClaimEvidence,
    ClaimGateEvaluator,
    CompensatoryScoreEvaluator,
    CorrelationOnlyEvaluator,
    LeaderboardOnlyEvaluator,
    ReliabilityOnlyEvaluator,
    TopologyOnlyEvaluator,
    aggregate_claim_confusion,
    claim_confusion_matrix,
)


DEFAULT_OUTPUT = Path("results/claim_adversarial_benchmark/summary.json")
DEFAULT_MARKDOWN = Path("results/claim_adversarial_benchmark/summary.md")


@dataclass(frozen=True)
class AdversarialCase:
    """One known-truth claim benchmark case."""

    name: str
    evidence: ClaimEvidence
    true_claims: tuple[str, ...]
    failure_mode: str


@dataclass(frozen=True)
class CaseTemplate:
    """Parameterized template used to create a broader adversarial suite."""

    name: str
    evidence: ClaimEvidence
    true_claims: tuple[str, ...]
    failure_mode: str
    variants: tuple[tuple[str, float, float], ...]


def _clip01(value: float) -> float:
    return max(0.0, min(1.0, value))


def _variant_evidence(evidence: ClaimEvidence, predictive_delta: float, repro_delta: float) -> ClaimEvidence:
    """Perturb prediction/reproducibility while preserving the structural truth block."""

    return ClaimEvidence(
        predictive_score=_clip01(evidence.predictive_score + predictive_delta),
        reproducibility_score=_clip01(evidence.reproducibility_score + repro_delta),
        topology_effect=evidence.topology_effect,
        topology_specific=evidence.topology_specific,
        directed_fraction=evidence.directed_fraction,
        structure_function_effect=evidence.structure_function_effect,
        matched_structure_function_effect=evidence.matched_structure_function_effect,
        structure_function_fdr_passed=evidence.structure_function_fdr_passed,
        causal_evidence=evidence.causal_evidence,
        whole_brain_coverage=evidence.whole_brain_coverage,
        independent_validation=evidence.independent_validation,
        reproducible_compute=evidence.reproducible_compute,
    )


def _templates() -> tuple[CaseTemplate, ...]:
    variants = (
        ("nominal", 0.00, 0.00),
        ("high_prediction", 0.06, 0.00),
        ("high_reproducibility", 0.00, 0.05),
        ("low_prediction_margin", -0.08, 0.00),
        ("low_reproducibility_margin", 0.00, -0.08),
        ("ood_shift", -0.12, -0.10),
    )
    return (
        CaseTemplate(
            name="directed_mechanistic_truth",
            evidence=ClaimEvidence(
                predictive_score=0.92,
                reproducibility_score=0.95,
                topology_effect=0.12,
                topology_specific=True,
                directed_fraction=0.85,
                independent_validation=True,
            ),
            true_claims=("predictive", "reproducible", "topology_specific", "directed", "mechanistic"),
            failure_mode="positive non-causal mechanistic control",
            variants=variants,
        ),
        CaseTemplate(
            name="common_drive_high_prediction",
            evidence=ClaimEvidence(predictive_score=0.91, reproducibility_score=0.94),
            true_claims=("predictive", "reproducible"),
            failure_mode="prediction and reproducibility caused by common drive",
            variants=variants,
        ),
        CaseTemplate(
            name="topology_without_direction",
            evidence=ClaimEvidence(
                predictive_score=0.82,
                reproducibility_score=0.91,
                topology_effect=0.11,
                topology_specific=True,
            ),
            true_claims=("predictive", "reproducible", "topology_specific"),
            failure_mode="regional specificity exists but no directed signature is present",
            variants=variants,
        ),
        CaseTemplate(
            name="direction_without_topology",
            evidence=ClaimEvidence(
                predictive_score=0.80,
                reproducibility_score=0.89,
                directed_fraction=0.88,
            ),
            true_claims=("predictive", "reproducible", "directed"),
            failure_mode="timing exists but does not identify the proposed topology",
            variants=variants,
        ),
        CaseTemplate(
            name="spatial_confound_structure_function",
            evidence=ClaimEvidence(
                predictive_score=0.72,
                reproducibility_score=0.86,
                structure_function_effect=0.05,
                matched_structure_function_effect=0.00,
                structure_function_fdr_passed=False,
            ),
            true_claims=("predictive", "reproducible"),
            failure_mode="structure-function association disappears under matched controls",
            variants=variants,
        ),
        CaseTemplate(
            name="local_structure_function_truth",
            evidence=ClaimEvidence(
                predictive_score=0.68,
                reproducibility_score=0.82,
                structure_function_effect=0.04,
                matched_structure_function_effect=0.02,
                structure_function_fdr_passed=True,
            ),
            true_claims=("predictive", "reproducible", "structure_function"),
            failure_mode="positive local observational structure-function control",
            variants=variants,
        ),
        CaseTemplate(
            name="causal_component_truth",
            evidence=ClaimEvidence(
                predictive_score=0.86,
                reproducibility_score=0.90,
                topology_effect=0.10,
                topology_specific=True,
                directed_fraction=0.80,
                causal_evidence=True,
                independent_validation=True,
            ),
            true_claims=(
                "predictive",
                "reproducible",
                "topology_specific",
                "directed",
                "mechanistic",
                "causal",
            ),
            failure_mode="causal positive control without whole-brain coverage",
            variants=variants,
        ),
        CaseTemplate(
            name="false_digital_twin_decoy",
            evidence=ClaimEvidence(
                predictive_score=0.93,
                reproducibility_score=0.94,
                topology_effect=0.10,
                topology_specific=True,
                directed_fraction=0.82,
                whole_brain_coverage=True,
                independent_validation=True,
                reproducible_compute=True,
            ),
            true_claims=("predictive", "reproducible", "topology_specific", "directed", "mechanistic"),
            failure_mode="whole-brain wording without causal evidence",
            variants=variants,
        ),
        CaseTemplate(
            name="whole_brain_digital_twin_truth",
            evidence=ClaimEvidence(
                predictive_score=0.91,
                reproducibility_score=0.93,
                topology_effect=0.10,
                topology_specific=True,
                directed_fraction=0.82,
                causal_evidence=True,
                whole_brain_coverage=True,
                independent_validation=True,
                reproducible_compute=True,
            ),
            true_claims=(
                "predictive",
                "reproducible",
                "topology_specific",
                "directed",
                "mechanistic",
                "causal",
                "digital_twin",
            ),
            failure_mode="upper-bound positive control for digital-twin wording",
            variants=variants,
        ),
    )


def build_cases() -> tuple[AdversarialCase, ...]:
    """Build the deterministic broad adversarial suite."""

    cases: list[AdversarialCase] = []
    for template in _templates():
        for variant_name, predictive_delta, repro_delta in template.variants:
            evidence = _variant_evidence(template.evidence, predictive_delta, repro_delta)
            cases.append(
                AdversarialCase(
                    name=f"{template.name}__{variant_name}",
                    evidence=evidence,
                    true_claims=template.true_claims,
                    failure_mode=f"{template.failure_mode}; variant={variant_name}",
                )
            )
    return tuple(cases)


CASES = build_cases()


def _evaluator_payload() -> tuple[dict[str, dict[str, set[str]]], list[dict[str, Any]]]:
    evaluators = (
        CorrelationOnlyEvaluator(),
        LeaderboardOnlyEvaluator(),
        ReliabilityOnlyEvaluator(),
        TopologyOnlyEvaluator(),
        CompensatoryScoreEvaluator(),
        AblatedClaimGateEvaluator("directed"),
        ClaimGateEvaluator(),
    )
    decisions: dict[str, dict[str, set[str]]] = {}
    case_rows: list[dict[str, Any]] = []
    for evaluator in evaluators:
        evaluator_decisions: dict[str, set[str]] = {}
        for case in CASES:
            decision = evaluator.evaluate(case.evidence)
            evaluator_decisions[case.name] = set(decision.allowed_claims)
            case_rows.append(
                {
                    "case": case.name,
                    "evaluator": evaluator.name,
                    "allowed_claims": list(decision.allowed_claims),
                    "true_claims": list(case.true_claims),
                    "false_positive_claims": sorted(set(decision.allowed_claims) - set(case.true_claims)),
                    "false_negative_claims": sorted(set(case.true_claims) - set(decision.allowed_claims)),
                    "rationale": decision.rationale,
                    "failure_mode": case.failure_mode,
                }
            )
        decisions[evaluator.name] = evaluator_decisions
    return decisions, case_rows


def run(output: Path = DEFAULT_OUTPUT, markdown: Path = DEFAULT_MARKDOWN) -> Path:
    """Run the adversarial claim benchmark."""

    truth = {case.name: set(case.true_claims) for case in CASES}
    decisions, case_rows = _evaluator_payload()
    confusion = claim_confusion_matrix(truth_by_case=truth, decisions_by_evaluator=decisions)
    aggregate = aggregate_claim_confusion(confusion)
    gate = next(row for row in aggregate if row["evaluator"] == "claim_gate")
    payload = {
        "version": __version__,
        "git_revision": code_revision(),
        "analysis": "claim_adversarial_benchmark",
        "claim_types": list(CLAIM_TYPES),
        "num_cases": len(CASES),
        "cases": [
            {
                "case": case.name,
                "true_claims": list(case.true_claims),
                "failure_mode": case.failure_mode,
            }
            for case in CASES
        ],
        "case_evaluator_rows": case_rows,
        "confusion_matrix": confusion,
        "aggregate_by_evaluator": aggregate,
        "decision": (
            "claim_gate_blocks_broad_adversarial_overclaims"
            if gate["fp"] == 0 and len(CASES) >= 40
            else "claim_gate_requires_revision"
        ),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2))
    write_markdown(payload, markdown)
    return output


def write_markdown(payload: dict[str, Any], markdown: Path) -> None:
    """Write a compact Markdown report."""

    lines = [
        "# Claim Adversarial Benchmark",
        "",
        f"- Decision: `{payload['decision']}`",
        f"- Cases: `{payload['num_cases']}`",
        f"- Claim types: `{len(payload['claim_types'])}`",
        "",
        "## Aggregate Confusion",
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

    lines.extend(
        [
            "",
            "## False-Positive Claims By Case",
            "",
            "| Case | Evaluator | False-positive claims |",
            "|---|---|---|",
        ]
    )
    for row in payload["case_evaluator_rows"]:
        if row["false_positive_claims"]:
            claims = ", ".join(f"`{claim}`" for claim in row["false_positive_claims"])
            lines.append(f"| `{row['case']}` | `{row['evaluator']}` | {claims} |")

    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "The benchmark is useful only if shortcut evaluators over-authorize claims in "
            "designed adversarial cases while the non-compensatory gate blocks them. "
            "ORI is the overclaiming risk index. CI is the conservativeness index.",
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
    print(json.dumps({"output": str(run(args.output, args.markdown).resolve())}))


if __name__ == "__main__":
    main()
