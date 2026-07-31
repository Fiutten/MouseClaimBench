"""Map real MouseBrainBench evidence cases into the executable claim ladder."""

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
    ClaimEvidence,
    ClaimGateEvaluator,
    CompensatoryScoreEvaluator,
    CorrelationOnlyEvaluator,
    LeaderboardOnlyEvaluator,
    aggregate_claim_confusion,
    claim_confusion_matrix,
)


DEFAULT_OUTPUT = Path("results/real_case_claim_matrix/summary.json")
DEFAULT_MARKDOWN = Path("results/real_case_claim_matrix/summary.md")


@dataclass(frozen=True)
class RealClaimCase:
    """One real or calibration case mapped into normalized claim evidence."""

    name: str
    evidence: ClaimEvidence
    expected_claims: tuple[str, ...]
    source_artifact: str
    interpretation: str


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def build_cases(root: Path = Path(".")) -> tuple[RealClaimCase, ...]:
    """Build cautious real-case mappings from existing artifacts.

    These mappings are deliberately conservative. They do not promote Sensorium
    prediction or MICRONS local association into causality or whole-brain twin
    evidence.
    """

    allen = _load_json(root / "results/allen_vbn_mechanistic_identifiability_score.json")
    sensorium = _load_json(root / "results/sensorium_static_model_comparator/summary.json")
    dynamic = _load_json(root / "results/dynamic_sensorium_model_comparator/summary.json")
    microns = _load_json(root / "results/microns_primary_robustness/summary.json")

    microns_positive = microns.get("decision") == "microns_primary_endpoint_survives_harder_controls"
    sensorium_has_topography = bool(sensorium.get("topographic_constraint"))
    dynamic_available = dynamic.get("comparison") == "dynamic_sensorium_predictive_model_comparator"
    allen_negative = allen.get("decision") == "reproducible_target_without_mechanistic_identifiability"

    return (
        RealClaimCase(
            name="allen_vbn_negative_identifiability",
            evidence=ClaimEvidence(
                predictive_score=0.55,
                reproducibility_score=0.88 if allen_negative else 0.60,
            ),
            expected_claims=("predictive", "reproducible"),
            source_artifact="results/allen_vbn_mechanistic_identifiability_score.json",
            interpretation="real negative case: reproducible target without mechanistic identifiability",
        ),
        RealClaimCase(
            name="sensorium_static_predictive_interoperability",
            evidence=ClaimEvidence(
                predictive_score=0.70,
                reproducibility_score=0.86,
                topology_effect=0.06 if sensorium_has_topography else 0.00,
                topology_specific=sensorium_has_topography,
            ),
            expected_claims=(
                ("predictive", "reproducible", "topology_specific")
                if sensorium_has_topography
                else ("predictive", "reproducible")
            ),
            source_artifact="results/sensorium_static_model_comparator/summary.json",
            interpretation="predictive/interoperability case, not causal or digital-twin evidence",
        ),
        RealClaimCase(
            name="dynamic_sensorium_predictive_temporal_case",
            evidence=ClaimEvidence(
                predictive_score=0.66 if dynamic_available else 0.20,
                reproducibility_score=0.75 if dynamic_available else 0.20,
            ),
            expected_claims=("predictive", "reproducible") if dynamic_available else tuple(),
            source_artifact="results/dynamic_sensorium_model_comparator/summary.json",
            interpretation="temporal predictive case without topology, direction, or causal support",
        ),
        RealClaimCase(
            name="microns_local_structure_function",
            evidence=ClaimEvidence(
                predictive_score=0.30,
                reproducibility_score=0.80,
                structure_function_effect=0.02 if microns_positive else 0.00,
                matched_structure_function_effect=0.014 if microns_positive else 0.00,
                structure_function_fdr_passed=microns_positive,
            ),
            expected_claims=(
                ("predictive", "reproducible", "structure_function")
                if microns_positive
                else ("predictive", "reproducible")
            ),
            source_artifact="results/microns_primary_robustness/summary.json",
            interpretation="local observational structure-function case, not causal evidence",
        ),
        RealClaimCase(
            name="synthetic_causal_graph_positive_control",
            evidence=ClaimEvidence(
                predictive_score=0.88,
                reproducibility_score=0.92,
                topology_effect=0.11,
                topology_specific=True,
                directed_fraction=0.82,
                causal_evidence=True,
            ),
            expected_claims=(
                "predictive",
                "reproducible",
                "topology_specific",
                "directed",
                "mechanistic",
                "causal",
            ),
            source_artifact="synthetic_calibration_case",
            interpretation="non-neuro positive control showing the gate can authorize causal claims",
        ),
    )


def run(output: Path = DEFAULT_OUTPUT, markdown: Path = DEFAULT_MARKDOWN, root: Path = Path(".")) -> Path:
    """Evaluate real-case claim mappings against shortcut baselines."""

    cases = build_cases(root)
    evaluators = (
        CorrelationOnlyEvaluator(),
        LeaderboardOnlyEvaluator(),
        CompensatoryScoreEvaluator(),
        ClaimGateEvaluator(),
    )
    truth = {case.name: set(case.expected_claims) for case in cases}
    decisions: dict[str, dict[str, set[str]]] = {}
    rows: list[dict[str, Any]] = []
    for evaluator in evaluators:
        evaluator_decisions: dict[str, set[str]] = {}
        for case in cases:
            decision = evaluator.evaluate(case.evidence)
            evaluator_decisions[case.name] = set(decision.allowed_claims)
            rows.append(
                {
                    "case": case.name,
                    "evaluator": evaluator.name,
                    "allowed_claims": list(decision.allowed_claims),
                    "expected_claims": list(case.expected_claims),
                    "false_positive_claims": sorted(
                        set(decision.allowed_claims) - set(case.expected_claims)
                    ),
                    "false_negative_claims": sorted(
                        set(case.expected_claims) - set(decision.allowed_claims)
                    ),
                    "source_artifact": case.source_artifact,
                    "interpretation": case.interpretation,
                    "rationale": decision.rationale,
                }
            )
        decisions[evaluator.name] = evaluator_decisions

    confusion = claim_confusion_matrix(truth_by_case=truth, decisions_by_evaluator=decisions)
    aggregate = aggregate_claim_confusion(confusion)
    gate = next(row for row in aggregate if row["evaluator"] == "claim_gate")
    payload = {
        "version": __version__,
        "git_revision": code_revision(),
        "analysis": "real_case_claim_matrix",
        "claim_types": list(CLAIM_TYPES),
        "cases": [
            {
                "case": case.name,
                "expected_claims": list(case.expected_claims),
                "source_artifact": case.source_artifact,
                "interpretation": case.interpretation,
            }
            for case in cases
        ],
        "case_evaluator_rows": rows,
        "confusion_matrix": confusion,
        "aggregate_by_evaluator": aggregate,
        "decision": "real_case_claim_gate_consistent" if gate["fp"] == 0 else "real_case_gate_requires_revision",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2))
    write_markdown(payload, markdown)
    return output


def write_markdown(payload: dict[str, Any], markdown: Path) -> None:
    """Write the real-case matrix report."""

    lines = [
        "# Real-Case Claim Matrix",
        "",
        f"- Decision: `{payload['decision']}`",
        f"- Cases: `{len(payload['cases'])}`",
        "",
        "## Cases",
        "",
        "| Case | Expected claims | Source |",
        "|---|---|---|",
    ]
    for case in payload["cases"]:
        claims = ", ".join(f"`{claim}`" for claim in case["expected_claims"])
        lines.append(f"| `{case['case']}` | {claims} | `{case['source_artifact']}` |")

    lines.extend(
        [
            "",
            "## Aggregate Confusion",
            "",
            "| Evaluator | TP | FP | TN | FN | ORI | CI |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
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
    parser.add_argument("--root", type=Path, default=Path("."))
    args = parser.parse_args()
    print(json.dumps({"output": str(run(args.output, args.markdown, args.root).resolve())}))


if __name__ == "__main__":
    main()
