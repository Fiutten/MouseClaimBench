"""ClaimBench v2 contract-conformance stress test.

This module is a post-submission hardening benchmark. It deliberately expands
the synthetic known-truth cases beyond the submitted artifact to stress-test
claim authorization under broader reviewer-style attacks. Labels are generated
from the same operational contract used by the evaluator, so this module tests
software conformance and attack coverage. It is not independent scientific
validation. Use ``oracle_sem_claim_benchmark`` for independently generated
structural-equation reference labels.
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
    ClaimGateThresholds,
    CompensatoryScoreEvaluator,
    CorrelationOnlyEvaluator,
    LeaderboardOnlyEvaluator,
    ReliabilityOnlyEvaluator,
    TopologyOnlyEvaluator,
    aggregate_claim_confusion,
    claim_confusion_matrix,
)


DEFAULT_OUTPUT = Path("results/claim_adversarial_v2/summary.json")
DEFAULT_MARKDOWN = Path("results/claim_adversarial_v2/summary.md")


@dataclass(frozen=True)
class AdversarialCaseV2:
    """Constructed contract-conformance case for claim authorization."""

    name: str
    family: str
    evidence: ClaimEvidence
    true_claims: tuple[str, ...]
    reviewer_attack: str


@dataclass(frozen=True)
class _Template:
    family: str
    evidence: ClaimEvidence
    true_claims: tuple[str, ...]
    reviewer_attack: str


def _clip01(value: float) -> float:
    return max(0.0, min(1.0, value))


def _perturb(evidence: ClaimEvidence, deltas: tuple[float, float, float, float, float]) -> ClaimEvidence:
    pred, repro, topo, directed, sf = deltas
    return ClaimEvidence(
        predictive_score=_clip01(evidence.predictive_score + pred),
        reproducibility_score=_clip01(evidence.reproducibility_score + repro),
        topology_effect=max(0.0, evidence.topology_effect + topo),
        topology_specific=evidence.topology_specific,
        directed_fraction=_clip01(evidence.directed_fraction + directed),
        structure_function_effect=max(0.0, evidence.structure_function_effect + sf),
        matched_structure_function_effect=max(0.0, evidence.matched_structure_function_effect + sf),
        structure_function_fdr_passed=evidence.structure_function_fdr_passed,
        causal_evidence=evidence.causal_evidence,
        whole_brain_coverage=evidence.whole_brain_coverage,
        independent_validation=evidence.independent_validation,
        reproducible_compute=evidence.reproducible_compute,
    )


def _supportable_truth(base_truth: tuple[str, ...], evidence: ClaimEvidence) -> tuple[str, ...]:
    """Return claims supportable after a deterministic evidence perturbation.

    Each template defines which claim types are biologically possible in that
    scenario. Variants then change the strength of observable evidence. This
    function prevents a variant from being counted as a false positive simply
    because the perturbation legitimately moved it above a declared threshold.
    """

    thresholds = ClaimGateThresholds()
    base = set(base_truth)
    claims: list[str] = []
    predictive = evidence.predictive_score >= thresholds.predictive_score
    reproducible = evidence.reproducibility_score >= thresholds.reproducibility_score
    topology = (
        "topology_specific" in base
        and evidence.topology_specific
        and evidence.topology_effect >= thresholds.topology_effect
    )
    directed = "directed" in base and evidence.directed_fraction >= thresholds.directed_fraction
    structure_function = (
        "structure_function" in base
        and evidence.structure_function_effect > 0.0
        and evidence.matched_structure_function_effect
        >= thresholds.matched_structure_function_effect
        and evidence.structure_function_fdr_passed
    )
    causal = "causal" in base and evidence.causal_evidence
    if predictive:
        claims.append("predictive")
    if reproducible:
        claims.append("reproducible")
    if topology:
        claims.append("topology_specific")
    if directed:
        claims.append("directed")
    if structure_function:
        claims.append("structure_function")
    if "mechanistic" in base and predictive and reproducible and topology and directed:
        claims.append("mechanistic")
    if causal:
        claims.append("causal")
    if (
        "digital_twin" in base
        and evidence.whole_brain_coverage
        and evidence.independent_validation
        and evidence.reproducible_compute
        and causal
    ):
        claims.append("digital_twin")
    return tuple(claims)


def _templates() -> tuple[_Template, ...]:
    return (
        _Template(
            "prediction_only_common_drive",
            ClaimEvidence(0.92, 0.94),
            ("predictive", "reproducible"),
            "High prediction is treated as mechanism although only common drive is present.",
        ),
        _Template(
            "leaderboard_overclaim",
            ClaimEvidence(0.88, 0.62),
            ("predictive",),
            "Leaderboard performance is treated as reproducible mechanism.",
        ),
        _Template(
            "reliability_without_structure",
            ClaimEvidence(0.24, 0.93),
            ("reproducible",),
            "Stability is treated as topology or causality.",
        ),
        _Template(
            "topology_without_direction",
            ClaimEvidence(0.80, 0.88, topology_effect=0.11, topology_specific=True),
            ("predictive", "reproducible", "topology_specific"),
            "Topological specificity is treated as directed mechanism.",
        ),
        _Template(
            "direction_without_topology",
            ClaimEvidence(0.78, 0.90, directed_fraction=0.84),
            ("predictive", "reproducible", "directed"),
            "Temporal direction is treated as topology-specific mechanism.",
        ),
        _Template(
            "spatial_structure_function_confound",
            ClaimEvidence(
                0.72,
                0.84,
                structure_function_effect=0.05,
                matched_structure_function_effect=0.00,
                structure_function_fdr_passed=False,
            ),
            ("predictive", "reproducible"),
            "Unmatched structure-function association is treated as a validated local effect.",
        ),
        _Template(
            "matched_structure_function_positive",
            ClaimEvidence(
                0.62,
                0.80,
                structure_function_effect=0.04,
                matched_structure_function_effect=0.018,
                structure_function_fdr_passed=True,
            ),
            ("predictive", "reproducible", "structure_function"),
            "Positive local structure-function control.",
        ),
        _Template(
            "mechanistic_positive_noncausal",
            ClaimEvidence(
                0.86,
                0.89,
                topology_effect=0.10,
                topology_specific=True,
                directed_fraction=0.78,
            ),
            ("predictive", "reproducible", "topology_specific", "directed", "mechanistic"),
            "Mechanistic non-causal positive control.",
        ),
        _Template(
            "causal_positive_local",
            ClaimEvidence(
                0.86,
                0.90,
                topology_effect=0.11,
                topology_specific=True,
                directed_fraction=0.80,
                causal_evidence=True,
                independent_validation=True,
            ),
            (
                "predictive",
                "reproducible",
                "topology_specific",
                "directed",
                "mechanistic",
                "causal",
            ),
            "Causal local positive control without whole-brain twin status.",
        ),
        _Template(
            "digital_twin_decoy_without_causality",
            ClaimEvidence(
                0.91,
                0.92,
                topology_effect=0.10,
                topology_specific=True,
                directed_fraction=0.82,
                whole_brain_coverage=True,
                independent_validation=True,
            ),
            ("predictive", "reproducible", "topology_specific", "directed", "mechanistic"),
            "Whole-brain wording is attempted without causal evidence.",
        ),
        _Template(
            "digital_twin_decoy_without_independent_validation",
            ClaimEvidence(
                0.91,
                0.92,
                topology_effect=0.10,
                topology_specific=True,
                directed_fraction=0.82,
                causal_evidence=True,
                whole_brain_coverage=True,
                independent_validation=False,
            ),
            ("predictive", "reproducible", "topology_specific", "directed", "mechanistic", "causal"),
            "Causal and whole-brain flags are present but independent validation is absent.",
        ),
        _Template(
            "digital_twin_positive_control",
            ClaimEvidence(
                0.92,
                0.94,
                topology_effect=0.11,
                topology_specific=True,
                directed_fraction=0.84,
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
            "Upper-bound positive control for complete digital-twin wording.",
        ),
    )


def build_cases() -> tuple[AdversarialCaseV2, ...]:
    """Build a deterministic 144-case adversarial suite."""

    variants = (
        ("nominal", (0.00, 0.00, 0.00, 0.00, 0.000)),
        ("high_prediction", (0.06, 0.00, 0.00, 0.00, 0.000)),
        ("low_prediction", (-0.10, 0.00, 0.00, 0.00, 0.000)),
        ("high_reproducibility", (0.00, 0.06, 0.00, 0.00, 0.000)),
        ("low_reproducibility", (0.00, -0.10, 0.00, 0.00, 0.000)),
        ("topology_margin_up", (0.00, 0.00, 0.025, 0.00, 0.000)),
        ("topology_margin_down", (0.00, 0.00, -0.035, 0.00, 0.000)),
        ("direction_margin_up", (0.00, 0.00, 0.00, 0.12, 0.000)),
        ("direction_margin_down", (0.00, 0.00, 0.00, -0.18, 0.000)),
        ("sf_margin_up", (0.00, 0.00, 0.00, 0.00, 0.006)),
        ("sf_margin_down", (0.00, 0.00, 0.00, 0.00, -0.008)),
        ("mild_ood", (-0.08, -0.08, -0.015, -0.08, -0.004)),
    )
    cases: list[AdversarialCaseV2] = []
    for template in _templates():
        for variant_name, deltas in variants:
            evidence = _perturb(template.evidence, deltas)
            cases.append(
                AdversarialCaseV2(
                    name=f"{template.family}__{variant_name}",
                    family=template.family,
                    evidence=evidence,
                    true_claims=_supportable_truth(template.true_claims, evidence),
                    reviewer_attack=f"{template.reviewer_attack} Variant={variant_name}.",
                )
            )
    return tuple(cases)


def _evaluate_cases(cases: tuple[AdversarialCaseV2, ...]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    evaluators = (
        CorrelationOnlyEvaluator(),
        LeaderboardOnlyEvaluator(),
        ReliabilityOnlyEvaluator(),
        TopologyOnlyEvaluator(),
        CompensatoryScoreEvaluator(),
        AblatedClaimGateEvaluator("topology"),
        AblatedClaimGateEvaluator("directed"),
        AblatedClaimGateEvaluator("reproducible"),
        ClaimGateEvaluator(),
    )
    truth = {case.name: set(case.true_claims) for case in cases}
    decisions_by_evaluator: dict[str, dict[str, set[str]]] = {}
    case_rows: list[dict[str, Any]] = []
    for evaluator in evaluators:
        decisions: dict[str, set[str]] = {}
        for case in cases:
            decision = evaluator.evaluate(case.evidence)
            predicted = set(decision.allowed_claims)
            expected = set(case.true_claims)
            decisions[case.name] = predicted
            case_rows.append(
                {
                    "case": case.name,
                    "family": case.family,
                    "evaluator": evaluator.name,
                    "true_claims": list(case.true_claims),
                    "allowed_claims": list(decision.allowed_claims),
                    "false_positive_claims": sorted(predicted - expected),
                    "false_negative_claims": sorted(expected - predicted),
                    "reviewer_attack": case.reviewer_attack,
                    "rationale": decision.rationale,
                }
            )
        decisions_by_evaluator[evaluator.name] = decisions
    confusion = claim_confusion_matrix(
        truth_by_case=truth,
        decisions_by_evaluator=decisions_by_evaluator,
    )
    return aggregate_claim_confusion(confusion), case_rows


def run(output: Path = DEFAULT_OUTPUT, markdown: Path = DEFAULT_MARKDOWN) -> Path:
    """Run ClaimBench v2 adversarial stress test."""

    cases = build_cases()
    aggregate, case_rows = _evaluate_cases(cases)
    gate = next(row for row in aggregate if row["evaluator"] == "claim_gate")
    shortcut_rows = [row for row in aggregate if row["evaluator"] != "claim_gate"]
    payload = {
        "version": __version__,
        "git_revision": code_revision(),
        "analysis": "claim_adversarial_v2",
        "validation_role": "software_contract_conformance_not_independent_validation",
        "label_source": "the same operational claim contract used by the evaluated gate",
        "claim_types": list(CLAIM_TYPES),
        "num_cases": len(cases),
        "families": sorted({case.family for case in cases}),
        "aggregate_by_evaluator": aggregate,
        "case_evaluator_rows": case_rows,
        "decision": (
            "claimbench_v2_blocks_overclaiming_under_broad_attacks"
            if gate["fp"] == 0
            and max(float(row["overclaiming_risk_index"]) for row in shortcut_rows) > 0.20
            else "claimbench_v2_requires_revision"
        ),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2))
    write_markdown(payload, markdown)
    return output


def write_markdown(payload: dict[str, Any], markdown: Path) -> None:
    """Write a compact ClaimBench v2 report."""

    lines = [
        "# ClaimBench v2 Contract-Conformance Stress Test",
        "",
        f"- Decision: `{payload['decision']}`",
        f"- Cases: `{payload['num_cases']}`",
        f"- Families: `{len(payload['families'])}`",
        "",
        "| Evaluator | TP | FP | TN | FN | FPR | FNR |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in payload["aggregate_by_evaluator"]:
        lines.append(
            f"| `{row['evaluator']}` | `{row['tp']}` | `{row['fp']}` | `{row['tn']}` | "
            f"`{row['fn']}` | `{row['false_positive_rate']:.3f}` | "
            f"`{row['false_negative_rate']:.3f}` |"
        )
    lines.extend(["", "## Families", ""])
    lines.extend(f"- `{family}`" for family in payload["families"])
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
