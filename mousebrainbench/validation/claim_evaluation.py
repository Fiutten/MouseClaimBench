"""Claim-level evaluators for comparing standard and non-compensatory validation.

The goal of this module is not to define biological truth. It defines a compact
decision layer for controlled benchmarks where the ground-truth claim labels are
known. This lets MouseBrainBench test whether common evaluation shortcuts would
authorize claims that the evidence does not support.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


CLAIM_TYPES = (
    "predictive",
    "reproducible",
    "topology_specific",
    "directed",
    "structure_function",
    "mechanistic",
    "causal",
    "digital_twin",
)


CLAIM_LADDER = (
    {
        "level": 1,
        "claim": "predictive",
        "meaning": "held-out response prediction is above a declared threshold",
        "required_evidence": ("predictive_score",),
    },
    {
        "level": 2,
        "claim": "reproducible",
        "meaning": "the result is stable enough under the declared reproducibility protocol",
        "required_evidence": ("reproducibility_score",),
    },
    {
        "level": 3,
        "claim": "topology_specific",
        "meaning": "topological controls are worse than the tested biological topology",
        "required_evidence": ("topology_specific", "topology_effect"),
    },
    {
        "level": 4,
        "claim": "directed",
        "meaning": "directional evidence is present rather than inferred from undirected association",
        "required_evidence": ("directed_fraction",),
    },
    {
        "level": 5,
        "claim": "structure_function",
        "meaning": "a local structure-function association survives matched controls",
        "required_evidence": (
            "structure_function_effect",
            "matched_structure_function_effect",
            "structure_function_fdr_passed",
        ),
    },
    {
        "level": 6,
        "claim": "mechanistic",
        "meaning": "prediction, reproducibility, topology, and direction all pass",
        "required_evidence": ("predictive", "reproducible", "topology_specific", "directed"),
    },
    {
        "level": 7,
        "claim": "causal",
        "meaning": "an explicit perturbational or intervention-based causal test is present",
        "required_evidence": ("causal_evidence",),
    },
    {
        "level": 8,
        "claim": "digital_twin",
        "meaning": "whole-brain, independently validated, causal and reproducible twin evidence exists",
        "required_evidence": (
            "whole_brain_coverage",
            "independent_validation",
            "reproducible_compute",
            "causal_evidence",
        ),
    },
)


@dataclass(frozen=True)
class ClaimEvidence:
    """Normalized evidence fields used by the comparative claim evaluators."""

    predictive_score: float
    reproducibility_score: float
    topology_effect: float = 0.0
    topology_specific: bool = False
    directed_fraction: float = 0.0
    structure_function_effect: float = 0.0
    matched_structure_function_effect: float = 0.0
    structure_function_fdr_passed: bool = False
    causal_evidence: bool = False
    whole_brain_coverage: bool = False
    independent_validation: bool = False
    reproducible_compute: bool = True


@dataclass(frozen=True)
class ClaimGateThresholds:
    """Thresholds used by the non-compensatory claim gate.

    Keeping thresholds in a dataclass makes sensitivity analysis explicit. The
    default values reproduce the nominal MouseBrainBench claim gate.
    """

    predictive_score: float = 0.30
    reproducibility_score: float = 0.70
    topology_effect: float = 0.05
    directed_fraction: float = 0.50
    matched_structure_function_effect: float = 0.01


@dataclass(frozen=True)
class ClaimDecision:
    """Allowed claims produced by one evaluator."""

    evaluator: str
    allowed_claims: tuple[str, ...]
    rationale: str

    def as_dict(self) -> dict[str, object]:
        return {
            "evaluator": self.evaluator,
            "allowed_claims": list(self.allowed_claims),
            "rationale": self.rationale,
        }


class ClaimEvaluator(Protocol):
    """Protocol implemented by all claim evaluators."""

    name: str

    def evaluate(self, evidence: ClaimEvidence) -> ClaimDecision:
        """Return the claim set allowed by this evaluator."""


def _clip01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


class CorrelationOnlyEvaluator:
    """Naive evaluator that overuses held-out prediction as claim support.

    This intentionally models a common failure mode: a model that predicts well
    is treated as supporting broad mechanistic language without asking whether
    topology, direction, or causal evidence is present.
    """

    name = "correlation_only"

    def __init__(self, threshold: float = 0.50) -> None:
        self.threshold = threshold

    def evaluate(self, evidence: ClaimEvidence) -> ClaimDecision:
        if evidence.predictive_score < self.threshold:
            return ClaimDecision(self.name, tuple(), "prediction below threshold")
        return ClaimDecision(
            self.name,
            (
                "predictive",
                "reproducible",
                "topology_specific",
                "directed",
                "structure_function",
                "mechanistic",
            ),
            "prediction above threshold is incorrectly treated as broad evidence",
        )


class CompensatoryScoreEvaluator:
    """Weighted-score evaluator where strong evidence can compensate weak blocks."""

    name = "compensatory_score"

    def __init__(self, threshold: float = 0.62) -> None:
        self.threshold = threshold

    def _score(self, evidence: ClaimEvidence) -> float:
        components = (
            _clip01(evidence.predictive_score),
            _clip01(evidence.reproducibility_score),
            _clip01(evidence.topology_effect / 0.08),
            _clip01(evidence.directed_fraction),
            _clip01(evidence.matched_structure_function_effect / 0.03),
        )
        weights = (0.25, 0.25, 0.20, 0.15, 0.15)
        return sum(weight * component for weight, component in zip(weights, components, strict=True))

    def evaluate(self, evidence: ClaimEvidence) -> ClaimDecision:
        claims: list[str] = []
        score = self._score(evidence)
        if evidence.predictive_score >= 0.30:
            claims.append("predictive")
        if evidence.reproducibility_score >= 0.70:
            claims.append("reproducible")
        if score >= self.threshold:
            claims.extend(["topology_specific", "directed", "structure_function", "mechanistic"])
        if score >= 0.85 and evidence.independent_validation:
            claims.append("digital_twin")
        return ClaimDecision(
            self.name,
            tuple(dict.fromkeys(claims)),
            f"weighted score={score:.3f}; high blocks can compensate failed blocks",
        )


class LeaderboardOnlyEvaluator:
    """Evaluator that equates leaderboard prediction with broad scientific support."""

    name = "leaderboard_only"

    def __init__(self, threshold: float = 0.60) -> None:
        self.threshold = threshold

    def evaluate(self, evidence: ClaimEvidence) -> ClaimDecision:
        if evidence.predictive_score < self.threshold:
            return ClaimDecision(self.name, tuple(), "leaderboard score below threshold")
        return ClaimDecision(
            self.name,
            (
                "predictive",
                "reproducible",
                "topology_specific",
                "directed",
                "structure_function",
                "mechanistic",
            ),
            "leaderboard prediction is incorrectly treated as mechanistic evidence",
        )


class ReliabilityOnlyEvaluator:
    """Evaluator that treats reproducibility as sufficient for mechanistic language."""

    name = "reliability_only"

    def __init__(self, threshold: float = 0.80) -> None:
        self.threshold = threshold

    def evaluate(self, evidence: ClaimEvidence) -> ClaimDecision:
        if evidence.reproducibility_score < self.threshold:
            return ClaimDecision(self.name, tuple(), "reproducibility below threshold")
        return ClaimDecision(
            self.name,
            ("reproducible", "topology_specific", "directed", "mechanistic"),
            "reproducibility alone is incorrectly treated as sufficient mechanism",
        )


class TopologyOnlyEvaluator:
    """Evaluator that overclaims from topological specificity alone."""

    name = "topology_only"

    def __init__(self, effect_threshold: float = 0.05) -> None:
        self.effect_threshold = effect_threshold

    def evaluate(self, evidence: ClaimEvidence) -> ClaimDecision:
        if not evidence.topology_specific or evidence.topology_effect < self.effect_threshold:
            return ClaimDecision(self.name, tuple(), "topology block below threshold")
        return ClaimDecision(
            self.name,
            ("topology_specific", "directed", "mechanistic"),
            "topology specificity is incorrectly treated as directed mechanism",
        )


class AblatedClaimGateEvaluator:
    """Claim gate with one evidence block removed for ablation testing."""

    name = "ablated_claim_gate_no_direction"

    def __init__(self, omitted_block: str = "directed") -> None:
        self.omitted_block = omitted_block
        self.name = f"ablated_claim_gate_no_{omitted_block}"

    def evaluate(self, evidence: ClaimEvidence) -> ClaimDecision:
        base = ClaimGateEvaluator().evaluate(evidence)
        claims = list(base.allowed_claims)
        predictive = evidence.predictive_score >= 0.30
        reproducible = evidence.reproducibility_score >= 0.70
        topology = evidence.topology_specific and evidence.topology_effect >= 0.05
        directed = evidence.directed_fraction >= 0.50
        if self.omitted_block == "directed":
            directed = True
        elif self.omitted_block == "topology":
            topology = True
        elif self.omitted_block == "reproducible":
            reproducible = True
        if predictive and reproducible and topology and directed and "mechanistic" not in claims:
            claims.append("mechanistic")
        return ClaimDecision(
            self.name,
            tuple(dict.fromkeys(claims)),
            f"non-compensatory gate with `{self.omitted_block}` evidence block ablated",
        )


class ClaimGateEvaluator:
    """MouseBrainBench non-compensatory claim gate."""

    name = "claim_gate"

    def __init__(self, thresholds: ClaimGateThresholds | None = None, name: str = "claim_gate") -> None:
        self.thresholds = thresholds or ClaimGateThresholds()
        self.name = name

    def evaluate(self, evidence: ClaimEvidence) -> ClaimDecision:
        claims: list[str] = []
        predictive = evidence.predictive_score >= self.thresholds.predictive_score
        reproducible = evidence.reproducibility_score >= self.thresholds.reproducibility_score
        topology = evidence.topology_specific and evidence.topology_effect >= self.thresholds.topology_effect
        directed = evidence.directed_fraction >= self.thresholds.directed_fraction
        structure_function = (
            evidence.structure_function_effect > 0.0
            and evidence.matched_structure_function_effect
            >= self.thresholds.matched_structure_function_effect
            and evidence.structure_function_fdr_passed
        )
        causal = evidence.causal_evidence
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
        if predictive and reproducible and topology and directed:
            claims.append("mechanistic")
        if causal:
            claims.append("causal")
        if (
            evidence.whole_brain_coverage
            and evidence.independent_validation
            and evidence.reproducible_compute
            and causal
        ):
            claims.append("digital_twin")
        return ClaimDecision(
            self.name,
            tuple(claims),
            "claims require their own evidence gates and cannot compensate each other",
        )


def claim_confusion_matrix(
    *,
    truth_by_case: dict[str, set[str]],
    decisions_by_evaluator: dict[str, dict[str, set[str]]],
) -> list[dict[str, int | str]]:
    """Build a per-evaluator, per-claim confusion matrix."""

    rows: list[dict[str, int | str]] = []
    for evaluator, case_decisions in decisions_by_evaluator.items():
        for claim in CLAIM_TYPES:
            tp = fp = tn = fn = 0
            for case_name, truth in truth_by_case.items():
                predicted = claim in case_decisions.get(case_name, set())
                expected = claim in truth
                if predicted and expected:
                    tp += 1
                elif predicted and not expected:
                    fp += 1
                elif not predicted and expected:
                    fn += 1
                else:
                    tn += 1
            rows.append(
                {
                    "evaluator": evaluator,
                    "claim": claim,
                    "tp": tp,
                    "fp": fp,
                    "tn": tn,
                    "fn": fn,
                }
            )
    return rows


def aggregate_claim_confusion(confusion: list[dict[str, int | str]]) -> list[dict[str, float | int | str]]:
    """Aggregate claim-level confusion rows into evaluator-level risk summaries."""

    rows: list[dict[str, float | int | str]] = []
    for evaluator in sorted({str(row["evaluator"]) for row in confusion}):
        subset = [row for row in confusion if row["evaluator"] == evaluator]
        tp = sum(int(row["tp"]) for row in subset)
        fp = sum(int(row["fp"]) for row in subset)
        tn = sum(int(row["tn"]) for row in subset)
        fn = sum(int(row["fn"]) for row in subset)
        rows.append(
            {
                "evaluator": evaluator,
                "tp": tp,
                "fp": fp,
                "tn": tn,
                "fn": fn,
                "false_positive_rate": fp / (fp + tn) if (fp + tn) else 0.0,
                "false_negative_rate": fn / (fn + tp) if (fn + tp) else 0.0,
                "overclaiming_risk_index": overclaiming_risk_index(fp=fp, tn=tn),
                "conservativeness_index": conservativeness_index(fn=fn, tp=tp),
            }
        )
    return rows


def overclaiming_risk_index(*, fp: int, tn: int) -> float:
    """Return the fraction of unsupported claim opportunities that were authorized."""

    return fp / (fp + tn) if (fp + tn) else 0.0


def conservativeness_index(*, fn: int, tp: int) -> float:
    """Return the fraction of supported claim opportunities that were blocked."""

    return fn / (fn + tp) if (fn + tp) else 0.0


def evidence_to_claim_contract(evidence: ClaimEvidence) -> dict[str, dict[str, object]]:
    """Expose the evidence contract used by the non-compensatory gate.

    The contract is intentionally executable. A manuscript can cite the same
    claim names used by this function, and the repository can verify whether the
    underlying evidence fields support those claims.
    """

    gate_claims = set(ClaimGateEvaluator().evaluate(evidence).allowed_claims)
    return {
        str(row["claim"]): {
            "level": row["level"],
            "meaning": row["meaning"],
            "required_evidence": list(row["required_evidence"]),
            "authorized": str(row["claim"]) in gate_claims,
        }
        for row in CLAIM_LADDER
    }
