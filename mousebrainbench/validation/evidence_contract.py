"""Domain-aware evidence contracts for claim authorization.

This module is the non-normalized successor to the original numerical claim
gate.  It deliberately keeps an observed measurement in the scale in which it
was reported and records the domain-specific rule that converted that
measurement into an evidence-block status.  This prevents a correlation, a
bootstrap interval, and a reproducibility coefficient from being compared as
if they shared a universal zero-to-one meaning.

The contract is a decision-support policy, not a truth engine.  A supported
status means that every declared block for the claim passed its own documented
rule.  It does not turn observational evidence into causal evidence and it does
not replace expert or external validation.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Iterable, Mapping


class EvidenceStatus(str, Enum):
    """Status of one evidence block before claim-level aggregation."""

    PASSED = "passed"
    FAILED = "failed"
    UNKNOWN = "unknown"
    NOT_APPLICABLE = "not_applicable"
    REQUIRES_REVIEW = "requires_review"


class DecisionStatus(str, Enum):
    """Workflow disposition produced for one scientific claim."""

    SUPPORTED = "supported"
    BLOCKED = "blocked"
    UNCERTAIN = "uncertain"
    OUT_OF_SCOPE = "out_of_scope"
    NEEDS_EXTERNAL_REVIEW = "needs_external_review"


@dataclass(frozen=True)
class EvidenceBlock:
    """One domain-specific evidence decision with its original observations.

    ``observations`` stores named values without rescaling them.  ``rule`` must
    describe the predicate applied by the source analysis.  A block may also be
    explicitly unknown or not applicable, which is different from failing a
    test that was actually performed.
    """

    name: str
    status: EvidenceStatus
    source: str
    rule: str
    rationale: str
    observations: tuple[tuple[str, Any], ...] = ()

    @classmethod
    def from_mapping(
        cls,
        *,
        name: str,
        status: EvidenceStatus,
        source: str,
        rule: str,
        rationale: str,
        observations: Mapping[str, Any] | None = None,
    ) -> "EvidenceBlock":
        """Build an immutable block while preserving observation names."""

        return cls(
            name=name,
            status=status,
            source=source,
            rule=rule,
            rationale=rationale,
            observations=tuple((str(key), value) for key, value in (observations or {}).items()),
        )

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""

        return {
            "name": self.name,
            "status": self.status.value,
            "source": self.source,
            "rule": self.rule,
            "rationale": self.rationale,
            "observations": dict(self.observations),
        }


@dataclass(frozen=True)
class ClaimRequirement:
    """Evidence blocks that must all pass for one claim."""

    claim: str
    required_blocks: tuple[str, ...]
    meaning: str


@dataclass(frozen=True)
class ContractDecision:
    """Auditable claim-level decision and the block states behind it."""

    claim: str
    status: DecisionStatus
    required_blocks: tuple[str, ...]
    block_statuses: tuple[tuple[str, EvidenceStatus], ...]
    rationale: str

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""

        return {
            "claim": self.claim,
            "status": self.status.value,
            "required_blocks": list(self.required_blocks),
            "block_statuses": {
                name: status.value for name, status in self.block_statuses
            },
            "rationale": self.rationale,
        }


CLAIM_REQUIREMENTS_V3 = (
    ClaimRequirement(
        "predictive",
        ("prediction",),
        "held-out prediction passes a protocol declared for the evaluated domain",
    ),
    ClaimRequirement(
        "computationally_reproducible",
        ("reproducible_compute",),
        "the declared software and artifact workflow can be reproduced from a clean revision",
    ),
    ClaimRequirement(
        "internally_reproduced",
        ("internal_reproduction",),
        "the result is reproduced in non-overlapping units or cohorts within the same resource",
    ),
    ClaimRequirement(
        "externally_replicated",
        ("external_replication",),
        "the result is replicated in an independent resource, laboratory, or study",
    ),
    ClaimRequirement(
        "topology_specific",
        ("topology_specificity",),
        "the tested topology outperforms prespecified topology controls",
    ),
    ClaimRequirement(
        "directed",
        ("directed_identifiability",),
        "direction is identified by a declared directional test",
    ),
    ClaimRequirement(
        "structure_function",
        ("structure_function_association",),
        "a local association survives its declared matched controls and multiplicity rule",
    ),
    ClaimRequirement(
        "mechanistic",
        (
            "prediction",
            "internal_reproduction",
            "topology_specificity",
            "directed_identifiability",
        ),
        "prediction, internal reproduction, topology specificity, and direction all pass",
    ),
    ClaimRequirement(
        "causal",
        ("causal_intervention",),
        "an intervention or another prespecified causal-identification design passes",
    ),
    ClaimRequirement(
        "digital_twin",
        (
            "prediction",
            "internal_reproduction",
            "topology_specificity",
            "directed_identifiability",
            "causal_intervention",
            "whole_brain_coverage",
            "independent_validation",
            "reproducible_compute",
        ),
        "all predictive, mechanistic, causal, coverage, validation, and compute blocks pass",
    ),
)


class EvidenceContractEvaluator:
    """Compatibility facade over the versioned claim knowledge system."""

    name = "evidence_contract_v3"

    def __init__(self, requirements: Iterable[ClaimRequirement] = CLAIM_REQUIREMENTS_V3) -> None:
        from mousebrainbench.knowledge.profile import KnowledgeProfile, load_default_profile

        declared = tuple(requirements)
        if declared == CLAIM_REQUIREMENTS_V3:
            profile = load_default_profile()
            if profile.requirements != CLAIM_REQUIREMENTS_V3:
                raise RuntimeError("packaged knowledge profile and v3 contract disagree")
        else:
            profile = KnowledgeProfile.from_requirements(declared)
        self.profile = profile
        self.requirements = {
            requirement.claim: requirement for requirement in profile.requirements
        }

    def evaluate_claim(
        self,
        claim: str,
        blocks: Mapping[str, EvidenceBlock],
    ) -> ContractDecision:
        """Evaluate one claim and retain every contributing block status."""

        from mousebrainbench.knowledge.engine import ClaimKnowledgeSystem

        return ClaimKnowledgeSystem(self.profile, blocks).infer(claim).decision

    def evaluate_all(self, blocks: Mapping[str, EvidenceBlock]) -> tuple[ContractDecision, ...]:
        """Evaluate all declared claims in stable contract order."""

        from mousebrainbench.knowledge.engine import ClaimKnowledgeSystem

        return tuple(
            inference.decision
            for inference in ClaimKnowledgeSystem(self.profile, blocks).infer_all()
        )

    def explain_claim(
        self,
        claim: str,
        blocks: Mapping[str, EvidenceBlock],
    ) -> dict[str, Any]:
        """Return the proof trace used to authorize or block one claim."""

        from mousebrainbench.knowledge.engine import ClaimKnowledgeSystem

        return ClaimKnowledgeSystem(self.profile, blocks).infer(claim).as_dict()


def blocks_by_name(blocks: Iterable[EvidenceBlock]) -> dict[str, EvidenceBlock]:
    """Index blocks and reject duplicate names that would hide provenance."""

    indexed: dict[str, EvidenceBlock] = {}
    for block in blocks:
        if block.name in indexed:
            raise ValueError(f"duplicate evidence block: {block.name}")
        indexed[block.name] = block
    return indexed
