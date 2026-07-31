"""Mechanistic-identifiability scoring for partial neural digital-twin targets."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


Direction = Literal["gt", "gte", "lt", "lte"]


@dataclass(frozen=True)
class Criterion:
    """One preregistered criterion contributing to a mechanistic claim."""

    name: str
    value: float
    threshold: float
    direction: Direction = "gt"

    @property
    def passed(self) -> bool:
        if self.direction == "gt":
            return self.value > self.threshold
        if self.direction == "gte":
            return self.value >= self.threshold
        if self.direction == "lt":
            return self.value < self.threshold
        return self.value <= self.threshold

    @property
    def normalized_score(self) -> float:
        """Map supporting evidence to [0, 1] without changing pass/fail semantics.

        The normalized value is descriptive only. A model cannot compensate for a
        failed non-interchangeable block by performing well on another block.
        """

        if self.threshold == 0:
            return 1.0 if self.passed else 0.0
        if self.direction in {"gt", "gte"}:
            return max(0.0, min(1.0, self.value / self.threshold))
        if self.value <= 0:
            return 1.0 if self.passed else 0.0
        return max(0.0, min(1.0, self.threshold / self.value))

    def as_dict(self) -> dict[str, float | str | bool]:
        return {
            "name": self.name,
            "value": self.value,
            "threshold": self.threshold,
            "direction": self.direction,
            "passed": self.passed,
            "normalized_score": self.normalized_score,
        }


@dataclass(frozen=True)
class EvidenceBlock:
    """A non-interchangeable validation layer in the MIS hierarchy.

    Blocks are intentionally conjunctive: reproducibility, topology specificity,
    and directed identifiability represent different scientific claims.
    """

    name: str
    criteria: tuple[Criterion, ...]

    @property
    def passed(self) -> bool:
        return all(criterion.passed for criterion in self.criteria)

    @property
    def score(self) -> float:
        if not self.criteria:
            raise ValueError("evidence block must contain at least one criterion")
        return sum(criterion.normalized_score for criterion in self.criteria) / len(self.criteria)

    def as_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "passed": self.passed,
            "score": self.score,
            "criteria": [criterion.as_dict() for criterion in self.criteria],
        }


@dataclass(frozen=True)
class MechanisticIdentifiabilityScore:
    """Conjunctive score for deciding whether a mechanistic claim is supported."""

    blocks: tuple[EvidenceBlock, ...]

    @property
    def passed(self) -> bool:
        return all(block.passed for block in self.blocks)

    @property
    def score(self) -> float:
        if not self.blocks:
            raise ValueError("MIS requires at least one evidence block")
        return sum(block.score for block in self.blocks) / len(self.blocks)

    def as_dict(self) -> dict[str, object]:
        return {
            "score": self.score,
            "passed": self.passed,
            "blocks": [block.as_dict() for block in self.blocks],
            "interpretation": (
                "mechanistically_identifiable"
                if self.passed
                else "not_mechanistically_identifiable"
            ),
        }


def build_mis_from_blocks(
    *,
    reproducibility: tuple[Criterion, ...],
    topology_specificity: tuple[Criterion, ...],
    directed_identifiability: tuple[Criterion, ...],
) -> MechanisticIdentifiabilityScore:
    """Build the fixed three-layer MIS used in the Q1 strategy."""

    return MechanisticIdentifiabilityScore(
        (
            EvidenceBlock("reproducibility", reproducibility),
            EvidenceBlock("topology_specificity", topology_specificity),
            EvidenceBlock("directed_identifiability", directed_identifiability),
        )
    )
