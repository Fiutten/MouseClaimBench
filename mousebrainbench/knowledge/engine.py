"""Explainable non-compensatory inference over scientific evidence facts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from mousebrainbench.knowledge.profile import KnowledgeProfile
from mousebrainbench.validation.evidence_contract import (
    ContractDecision,
    DecisionStatus,
    EvidenceBlock,
    EvidenceStatus,
)


@dataclass(frozen=True)
class InferenceStep:
    """Evaluation of one knowledge rule with the facts that witness it."""

    rule_id: str
    priority: int
    triggered: bool
    witness_blocks: tuple[str, ...]
    conclusion: DecisionStatus
    rationale: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "rule_id": self.rule_id,
            "priority": self.priority,
            "triggered": self.triggered,
            "witness_blocks": list(self.witness_blocks),
            "conclusion": self.conclusion.value,
            "rationale": self.rationale,
        }


@dataclass(frozen=True)
class KnowledgeInference:
    """Claim decision accompanied by a complete, machine-readable proof trace."""

    profile_id: str
    profile_version: str
    profile_hash: str
    decision: ContractDecision
    fired_rule: str
    evidence_facts: tuple[EvidenceBlock, ...]
    steps: tuple[InferenceStep, ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "profile_id": self.profile_id,
            "profile_version": self.profile_version,
            "profile_hash": self.profile_hash,
            "decision": self.decision.as_dict(),
            "fired_rule": self.fired_rule,
            "evidence_facts": [fact.as_dict() for fact in self.evidence_facts],
            "steps": [step.as_dict() for step in self.steps],
        }


class ClaimKnowledgeSystem:
    """Knowledge base, inference engine, and explanation facility for one case."""

    def __init__(
        self,
        profile: KnowledgeProfile,
        evidence_blocks: Mapping[str, EvidenceBlock],
    ) -> None:
        self.profile = profile
        self.evidence_blocks = dict(evidence_blocks)
        for name, block in self.evidence_blocks.items():
            if name != block.name:
                raise ValueError(
                    f"evidence index {name!r} does not match block name {block.name!r}"
                )

    def infer(self, claim: str) -> KnowledgeInference:
        """Apply prioritized rules and return the decision with its proof trace."""

        requirement = self.profile.requirement(claim)
        if requirement is None:
            decision = ContractDecision(
                claim=claim,
                status=DecisionStatus.OUT_OF_SCOPE,
                required_blocks=(),
                block_statuses=(),
                rationale="no executable contract is declared for this claim",
            )
            return KnowledgeInference(
                profile_id=self.profile.profile_id,
                profile_version=self.profile.version,
                profile_hash=self.profile.source_hash,
                decision=decision,
                fired_rule="undeclared_claim_boundary",
                evidence_facts=(),
                steps=(
                    InferenceStep(
                        rule_id="undeclared_claim_boundary",
                        priority=1000,
                        triggered=True,
                        witness_blocks=(),
                        conclusion=DecisionStatus.OUT_OF_SCOPE,
                        rationale="no executable contract is declared for this claim",
                    ),
                ),
            )

        facts = tuple(
            self.evidence_blocks.get(
                block_name,
                EvidenceBlock.from_mapping(
                        name=block_name,
                        status=EvidenceStatus.UNKNOWN,
                        source="missing",
                        rule="no rule was executed",
                        rationale="required evidence block is absent",
                ),
            )
            for block_name in requirement.required_blocks
        )
        statuses = tuple((fact.name, fact.status) for fact in facts)

        steps: list[InferenceStep] = []
        for rule in self.profile.rules:
            witnesses = rule.witnesses(statuses)
            triggered = bool(witnesses)
            steps.append(
                InferenceStep(
                    rule_id=rule.rule_id,
                    priority=rule.priority,
                    triggered=triggered,
                    witness_blocks=witnesses,
                    conclusion=rule.conclusion,
                    rationale=rule.rationale,
                )
            )
            if triggered:
                decision = ContractDecision(
                    claim=claim,
                    status=rule.conclusion,
                    required_blocks=requirement.required_blocks,
                    block_statuses=statuses,
                    rationale=rule.rationale,
                )
                return KnowledgeInference(
                    profile_id=self.profile.profile_id,
                    profile_version=self.profile.version,
                    profile_hash=self.profile.source_hash,
                    decision=decision,
                    fired_rule=rule.rule_id,
                    evidence_facts=facts,
                    steps=tuple(steps),
                )
        raise RuntimeError(f"knowledge profile produced no disposition for claim {claim!r}")

    def infer_all(self) -> tuple[KnowledgeInference, ...]:
        """Infer every claim in stable ontology order."""

        return tuple(
            self.infer(requirement.claim) for requirement in self.profile.requirements
        )

    def knowledge_graph(self) -> dict[str, Any]:
        """Export a deterministic graph of ontology, rules, and case evidence facts."""

        nodes: list[dict[str, Any]] = [
            {
                "id": f"profile:{self.profile.profile_id}",
                "type": "knowledge_profile",
                "label": self.profile.profile_id,
            }
        ]
        edges: list[dict[str, str]] = []
        block_names = sorted(
            {
                block
                for requirement in self.profile.requirements
                for block in requirement.required_blocks
            }
        )
        for block_name in block_names:
            block = self.evidence_blocks.get(block_name)
            nodes.append(
                {
                    "id": f"evidence:{block_name}",
                    "type": "evidence_block",
                    "label": block_name,
                    "status": block.status.value if block is not None else "unknown",
                    "source": block.source if block is not None else "missing",
                }
            )
        for requirement in self.profile.requirements:
            claim_id = f"claim:{requirement.claim}"
            nodes.append(
                {"id": claim_id, "type": "claim", "label": requirement.claim}
            )
            edges.append(
                {
                    "source": f"profile:{self.profile.profile_id}",
                    "target": claim_id,
                    "relation": "declares",
                }
            )
            for block_name in requirement.required_blocks:
                edges.append(
                    {
                        "source": claim_id,
                        "target": f"evidence:{block_name}",
                        "relation": "requires",
                    }
                )
        for rule in self.profile.rules:
            rule_id = f"rule:{rule.rule_id}"
            decision_id = f"decision:{rule.conclusion.value}"
            status_id = f"evidence_status:{rule.evidence_status.value}"
            nodes.append(
                {
                    "id": rule_id,
                    "type": "inference_rule",
                    "label": rule.rule_id,
                    "priority": rule.priority,
                    "quantifier": rule.quantifier,
                }
            )
            if not any(node["id"] == status_id for node in nodes):
                nodes.append(
                    {
                        "id": status_id,
                        "type": "evidence_status",
                        "label": rule.evidence_status.value,
                    }
                )
            if not any(node["id"] == decision_id for node in nodes):
                nodes.append(
                    {
                        "id": decision_id,
                        "type": "decision_status",
                        "label": rule.conclusion.value,
                    }
                )
            edges.append(
                {
                    "source": f"profile:{self.profile.profile_id}",
                    "target": rule_id,
                    "relation": "declares",
                }
            )
            edges.append(
                {"source": rule_id, "target": status_id, "relation": "tests_status"}
            )
            edges.append(
                {"source": rule_id, "target": decision_id, "relation": "concludes"}
            )
        return {
            "profile_id": self.profile.profile_id,
            "profile_version": self.profile.version,
            "profile_hash": self.profile.source_hash,
            "nodes": sorted(nodes, key=lambda node: node["id"]),
            "edges": sorted(
                edges,
                key=lambda edge: (edge["source"], edge["relation"], edge["target"]),
            ),
        }
