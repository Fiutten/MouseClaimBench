"""Versioned knowledge profiles for evidence-constrained claim inference."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from importlib.resources import files
from typing import Any, Iterable, Mapping

import yaml

from mousebrainbench.validation.evidence_contract import (
    ClaimRequirement,
    DecisionStatus,
    EvidenceStatus,
)


@dataclass(frozen=True)
class InferenceRule:
    """One declarative rule in the non-compensatory inference policy."""

    rule_id: str
    priority: int
    quantifier: str
    evidence_status: EvidenceStatus
    conclusion: DecisionStatus
    rationale: str

    def witnesses(
        self,
        statuses: tuple[tuple[str, EvidenceStatus], ...],
    ) -> tuple[str, ...]:
        """Return evidence blocks that make the rule true, or an empty tuple."""

        matching = tuple(name for name, status in statuses if status is self.evidence_status)
        if self.quantifier == "any":
            return matching
        if self.quantifier == "all" and statuses and len(matching) == len(statuses):
            return matching
        return ()


@dataclass(frozen=True)
class KnowledgeProfile:
    """Immutable claim ontology and rule base for one application domain."""

    profile_id: str
    version: str
    domain: str
    description: str
    requirements: tuple[ClaimRequirement, ...]
    rules: tuple[InferenceRule, ...]
    source_hash: str

    @classmethod
    def from_mapping(
        cls,
        payload: Mapping[str, Any],
        *,
        source_hash: str = "in-memory",
    ) -> "KnowledgeProfile":
        """Validate and construct a profile from a YAML-compatible mapping."""

        claims = tuple(
            ClaimRequirement(
                claim=str(row["id"]),
                required_blocks=tuple(str(name) for name in row["required_blocks"]),
                meaning=str(row["meaning"]),
            )
            for row in payload.get("claims", ())
        )
        rules = tuple(
            InferenceRule(
                rule_id=str(row["id"]),
                priority=int(row["priority"]),
                quantifier=str(row["quantifier"]),
                evidence_status=EvidenceStatus(str(row["evidence_status"])),
                conclusion=DecisionStatus(str(row["conclusion"])),
                rationale=str(row["rationale"]),
            )
            for row in payload.get("inference_rules", ())
        )
        cls._validate(claims, rules)
        return cls(
            profile_id=str(payload["profile_id"]),
            version=str(payload["version"]),
            domain=str(payload["domain"]),
            description=str(payload["description"]),
            requirements=claims,
            rules=tuple(sorted(rules, key=lambda rule: (-rule.priority, rule.rule_id))),
            source_hash=source_hash,
        )

    @classmethod
    def from_requirements(
        cls,
        requirements: Iterable[ClaimRequirement],
        *,
        profile_id: str = "runtime_contract",
        domain: str = "runtime",
    ) -> "KnowledgeProfile":
        """Create a structurally checked runtime profile for custom contracts."""

        payload = {
            "profile_id": profile_id,
            "version": "runtime",
            "domain": domain,
            "description": "Runtime profile generated from explicit claim requirements.",
            "claims": [
                {
                    "id": requirement.claim,
                    "required_blocks": list(requirement.required_blocks),
                    "meaning": requirement.meaning,
                }
                for requirement in requirements
            ],
            "inference_rules": _standard_rule_payload(),
        }
        return cls.from_mapping(payload)

    @staticmethod
    def _validate(
        requirements: tuple[ClaimRequirement, ...],
        rules: tuple[InferenceRule, ...],
    ) -> None:
        if not requirements:
            raise ValueError("a knowledge profile must declare at least one claim")
        claim_ids = [requirement.claim for requirement in requirements]
        if len(claim_ids) != len(set(claim_ids)):
            raise ValueError("duplicate claim identifier in knowledge profile")
        for requirement in requirements:
            if not requirement.required_blocks:
                raise ValueError(f"claim {requirement.claim!r} has no evidence requirements")
            if len(requirement.required_blocks) != len(set(requirement.required_blocks)):
                raise ValueError(f"claim {requirement.claim!r} repeats an evidence block")

        rule_ids = [rule.rule_id for rule in rules]
        if len(rule_ids) != len(set(rule_ids)):
            raise ValueError("duplicate inference-rule identifier in knowledge profile")
        priorities = [rule.priority for rule in rules]
        if len(priorities) != len(set(priorities)):
            raise ValueError("inference-rule priorities must be unique")
        if any(rule.quantifier not in {"any", "all"} for rule in rules):
            raise ValueError("inference-rule quantifier must be 'any' or 'all'")
        conclusions = {rule.conclusion for rule in rules}
        if conclusions != set(DecisionStatus):
            missing = sorted(status.value for status in set(DecisionStatus) - conclusions)
            extra = sorted(status.value for status in conclusions - set(DecisionStatus))
            raise ValueError(
                "knowledge profile must cover every decision status: "
                f"missing={missing}, extra={extra}"
            )
        terminal = [
            rule
            for rule in rules
            if rule.quantifier == "all"
            and rule.evidence_status is EvidenceStatus.PASSED
            and rule.conclusion is DecisionStatus.SUPPORTED
        ]
        if len(terminal) != 1:
            raise ValueError("knowledge profile requires one all-passed terminal rule")

    def requirement(self, claim: str) -> ClaimRequirement | None:
        """Return the declared requirement for a claim identifier."""

        return next(
            (requirement for requirement in self.requirements if requirement.claim == claim),
            None,
        )

    def as_dict(self) -> dict[str, Any]:
        """Return the profile as a deterministic serializable knowledge artifact."""

        return {
            "profile_id": self.profile_id,
            "version": self.version,
            "domain": self.domain,
            "description": self.description,
            "source_hash": self.source_hash,
            "claims": [
                {
                    "id": requirement.claim,
                    "required_blocks": list(requirement.required_blocks),
                    "meaning": requirement.meaning,
                }
                for requirement in self.requirements
            ],
            "inference_rules": [
                {
                    "id": rule.rule_id,
                    "priority": rule.priority,
                    "quantifier": rule.quantifier,
                    "evidence_status": rule.evidence_status.value,
                    "conclusion": rule.conclusion.value,
                    "rationale": rule.rationale,
                }
                for rule in self.rules
            ],
        }


def _standard_rule_payload() -> list[dict[str, Any]]:
    return [
        {
            "id": "failed_block_veto",
            "priority": 100,
            "quantifier": "any",
            "evidence_status": "failed",
            "conclusion": "blocked",
            "rationale": "at least one required evidence block failed",
        },
        {
            "id": "external_review_escalation",
            "priority": 90,
            "quantifier": "any",
            "evidence_status": "requires_review",
            "conclusion": "needs_external_review",
            "rationale": "at least one required block cannot be decided automatically",
        },
        {
            "id": "missing_evidence_uncertainty",
            "priority": 80,
            "quantifier": "any",
            "evidence_status": "unknown",
            "conclusion": "uncertain",
            "rationale": "at least one required evidence block was not observed",
        },
        {
            "id": "protocol_scope_boundary",
            "priority": 70,
            "quantifier": "any",
            "evidence_status": "not_applicable",
            "conclusion": "out_of_scope",
            "rationale": "the evaluated protocol did not target at least one required block",
        },
        {
            "id": "all_requirements_satisfied",
            "priority": 0,
            "quantifier": "all",
            "evidence_status": "passed",
            "conclusion": "supported",
            "rationale": "all required evidence blocks passed their declared domain-specific rules",
        },
    ]


def load_default_profile() -> KnowledgeProfile:
    """Load the packaged mouse-brain claim ontology and rule base."""

    resource = files("mousebrainbench.knowledge.profiles").joinpath(
        "mouse_brain_claims_v1.yaml"
    )
    source = resource.read_text(encoding="utf-8")
    payload = yaml.safe_load(source)
    if not isinstance(payload, Mapping):
        raise ValueError("default knowledge profile must contain a mapping")
    digest = hashlib.sha256(source.encode("utf-8")).hexdigest()
    return KnowledgeProfile.from_mapping(payload, source_hash=f"sha256:{digest}")


def load_default_profile_basis() -> dict[str, Any]:
    """Load and validate the curation record for every default claim relation."""

    resource = files("mousebrainbench.knowledge.profiles").joinpath(
        "mouse_brain_claims_v1_basis.yaml"
    )
    payload = yaml.safe_load(resource.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError("default profile basis must contain a mapping")

    profile = load_default_profile()
    if payload.get("profile_id") != profile.profile_id:
        raise ValueError("profile basis identifier does not match the default profile")
    if str(payload.get("version")) != profile.version:
        raise ValueError("profile basis version does not match the default profile")
    if payload.get("independent_expert_validation") != "not_performed":
        raise ValueError("profile basis must report the actual expert-validation status")

    rows = payload.get("relations")
    if not isinstance(rows, list):
        raise ValueError("profile basis must contain a relation list")
    required_fields = {
        "claim",
        "evidence_block",
        "role",
        "rationale",
        "scope",
        "exceptions",
        "alternatives_rejected",
        "source_ids",
    }
    observed: set[tuple[str, str]] = set()
    for row in rows:
        if not isinstance(row, Mapping) or not required_fields <= row.keys():
            raise ValueError("every profile relation requires a complete curation record")
        key = (str(row["claim"]), str(row["evidence_block"]))
        if key in observed:
            raise ValueError(f"duplicate profile-basis relation: {key}")
        if not row["source_ids"] or not all(
            isinstance(source_id, str) and source_id for source_id in row["source_ids"]
        ):
            raise ValueError(f"profile-basis relation has no sources: {key}")
        observed.add(key)

    expected = {
        (requirement.claim, block)
        for requirement in profile.requirements
        for block in requirement.required_blocks
    }
    if observed != expected:
        missing = sorted(expected - observed)
        extra = sorted(observed - expected)
        raise ValueError(
            f"profile basis does not cover the executable relation set: missing={missing}, "
            f"extra={extra}"
        )
    return dict(payload)
