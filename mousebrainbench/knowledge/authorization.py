"""Strict profile authorization with complete, non-prioritized deficit traces.

The v1 knowledge engine is retained for reproduction of frozen experiments. This
module implements the hardened v2 semantics. It deliberately distinguishes
profile authorization from scientific truth and preserves every unmet evidence
requirement instead of collapsing mixed states through a priority rule.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from functools import lru_cache
from importlib.resources import files
from typing import Any

import yaml

from mousebrainbench.validation.evidence_contract import EvidenceBlock, EvidenceStatus


class ProfileAuthorizationStatus(str, Enum):
    """Disposition under one named and versioned operational profile."""

    AUTHORIZED = "profile_authorized"
    NOT_AUTHORIZED = "profile_not_authorized"
    OUTSIDE_PROFILE = "outside_profile"


@dataclass(frozen=True)
class EvidenceBlockSpecification:
    """Minimum provenance fields required before a passing fact is admissible."""

    name: str
    meaning: str
    required_observations_when_passed: tuple[str, ...]


@dataclass(frozen=True)
class AuthorizationRequirement:
    """Evidence blocks that must all be admissibly passed for one bounded claim."""

    claim: str
    required_blocks: tuple[str, ...]
    meaning: str


@dataclass(frozen=True)
class ClaimAuthorizationProfile:
    """Immutable v2 claim vocabulary and evidence-schema contract."""

    profile_id: str
    version: str
    domain: str
    status: str
    description: str
    claim_boundary: str
    evidence_blocks: tuple[EvidenceBlockSpecification, ...]
    requirements: tuple[AuthorizationRequirement, ...]
    source_hash: str

    @classmethod
    def from_mapping(
        cls,
        payload: Mapping[str, Any],
        *,
        source_hash: str = "in-memory",
    ) -> ClaimAuthorizationProfile:
        """Build and structurally validate a v2 authorization profile."""

        specifications = tuple(
            EvidenceBlockSpecification(
                name=str(row["id"]),
                meaning=str(row["meaning"]),
                required_observations_when_passed=tuple(
                    str(value) for value in row["required_observations_when_passed"]
                ),
            )
            for row in payload.get("evidence_blocks", ())
        )
        requirements = tuple(
            AuthorizationRequirement(
                claim=str(row["id"]),
                required_blocks=tuple(str(value) for value in row["required_blocks"]),
                meaning=str(row["meaning"]),
            )
            for row in payload.get("claims", ())
        )
        cls._validate(specifications, requirements)
        return cls(
            profile_id=str(payload["profile_id"]),
            version=str(payload["version"]),
            domain=str(payload["domain"]),
            status=str(payload["status"]),
            description=str(payload["description"]),
            claim_boundary=str(payload["claim_boundary"]),
            evidence_blocks=specifications,
            requirements=requirements,
            source_hash=source_hash,
        )

    @staticmethod
    def _validate(
        specifications: tuple[EvidenceBlockSpecification, ...],
        requirements: tuple[AuthorizationRequirement, ...],
    ) -> None:
        if not specifications or not requirements:
            raise ValueError("an authorization profile requires evidence blocks and claims")
        block_names = [item.name for item in specifications]
        claim_names = [item.claim for item in requirements]
        if len(block_names) != len(set(block_names)):
            raise ValueError("duplicate evidence-block specification")
        if len(claim_names) != len(set(claim_names)):
            raise ValueError("duplicate authorization claim")
        declared = set(block_names)
        for specification in specifications:
            fields = specification.required_observations_when_passed
            if not fields or len(fields) != len(set(fields)):
                raise ValueError(
                    f"evidence block {specification.name!r} requires unique observation fields"
                )
        for requirement in requirements:
            if not requirement.required_blocks:
                raise ValueError(f"claim {requirement.claim!r} has no evidence requirements")
            if len(requirement.required_blocks) != len(set(requirement.required_blocks)):
                raise ValueError(f"claim {requirement.claim!r} repeats an evidence block")
            unknown = set(requirement.required_blocks) - declared
            if unknown:
                raise ValueError(
                    f"claim {requirement.claim!r} uses undeclared evidence blocks: "
                    f"{sorted(unknown)}"
                )

    def requirement(self, claim: str) -> AuthorizationRequirement | None:
        return next((item for item in self.requirements if item.claim == claim), None)

    def block_specification(self, name: str) -> EvidenceBlockSpecification:
        match = next((item for item in self.evidence_blocks if item.name == name), None)
        if match is None:
            raise KeyError(f"undeclared evidence block: {name}")
        return match

    def as_dict(self) -> dict[str, Any]:
        return {
            "profile_id": self.profile_id,
            "version": self.version,
            "domain": self.domain,
            "status": self.status,
            "description": self.description,
            "claim_boundary": self.claim_boundary,
            "source_hash": self.source_hash,
            "evidence_blocks": [
                {
                    "id": item.name,
                    "meaning": item.meaning,
                    "required_observations_when_passed": list(
                        item.required_observations_when_passed
                    ),
                }
                for item in self.evidence_blocks
            ],
            "claims": [
                {
                    "id": item.claim,
                    "required_blocks": list(item.required_blocks),
                    "meaning": item.meaning,
                }
                for item in self.requirements
            ],
        }


@dataclass(frozen=True)
class EvaluatedEvidenceFact:
    """A source fact after applying the profile's provenance-schema guard."""

    name: str
    declared_status: EvidenceStatus
    effective_status: EvidenceStatus
    source: str
    rule: str
    rationale: str
    observations: tuple[tuple[str, Any], ...]
    missing_required_observations: tuple[str, ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "declared_status": self.declared_status.value,
            "effective_status": self.effective_status.value,
            "source": self.source,
            "rule": self.rule,
            "rationale": self.rationale,
            "observations": dict(self.observations),
            "missing_required_observations": list(self.missing_required_observations),
        }


@dataclass(frozen=True)
class ProfileAuthorizationDecision:
    """Authorization result with every deficit preserved as a first-class fact."""

    profile_id: str
    profile_version: str
    profile_hash: str
    claim: str
    status: ProfileAuthorizationStatus
    meaning: str
    required_blocks: tuple[str, ...]
    facts: tuple[EvaluatedEvidenceFact, ...]

    @property
    def authorized(self) -> bool:
        return self.status is ProfileAuthorizationStatus.AUTHORIZED

    @property
    def deficits(self) -> tuple[EvaluatedEvidenceFact, ...]:
        return tuple(
            fact for fact in self.facts if fact.effective_status is not EvidenceStatus.PASSED
        )

    def as_dict(self) -> dict[str, Any]:
        counts = {
            status.value: sum(
                fact.effective_status is status for fact in self.deficits
            )
            for status in EvidenceStatus
            if status is not EvidenceStatus.PASSED
        }
        return {
            "profile_id": self.profile_id,
            "profile_version": self.profile_version,
            "profile_hash": self.profile_hash,
            "claim": self.claim,
            "status": self.status.value,
            "authorized": self.authorized,
            "meaning": self.meaning,
            "required_blocks": list(self.required_blocks),
            "facts": [fact.as_dict() for fact in self.facts],
            "deficits": [fact.as_dict() for fact in self.deficits],
            "deficit_counts": counts,
        }


def _present(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, (tuple, list, dict, set)):
        return bool(value)
    return True


class ClaimAuthorizationSystem:
    """Authorize bounded claims without compensation or deficit masking."""

    def __init__(
        self,
        profile: ClaimAuthorizationProfile,
        evidence_blocks: Mapping[str, EvidenceBlock],
    ) -> None:
        self.profile = profile
        self.evidence_blocks = dict(evidence_blocks)
        for name, block in self.evidence_blocks.items():
            if name != block.name:
                raise ValueError(
                    f"evidence index {name!r} does not match block name {block.name!r}"
                )

    def _evaluate_fact(self, name: str) -> EvaluatedEvidenceFact:
        block = self.evidence_blocks.get(name)
        if block is None:
            return EvaluatedEvidenceFact(
                name=name,
                declared_status=EvidenceStatus.UNKNOWN,
                effective_status=EvidenceStatus.UNKNOWN,
                source="missing",
                rule="no evidence predicate was executed",
                rationale="required evidence block is absent",
                observations=(),
                missing_required_observations=(),
            )
        specification = self.profile.block_specification(name)
        observations = dict(block.observations)
        missing = (
            tuple(
                field
                for field in specification.required_observations_when_passed
                if field not in observations or not _present(observations[field])
            )
            if block.status is EvidenceStatus.PASSED
            else ()
        )
        metadata_missing = not _present(block.source) or not _present(block.rule) or not _present(
            block.rationale
        )
        if metadata_missing:
            missing = (*missing, "source_rule_or_rationale")
        effective = (
            EvidenceStatus.REQUIRES_REVIEW
            if block.status is EvidenceStatus.PASSED and missing
            else block.status
        )
        return EvaluatedEvidenceFact(
            name=name,
            declared_status=block.status,
            effective_status=effective,
            source=block.source,
            rule=block.rule,
            rationale=block.rationale,
            observations=block.observations,
            missing_required_observations=tuple(missing),
        )

    def infer(self, claim: str) -> ProfileAuthorizationDecision:
        """Return authorization plus the complete set of unmet requirements."""

        requirement = self.profile.requirement(claim)
        if requirement is None:
            return ProfileAuthorizationDecision(
                profile_id=self.profile.profile_id,
                profile_version=self.profile.version,
                profile_hash=self.profile.source_hash,
                claim=claim,
                status=ProfileAuthorizationStatus.OUTSIDE_PROFILE,
                meaning="the claim is not declared by this profile",
                required_blocks=(),
                facts=(),
            )
        facts = tuple(self._evaluate_fact(name) for name in requirement.required_blocks)
        authorized = all(
            fact.effective_status is EvidenceStatus.PASSED for fact in facts
        )
        return ProfileAuthorizationDecision(
            profile_id=self.profile.profile_id,
            profile_version=self.profile.version,
            profile_hash=self.profile.source_hash,
            claim=claim,
            status=(
                ProfileAuthorizationStatus.AUTHORIZED
                if authorized
                else ProfileAuthorizationStatus.NOT_AUTHORIZED
            ),
            meaning=requirement.meaning,
            required_blocks=requirement.required_blocks,
            facts=facts,
        )

    def infer_all(self) -> tuple[ProfileAuthorizationDecision, ...]:
        return tuple(self.infer(item.claim) for item in self.profile.requirements)


@lru_cache(maxsize=1)
def load_authorization_profile_v2() -> ClaimAuthorizationProfile:
    """Load the hardened mouse-brain claim-authorization profile."""

    resource = files("mousebrainbench.knowledge.profiles").joinpath(
        "mouse_brain_claims_v2.yaml"
    )
    source = resource.read_text(encoding="utf-8")
    payload = yaml.safe_load(source)
    if not isinstance(payload, Mapping):
        raise TypeError("v2 authorization profile must contain a mapping")
    digest = hashlib.sha256(source.encode("utf-8")).hexdigest()
    return ClaimAuthorizationProfile.from_mapping(
        payload,
        source_hash=f"sha256:{digest}",
    )


def load_authorization_profile_v2_basis() -> dict[str, Any]:
    """Load and validate the complete v2 curation and internal-audit record."""

    resource = files("mousebrainbench.knowledge.profiles").joinpath(
        "mouse_brain_claims_v2_basis.yaml"
    )
    payload = yaml.safe_load(resource.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise TypeError("v2 profile basis must contain a mapping")
    profile = load_authorization_profile_v2()
    if payload.get("profile_id") != profile.profile_id:
        raise ValueError("v2 basis identifier does not match the profile")
    if str(payload.get("version")) != profile.version:
        raise ValueError("v2 basis version does not match the profile")
    rows = payload.get("relations")
    if not isinstance(rows, list):
        raise TypeError("v2 profile basis must contain a relation list")
    expected = {
        (requirement.claim, block)
        for requirement in profile.requirements
        for block in requirement.required_blocks
    }
    required_fields = {
        "claim",
        "evidence_block",
        "role",
        "rationale",
        "scope",
        "exceptions",
        "source_ids",
        "audit_resolution",
    }
    observed: set[tuple[str, str]] = set()
    for row in rows:
        if not isinstance(row, Mapping) or not required_fields <= row.keys():
            raise ValueError("every v2 relation requires a complete curation record")
        key = (str(row["claim"]), str(row["evidence_block"]))
        if key in observed:
            raise ValueError(f"duplicate v2 profile-basis relation: {key}")
        if not row["source_ids"]:
            raise ValueError(f"v2 profile-basis relation has no source: {key}")
        observed.add(key)
    if observed != expected:
        raise ValueError(
            "v2 profile basis does not exactly cover the executable relations: "
            f"missing={sorted(expected - observed)}, extra={sorted(observed - expected)}"
        )
    return dict(payload)
