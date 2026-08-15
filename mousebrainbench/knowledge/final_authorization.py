"""Explicit non-compensatory composition of structural, domain, and integrity gates."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from mousebrainbench.knowledge.authorization import (
    ClaimAuthorizationProfile,
    ClaimAuthorizationSystem,
    ProfileAuthorizationDecision,
)
from mousebrainbench.knowledge.integrity import (
    EvidencePackageManifest,
    IntegrityDeficit,
    validate_evidence_manifest,
)
from mousebrainbench.knowledge.standards import (
    StructuralConformanceDecision,
    validate_structure_with_shacl_v2,
)
from mousebrainbench.validation.evidence_contract import EvidenceBlock


def compose_final_authorization(
    structural_conforms: bool,
    domain_authorized: bool,
    integrity_conforms: bool,
) -> bool:
    """Return the truth-functional S and A and I composition."""

    return structural_conforms and domain_authorized and integrity_conforms


@dataclass(frozen=True)
class FinalAuthorizationDecision:
    """Layer-resolved decision for one graph, evidence package, and manifest."""

    structural: StructuralConformanceDecision
    domain: ProfileAuthorizationDecision
    integrity_deficits: tuple[IntegrityDeficit, ...]
    authorized: bool

    @property
    def integrity_conforms(self) -> bool:
        return not self.integrity_deficits

    def as_dict(self) -> dict[str, object]:
        return {
            "authorized": self.authorized,
            "structural": self.structural.as_dict(),
            "domain": self.domain.as_dict(),
            "integrity": {
                "conforms": self.integrity_conforms,
                "deficits": [row.as_dict() for row in self.integrity_deficits],
            },
        }


class FinalAuthorizationSystem:
    """Evaluate all three gates over one shared claim-package-manifest input.

    The structural gate requires the optional ``full-authorization`` or
    backward-compatible ``standards-validation`` dependency group.
    """

    def __init__(
        self,
        profile: ClaimAuthorizationProfile,
        evidence_blocks: Mapping[str, EvidenceBlock],
        manifest: EvidencePackageManifest,
    ) -> None:
        self.profile = profile
        self.evidence_blocks = dict(evidence_blocks)
        self.manifest = manifest

    def infer(self, claim: str) -> FinalAuthorizationDecision:
        """Execute S, A, and I independently, then compose their decisions."""

        structural = validate_structure_with_shacl_v2(
            self.profile,
            claim,
            self.evidence_blocks,
            package_id=self.manifest.package_id,
            manifest=self.manifest,
        )
        domain = ClaimAuthorizationSystem(
            self.profile, self.evidence_blocks
        ).infer(claim)
        integrity_deficits = validate_evidence_manifest(
            self.profile, self.evidence_blocks, self.manifest
        )
        authorized = compose_final_authorization(
            structural.conforms,
            domain.authorized,
            not integrity_deficits,
        )
        return FinalAuthorizationDecision(
            structural=structural,
            domain=domain,
            integrity_deficits=integrity_deficits,
            authorized=authorized,
        )
