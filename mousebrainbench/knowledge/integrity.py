"""Evidence-package integrity checks that precede profile authorization.

The validator handles relationships among artifacts. These checks complement
SHACL field validation and must not be interpreted as proof that an artifact is
scientifically correct or that a source is trustworthy.
"""

from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass
from enum import Enum

from mousebrainbench.knowledge.authorization import (
    ClaimAuthorizationProfile,
    ClaimAuthorizationSystem,
    ProfileAuthorizationDecision,
)
from mousebrainbench.validation.evidence_contract import EvidenceBlock, EvidenceStatus

SHA256_PATTERN = re.compile(r"^sha256:[0-9a-f]{64}$")


class IntegrityDeficitCode(str, Enum):
    """Non-compensatory package-integrity failures."""

    PROFILE_IDENTITY_MISMATCH = "profile_identity_mismatch"
    ARTIFACT_HASH_MISMATCH = "artifact_hash_mismatch"
    UNKNOWN_PROVENANCE_REFERENCE = "unknown_provenance_reference"
    PROVENANCE_CYCLE = "provenance_cycle"
    DUPLICATE_INDEPENDENT_ARTIFACT = "duplicate_independent_artifact"
    OVERLAPPING_INDEPENDENT_COHORTS = "overlapping_independent_cohorts"
    CONTRADICTORY_ATTESTATION = "contradictory_attestation"
    MISSING_BLOCK_LINEAGE = "missing_block_lineage"


@dataclass(frozen=True)
class ArtifactRecord:
    """One content-addressed source and its declared derivation context."""

    artifact_id: str
    declared_sha256: str
    observed_sha256: str
    derived_from: tuple[str, ...] = ()
    cohorts: tuple[str, ...] = ()
    study_id: str = ""
    data_generation_id: str = ""


@dataclass(frozen=True)
class EvidenceAttestation:
    """A source-specific status statement for one evidence block."""

    block_name: str
    status: EvidenceStatus
    artifact_id: str


@dataclass(frozen=True)
class EvidencePackageManifest:
    """Profile identity, source lineage, and independence declarations."""

    package_id: str
    profile_id: str
    profile_version: str
    profile_hash: str
    artifacts: tuple[ArtifactRecord, ...]
    block_artifacts: tuple[tuple[str, tuple[str, ...]], ...]
    attestations: tuple[EvidenceAttestation, ...]
    independent_artifact_pairs: tuple[tuple[str, str], ...] = ()
    disjoint_cohort_pairs: tuple[tuple[str, str], ...] = ()

    def artifact_index(self) -> dict[str, ArtifactRecord]:
        indexed = {artifact.artifact_id: artifact for artifact in self.artifacts}
        if len(indexed) != len(self.artifacts):
            raise ValueError("manifest repeats an artifact identifier")
        return indexed

    def block_index(self) -> dict[str, tuple[str, ...]]:
        indexed = dict(self.block_artifacts)
        if len(indexed) != len(self.block_artifacts):
            raise ValueError("manifest repeats block lineage")
        return indexed


@dataclass(frozen=True)
class IntegrityDeficit:
    """One actionable integrity failure with its witnesses."""

    code: IntegrityDeficitCode
    witnesses: tuple[str, ...]
    rationale: str

    def as_dict(self) -> dict[str, object]:
        return {
            "code": self.code.value,
            "witnesses": list(self.witnesses),
            "rationale": self.rationale,
        }


@dataclass(frozen=True)
class IntegrityAwareDecision:
    """Core profile decision gated by complete package-integrity deficits."""

    core: ProfileAuthorizationDecision
    integrity_deficits: tuple[IntegrityDeficit, ...]

    @property
    def authorized(self) -> bool:
        return self.core.authorized and not self.integrity_deficits

    def as_dict(self) -> dict[str, object]:
        return {
            "authorized": self.authorized,
            "core": self.core.as_dict(),
            "integrity_deficits": [row.as_dict() for row in self.integrity_deficits],
        }


def _has_cycle(artifacts: dict[str, ArtifactRecord]) -> tuple[str, ...]:
    # Iterative depth-first search avoids Python's recursion limit for evidence
    # packages containing thousands of artifacts.
    state: dict[str, int] = {}
    for root in sorted(artifacts):
        if state.get(root, 0) != 0:
            continue
        stack: list[tuple[str, int]] = [(root, 0)]
        path: list[str] = []
        positions: dict[str, int] = {}
        while stack:
            node, parent_index = stack[-1]
            if parent_index == 0 and state.get(node, 0) == 0:
                state[node] = 1
                positions[node] = len(path)
                path.append(node)
            parents = artifacts[node].derived_from
            if parent_index < len(parents):
                parent = parents[parent_index]
                stack[-1] = (node, parent_index + 1)
                if parent not in artifacts:
                    continue
                parent_state = state.get(parent, 0)
                if parent_state == 1:
                    start = positions[parent]
                    return tuple(path[start:] + [parent])
                if parent_state == 0:
                    stack.append((parent, 0))
                continue
            stack.pop()
            state[node] = 2
            positions.pop(node, None)
            path.pop()
    return ()


def validate_evidence_manifest(
    profile: ClaimAuthorizationProfile,
    evidence_blocks: dict[str, EvidenceBlock],
    manifest: EvidencePackageManifest,
) -> tuple[IntegrityDeficit, ...]:
    """Return every detected integrity deficit without priority masking."""

    artifacts = manifest.artifact_index()
    block_artifacts = manifest.block_index()
    deficits: list[IntegrityDeficit] = []
    if (
        manifest.profile_id != profile.profile_id
        or manifest.profile_version != profile.version
        or manifest.profile_hash != profile.source_hash
    ):
        deficits.append(
            IntegrityDeficit(
                IntegrityDeficitCode.PROFILE_IDENTITY_MISMATCH,
                (manifest.profile_id, manifest.profile_version, manifest.profile_hash),
                "manifest profile identity does not match the executable profile",
            )
        )
    invalid_hashes = tuple(
        artifact.artifact_id
        for artifact in manifest.artifacts
        if not SHA256_PATTERN.fullmatch(artifact.declared_sha256)
        or artifact.observed_sha256 != artifact.declared_sha256
    )
    if invalid_hashes:
        deficits.append(
            IntegrityDeficit(
                IntegrityDeficitCode.ARTIFACT_HASH_MISMATCH,
                invalid_hashes,
                "declared and observed content hashes must be identical SHA-256 values",
            )
        )
    unknown = sorted(
        {
            parent
            for artifact in manifest.artifacts
            for parent in artifact.derived_from
            if parent not in artifacts
        }
        | {
            artifact_id
            for sources in block_artifacts.values()
            for artifact_id in sources
            if artifact_id not in artifacts
        }
    )
    if unknown:
        deficits.append(
            IntegrityDeficit(
                IntegrityDeficitCode.UNKNOWN_PROVENANCE_REFERENCE,
                tuple(unknown),
                "every provenance and block-lineage reference must resolve",
            )
        )
    cycle = _has_cycle(artifacts)
    if cycle:
        deficits.append(
            IntegrityDeficit(
                IntegrityDeficitCode.PROVENANCE_CYCLE,
                cycle,
                "artifact derivation must be acyclic",
            )
        )
    duplicate_pairs = []
    for left, right in manifest.independent_artifact_pairs:
        if left in artifacts and right in artifacts:
            first, second = artifacts[left], artifacts[right]
            if (
                first.declared_sha256 == second.declared_sha256
                or (
                    first.study_id
                    and first.study_id == second.study_id
                    and first.data_generation_id
                    and first.data_generation_id == second.data_generation_id
                )
            ):
                duplicate_pairs.append(f"{left}|{right}")
    if duplicate_pairs:
        deficits.append(
            IntegrityDeficit(
                IntegrityDeficitCode.DUPLICATE_INDEPENDENT_ARTIFACT,
                tuple(duplicate_pairs),
                "artifacts declared independent share content or a data-generation identity",
            )
        )
    overlapping_pairs = []
    for left, right in manifest.disjoint_cohort_pairs:
        if left in artifacts and right in artifacts:
            overlap = set(artifacts[left].cohorts) & set(artifacts[right].cohorts)
            if overlap:
                overlapping_pairs.append(f"{left}|{right}:{','.join(sorted(overlap))}")
    if overlapping_pairs:
        deficits.append(
            IntegrityDeficit(
                IntegrityDeficitCode.OVERLAPPING_INDEPENDENT_COHORTS,
                tuple(overlapping_pairs),
                "cohorts declared disjoint contain shared independent units",
            )
        )
    statuses: dict[str, set[EvidenceStatus]] = defaultdict(set)
    for attestation in manifest.attestations:
        statuses[attestation.block_name].add(attestation.status)
    contradictory = tuple(
        sorted(block_name for block_name, values in statuses.items() if len(values) > 1)
    )
    if contradictory:
        deficits.append(
            IntegrityDeficit(
                IntegrityDeficitCode.CONTRADICTORY_ATTESTATION,
                contradictory,
                "the package contains incompatible statuses for the same evidence block",
            )
        )
    missing_lineage = tuple(
        sorted(
            name
            for name in evidence_blocks
            if name not in block_artifacts or not block_artifacts[name]
        )
    )
    if missing_lineage:
        deficits.append(
            IntegrityDeficit(
                IntegrityDeficitCode.MISSING_BLOCK_LINEAGE,
                missing_lineage,
                "every supplied evidence block must identify at least one source artifact",
            )
        )
    return tuple(sorted(deficits, key=lambda row: row.code.value))


class IntegrityAwareAuthorizationSystem:
    """Compose profile authorization with non-compensatory manifest integrity."""

    def __init__(
        self,
        profile: ClaimAuthorizationProfile,
        evidence_blocks: dict[str, EvidenceBlock],
        manifest: EvidencePackageManifest,
    ) -> None:
        self.profile = profile
        self.evidence_blocks = evidence_blocks
        self.manifest = manifest

    def infer(self, claim: str) -> IntegrityAwareDecision:
        core = ClaimAuthorizationSystem(self.profile, self.evidence_blocks).infer(claim)
        integrity = validate_evidence_manifest(
            self.profile, self.evidence_blocks, self.manifest
        )
        return IntegrityAwareDecision(core=core, integrity_deficits=integrity)
