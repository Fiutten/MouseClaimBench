"""Evidence-package integrity checks that precede profile authorization.

The validator handles relationships among artifacts. These checks complement
SHACL field validation and must not be interpreted as proof that an artifact is
scientifically correct or that a source is trustworthy.
"""

from __future__ import annotations

import re
import warnings
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
    DUPLICATE_ARTIFACT_ID = "duplicate_artifact_id"
    DUPLICATE_BLOCK_LINEAGE = "duplicate_block_lineage"
    ARTIFACT_HASH_MISMATCH = "artifact_hash_mismatch"
    UNKNOWN_PROVENANCE_REFERENCE = "unknown_provenance_reference"
    PROVENANCE_CYCLE = "provenance_cycle"
    DUPLICATE_INDEPENDENT_ARTIFACT = "duplicate_independent_artifact"
    OVERLAPPING_INDEPENDENT_COHORTS = "overlapping_independent_cohorts"
    CONTRADICTORY_ATTESTATION = "contradictory_attestation"
    ATTESTATION_BLOCK_STATUS_MISMATCH = "attestation_block_status_mismatch"
    UNKNOWN_BLOCK_REFERENCE = "unknown_block_reference"
    MISSING_BLOCK_LINEAGE = "missing_block_lineage"
    MISSING_BLOCK_ATTESTATION = "missing_block_attestation"


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

    def duplicate_artifact_ids(self) -> tuple[str, ...]:
        """Return repeated artifact identifiers in deterministic order."""

        counts: dict[str, int] = defaultdict(int)
        for artifact in self.artifacts:
            counts[artifact.artifact_id] += 1
        return tuple(sorted(name for name, count in counts.items() if count > 1))

    def duplicate_block_lineages(self) -> tuple[str, ...]:
        """Return block names with more than one lineage declaration."""

        counts: dict[str, int] = defaultdict(int)
        for block_name, _ in self.block_artifacts:
            counts[block_name] += 1
        return tuple(sorted(name for name, count in counts.items() if count > 1))

    def artifact_groups(self) -> dict[str, tuple[ArtifactRecord, ...]]:
        """Group every artifact declaration without discarding duplicates."""

        groups: dict[str, list[ArtifactRecord]] = defaultdict(list)
        for artifact in self.artifacts:
            groups[artifact.artifact_id].append(artifact)
        return {name: tuple(records) for name, records in groups.items()}

    def block_groups(self) -> dict[str, tuple[tuple[str, ...], ...]]:
        """Group every block-lineage declaration without discarding duplicates."""

        groups: dict[str, list[tuple[str, ...]]] = defaultdict(list)
        for block_name, sources in self.block_artifacts:
            groups[block_name].append(sources)
        return {name: tuple(lineages) for name, lineages in groups.items()}

    def artifact_index(self) -> dict[str, ArtifactRecord]:
        """Index an already-unique artifact collection.

        Validation uses :meth:`artifact_groups` so malformed packages retain all
        declarations. This convenience index rejects duplicates instead of
        silently applying order-dependent last-write-wins semantics.
        """

        groups = self.artifact_groups()
        duplicates = tuple(sorted(name for name, rows in groups.items() if len(rows) > 1))
        if duplicates:
            raise ValueError(f"duplicate artifact identifiers: {duplicates}")
        return {name: records[0] for name, records in groups.items()}

    def block_index(self) -> dict[str, tuple[str, ...]]:
        """Index already-consolidated block lineage without silent overwrites."""

        groups = self.block_groups()
        duplicates = tuple(sorted(name for name, rows in groups.items() if len(rows) > 1))
        if duplicates:
            raise ValueError(f"duplicate block-lineage declarations: {duplicates}")
        return {name: lineages[0] for name, lineages in groups.items()}


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
    """Core profile decision gated by category-complete integrity deficits."""

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


def _has_cycle(parents_by_artifact: dict[str, tuple[str, ...]]) -> tuple[str, ...]:
    # Iterative depth-first search avoids Python's recursion limit for evidence
    # packages containing thousands of artifacts.
    state: dict[str, int] = {}
    for root in sorted(parents_by_artifact):
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
            parents = parents_by_artifact[node]
            if parent_index < len(parents):
                parent = parents[parent_index]
                stack[-1] = (node, parent_index + 1)
                if parent not in parents_by_artifact:
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
    """Return every detected deficit category without priority masking."""

    deficits: list[IntegrityDeficit] = []
    duplicate_artifact_ids = manifest.duplicate_artifact_ids()
    if duplicate_artifact_ids:
        deficits.append(
            IntegrityDeficit(
                IntegrityDeficitCode.DUPLICATE_ARTIFACT_ID,
                duplicate_artifact_ids,
                "artifact identifiers must be unique within an evidence package",
            )
        )
    duplicate_block_lineages = manifest.duplicate_block_lineages()
    if duplicate_block_lineages:
        deficits.append(
            IntegrityDeficit(
                IntegrityDeficitCode.DUPLICATE_BLOCK_LINEAGE,
                duplicate_block_lineages,
                "each evidence block must have one consolidated lineage declaration",
            )
        )
    artifact_groups = manifest.artifact_groups()
    lineage_groups = manifest.block_groups()
    artifact_ids = set(artifact_groups)
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
    referenced_artifacts = {
        parent
        for artifact in manifest.artifacts
        for parent in artifact.derived_from
    }
    referenced_artifacts.update(
        artifact_id
        for _, sources in manifest.block_artifacts
        for artifact_id in sources
    )
    referenced_artifacts.update(
        attestation.artifact_id for attestation in manifest.attestations
    )
    referenced_artifacts.update(
        artifact_id
        for pair in manifest.independent_artifact_pairs
        for artifact_id in pair
    )
    referenced_artifacts.update(
        artifact_id
        for pair in manifest.disjoint_cohort_pairs
        for artifact_id in pair
    )
    unknown = sorted(referenced_artifacts - artifact_ids)
    if unknown:
        deficits.append(
            IntegrityDeficit(
                IntegrityDeficitCode.UNKNOWN_PROVENANCE_REFERENCE,
                tuple(unknown),
                "every provenance, attestation, independence, and cohort reference must resolve",
            )
        )
    supplied_blocks = set(evidence_blocks)
    referenced_blocks = set(lineage_groups)
    referenced_blocks.update(
        attestation.block_name for attestation in manifest.attestations
    )
    unknown_blocks = tuple(sorted(referenced_blocks - supplied_blocks))
    if unknown_blocks:
        deficits.append(
            IntegrityDeficit(
                IntegrityDeficitCode.UNKNOWN_BLOCK_REFERENCE,
                unknown_blocks,
                "block lineage and attestations must reference supplied evidence blocks",
            )
        )
    parents_by_artifact = {
        artifact_id: tuple(
            sorted(
                {
                    parent
                    for artifact in records
                    for parent in artifact.derived_from
                }
            )
        )
        for artifact_id, records in artifact_groups.items()
    }
    cycle = _has_cycle(parents_by_artifact)
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
        if left == right:
            duplicate_pairs.append(f"{left}|{right}:reflexive")
            continue
        if (
            left in artifact_groups
            and right in artifact_groups
            and any(
                first.declared_sha256 == second.declared_sha256
                or (
                    first.study_id
                    and first.data_generation_id
                    and (first.study_id, first.data_generation_id)
                    == (second.study_id, second.data_generation_id)
                )
                for first in artifact_groups[left]
                for second in artifact_groups[right]
            )
        ):
            duplicate_pairs.append(f"{left}|{right}")
    if duplicate_pairs:
        deficits.append(
            IntegrityDeficit(
                IntegrityDeficitCode.DUPLICATE_INDEPENDENT_ARTIFACT,
                tuple(duplicate_pairs),
                "independence pairs must be irreflexive and cannot share content or "
                "a composite study and data-generation identity",
            )
        )
    overlapping_pairs = []
    for left, right in manifest.disjoint_cohort_pairs:
        if left == right:
            overlapping_pairs.append(f"{left}|{right}:reflexive")
            continue
        if left in artifact_groups and right in artifact_groups:
            left_cohorts = {
                cohort for artifact in artifact_groups[left] for cohort in artifact.cohorts
            }
            right_cohorts = {
                cohort for artifact in artifact_groups[right] for cohort in artifact.cohorts
            }
            overlap = left_cohorts & right_cohorts
            if overlap:
                overlapping_pairs.append(f"{left}|{right}:{','.join(sorted(overlap))}")
    if overlapping_pairs:
        deficits.append(
            IntegrityDeficit(
                IntegrityDeficitCode.OVERLAPPING_INDEPENDENT_COHORTS,
                tuple(overlapping_pairs),
                "cohort-disjointness pairs must be irreflexive and cannot share units",
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
    mismatches = tuple(
        sorted(
            f"{attestation.block_name}:{evidence_blocks[attestation.block_name].status.value}"
            f"!={attestation.status.value}@{attestation.artifact_id}"
            for attestation in manifest.attestations
            if attestation.block_name in evidence_blocks
            and evidence_blocks[attestation.block_name].status is not attestation.status
        )
    )
    if mismatches:
        deficits.append(
            IntegrityDeficit(
                IntegrityDeficitCode.ATTESTATION_BLOCK_STATUS_MISMATCH,
                mismatches,
                "an unambiguous direct attestation must match the represented block status",
            )
        )
    missing_lineage = tuple(
        sorted(
            name
            for name in evidence_blocks
            if name not in lineage_groups
            or not any(sources for sources in lineage_groups[name])
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
    attested_blocks = {row.block_name for row in manifest.attestations}
    missing_attestations = tuple(sorted(supplied_blocks - attested_blocks))
    if missing_attestations:
        deficits.append(
            IntegrityDeficit(
                IntegrityDeficitCode.MISSING_BLOCK_ATTESTATION,
                missing_attestations,
                "every supplied evidence block must have at least one attestation",
            )
        )
    return tuple(sorted(deficits, key=lambda row: row.code.value))


class DomainIntegrityAuthorizationSystem:
    """Compose only the domain and manifest-integrity gates.

    This partial system is useful for integrity ablations. It is not the
    canonical paper-level API because it does not execute external structural
    conformance. Use ``FinalAuthorizationSystem`` for the complete S and A and
    I decision.
    """

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


class IntegrityAwareAuthorizationSystem(DomainIntegrityAuthorizationSystem):
    """Deprecated compatibility name for the domain-plus-integrity subsystem."""

    def __init__(
        self,
        profile: ClaimAuthorizationProfile,
        evidence_blocks: dict[str, EvidenceBlock],
        manifest: EvidencePackageManifest,
    ) -> None:
        warnings.warn(
            "IntegrityAwareAuthorizationSystem is a partial domain-plus-integrity "
            "subsystem; use DomainIntegrityAuthorizationSystem for ablations or "
            "FinalAuthorizationSystem for the canonical three-gate decision.",
            DeprecationWarning,
            stacklevel=2,
        )
        super().__init__(profile, evidence_blocks, manifest)
