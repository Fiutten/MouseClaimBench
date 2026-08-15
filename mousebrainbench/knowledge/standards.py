"""PROV-O export and SHACL conformance for profile-v2 evidence packages.

The standards layer deliberately reuses W3C vocabularies instead of presenting
schema validation or provenance representation as MouseClaimBench inventions.
SHACL checks the structural evidence-package contract. The domain meaning of the
claims and the statistical evidence predicates remain outside SHACL itself.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from typing import Any
from urllib.parse import quote

from mousebrainbench.knowledge.authorization import (
    ClaimAuthorizationProfile,
)
from mousebrainbench.validation.evidence_contract import EvidenceBlock, EvidenceStatus

MCB_NAMESPACE = "https://w3id.org/mouseclaimbench/"


def _rdf_modules():
    try:
        from rdflib import BNode, Graph, Literal, Namespace, URIRef
        from rdflib.namespace import PROV, RDF, SH, XSD
    except ImportError as exc:
        raise RuntimeError(
            "standards interoperability requires the `standards-validation` dependencies"
        ) from exc
    return BNode, Graph, Literal, Namespace, URIRef, PROV, RDF, SH, XSD


def _pyshacl_validate():
    try:
        from pyshacl import validate
    except ImportError as exc:
        raise RuntimeError(
            "SHACL structural validation requires the `standards-validation` dependencies"
        ) from exc
    return validate


def _identifier(kind: str, value: str):
    URIRef = _rdf_modules()[4]
    return URIRef(f"{MCB_NAMESPACE}{kind}/{quote(value, safe='')}")


def _present(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, (tuple, list, dict, set)):
        return bool(value)
    return True


def profile_to_rdf(profile: ClaimAuthorizationProfile):
    """Represent one executable profile as an RDF graph using PROV-O lineage."""

    _, Graph, Literal, Namespace, _, PROV, RDF, _, _ = _rdf_modules()
    mcb = Namespace(MCB_NAMESPACE)
    graph = Graph()
    graph.bind("mcb", mcb)
    graph.bind("prov", PROV)
    profile_node = _identifier("profile", f"{profile.profile_id}-{profile.version}")
    graph.add((profile_node, RDF.type, mcb.AuthorizationProfile))
    graph.add((profile_node, RDF.type, PROV.Entity))
    graph.add((profile_node, mcb.profileId, Literal(profile.profile_id)))
    graph.add((profile_node, mcb.profileVersion, Literal(profile.version)))
    graph.add((profile_node, mcb.profileHash, Literal(profile.source_hash)))
    graph.add((profile_node, mcb.domain, Literal(profile.domain)))
    for specification in profile.evidence_blocks:
        block_node = _identifier("evidence-block", specification.name)
        graph.add((block_node, RDF.type, mcb.EvidenceBlockSpecification))
        graph.add((block_node, mcb.blockName, Literal(specification.name)))
        graph.add((block_node, mcb.meaning, Literal(specification.meaning)))
        graph.add((block_node, PROV.wasDerivedFrom, profile_node))
        for field in specification.required_observations_when_passed:
            graph.add((block_node, mcb.requiresObservation, Literal(field)))
    for requirement in profile.requirements:
        claim_node = _identifier("claim", requirement.claim)
        graph.add((claim_node, RDF.type, mcb.ClaimType))
        graph.add((claim_node, mcb.claimId, Literal(requirement.claim)))
        graph.add((claim_node, mcb.meaning, Literal(requirement.meaning)))
        graph.add((profile_node, mcb.declaresClaim, claim_node))
        for block_name in requirement.required_blocks:
            graph.add((claim_node, mcb.requiresEvidence, _identifier("evidence-block", block_name)))
    return graph


def evidence_package_to_rdf(
    profile: ClaimAuthorizationProfile,
    claim: str,
    evidence_blocks: Mapping[str, EvidenceBlock],
    *,
    package_id: str,
    manifest: Any | None = None,
):
    """Serialize one claim, its facts, and optional manifest without inference."""

    _, Graph, Literal, Namespace, _, PROV, RDF, _, _ = _rdf_modules()
    mcb = Namespace(MCB_NAMESPACE)
    graph = Graph()
    graph.bind("mcb", mcb)
    graph.bind("prov", PROV)
    package = _identifier("package", package_id)
    profile_node = _identifier("profile", f"{profile.profile_id}-{profile.version}")
    graph.add((package, RDF.type, mcb.EvidencePackage))
    graph.add((package, RDF.type, PROV.Entity))
    graph.add((package, mcb.packageId, Literal(package_id)))
    graph.add((package, mcb.claimId, Literal(claim)))
    graph.add(
        (
            package,
            mcb.profileId,
            Literal(manifest.profile_id if manifest is not None else profile.profile_id),
        )
    )
    graph.add(
        (
            package,
            mcb.profileVersion,
            Literal(manifest.profile_version if manifest is not None else profile.version),
        )
    )
    graph.add(
        (
            package,
            mcb.profileHash,
            Literal(manifest.profile_hash if manifest is not None else profile.source_hash),
        )
    )
    graph.add((package, PROV.wasDerivedFrom, profile_node))
    for block_name, block in evidence_blocks.items():
        fact = _identifier("fact", f"{package_id}-{block_name}")
        source_digest = hashlib.sha256(block.source.encode()).hexdigest()
        source = _identifier("source", source_digest)
        graph.add((package, mcb.hasEvidence, fact))
        graph.add((fact, RDF.type, mcb.EvidenceFact))
        graph.add((fact, RDF.type, PROV.Entity))
        graph.add((fact, mcb.blockName, Literal(block_name)))
        graph.add((fact, mcb.declaredStatus, Literal(block.status.value)))
        graph.add((fact, PROV.wasDerivedFrom, source))
        graph.add((source, RDF.type, PROV.Entity))
        if _present(block.source):
            graph.add((fact, mcb.source, Literal(block.source)))
        if _present(block.rule):
            graph.add((fact, mcb.rule, Literal(block.rule)))
        if _present(block.rationale):
            graph.add((fact, mcb.rationale, Literal(block.rationale)))
        for field, value in block.observations:
            if _present(value):
                predicate = _identifier("observation", field)
                encoded = json.dumps(value, sort_keys=True, ensure_ascii=True)
                graph.add((fact, predicate, Literal(encoded, datatype=RDF.JSON)))
    if manifest is not None:
        for artifact in manifest.artifacts:
            artifact_node = _identifier("artifact", artifact.artifact_id)
            graph.add((package, mcb.hasArtifact, artifact_node))
            graph.add((artifact_node, RDF.type, mcb.Artifact))
            graph.add((artifact_node, RDF.type, PROV.Entity))
            graph.add((artifact_node, mcb.artifactId, Literal(artifact.artifact_id)))
            graph.add((artifact_node, mcb.declaredSha256, Literal(artifact.declared_sha256)))
            graph.add((artifact_node, mcb.observedSha256, Literal(artifact.observed_sha256)))
            for parent in artifact.derived_from:
                graph.add((artifact_node, PROV.wasDerivedFrom, _identifier("artifact", parent)))
            for cohort in artifact.cohorts:
                graph.add((artifact_node, mcb.cohort, Literal(cohort)))
            if artifact.study_id:
                graph.add((artifact_node, mcb.studyId, Literal(artifact.study_id)))
            if artifact.data_generation_id:
                graph.add(
                    (artifact_node, mcb.dataGenerationId, Literal(artifact.data_generation_id))
                )
        for block_name, artifact_ids in manifest.block_artifacts:
            fact = _identifier("fact", f"{package_id}-{block_name}")
            for artifact_id in artifact_ids:
                graph.add((fact, PROV.wasDerivedFrom, _identifier("artifact", artifact_id)))
        for index, attestation in enumerate(manifest.attestations):
            attestation_node = _identifier(
                "attestation", f"{package_id}-{index}-{attestation.block_name}"
            )
            graph.add((package, mcb.hasAttestation, attestation_node))
            graph.add((attestation_node, RDF.type, mcb.EvidenceAttestation))
            graph.add((attestation_node, mcb.blockName, Literal(attestation.block_name)))
            graph.add((attestation_node, mcb.declaredStatus, Literal(attestation.status.value)))
            graph.add(
                (
                    attestation_node,
                    PROV.wasDerivedFrom,
                    _identifier("artifact", attestation.artifact_id),
                )
            )
        for relation, pairs in (
            (mcb.declaredIndependentFrom, manifest.independent_artifact_pairs),
            (mcb.declaredCohortDisjointFrom, manifest.disjoint_cohort_pairs),
        ):
            for left, right in pairs:
                graph.add((_identifier("artifact", left), relation, _identifier("artifact", right)))
    return graph, package


def shacl_shapes_for_claim(
    profile: ClaimAuthorizationProfile,
    claim: str,
    *,
    package_node,
):
    """Compile structural constraints without authorizing scientific status."""

    BNode, Graph, Literal, Namespace, _, _, RDF, SH, _ = _rdf_modules()
    requirement = profile.requirement(claim)
    if requirement is None:
        raise KeyError(f"claim is outside profile: {claim}")
    mcb = Namespace(MCB_NAMESPACE)
    shapes = Graph()
    shapes.bind("mcb", mcb)
    shapes.bind("sh", SH)
    def add_constraint(
        identifier: str,
        code: str,
        message: str,
        query: str,
        *,
        block_name: str | None = None,
    ) -> None:
        shape = _identifier("shape", f"{claim}-{identifier}")
        constraint = BNode()
        shapes.add((shape, RDF.type, SH.NodeShape))
        shapes.add((shape, SH.targetNode, package_node))
        shapes.add((shape, SH.sparql, constraint))
        shapes.add((shape, mcb.deficitCode, Literal(code)))
        if block_name is not None:
            shapes.add((shape, mcb.blockName, Literal(block_name)))
        shapes.add((constraint, SH.message, Literal(message)))
        shapes.add((constraint, SH.select, Literal(query)))

    package_fields = (
        ("packageId", None),
        ("claimId", claim),
        ("profileId", profile.profile_id),
        ("profileVersion", profile.version),
        ("profileHash", profile.source_hash),
    )
    for field, expected in package_fields:
        object_pattern = "?value" if expected is None else json.dumps(expected)
        add_constraint(
            f"package-{field}",
            "missing_or_invalid_package_field",
            f"missing or invalid package field: {field}",
            "\n".join(
                (
                    "SELECT $this WHERE {",
                    "  FILTER NOT EXISTS {",
                    f"    $this <{MCB_NAMESPACE}{field}> {object_pattern} .",
                    "  }",
                    "}",
                )
            ),
        )

    allowed_statuses = ", ".join(json.dumps(status.value) for status in EvidenceStatus)
    for block_name in requirement.required_blocks:
        specification = profile.block_specification(block_name)
        block_literal = json.dumps(block_name)
        add_constraint(
            f"{block_name}-required",
            "missing_required_block",
            f"missing required evidence block: {block_name}",
            "\n".join(
                (
                    "SELECT $this WHERE {",
                    "  FILTER NOT EXISTS {",
                    f"    $this <{MCB_NAMESPACE}hasEvidence> ?fact .",
                    f"    ?fact <{MCB_NAMESPACE}blockName> {block_literal} .",
                    "  }",
                    "}",
                )
            ),
            block_name=block_name,
        )
        add_constraint(
            f"{block_name}-status",
            "invalid_or_missing_status",
            f"invalid or missing declared status: {block_name}",
            "\n".join(
                (
                    "SELECT $this WHERE {",
                    f"  $this <{MCB_NAMESPACE}hasEvidence> ?fact .",
                    f"  ?fact <{MCB_NAMESPACE}blockName> {block_literal} .",
                    "  FILTER NOT EXISTS {",
                    f"    ?fact <{MCB_NAMESPACE}declaredStatus> ?status .",
                    f"    FILTER(?status IN ({allowed_statuses}))",
                    "  }",
                    "}",
                )
            ),
            block_name=block_name,
        )
        add_constraint(
            f"{block_name}-metadata",
            "missing_fact_metadata",
            f"missing source, rule, or rationale: {block_name}",
            "\n".join(
                (
                    "SELECT $this WHERE {",
                    f"  $this <{MCB_NAMESPACE}hasEvidence> ?fact .",
                    f"  ?fact <{MCB_NAMESPACE}blockName> {block_literal} .",
                    "  FILTER NOT EXISTS {",
                    f"    ?fact <{MCB_NAMESPACE}source> ?source .",
                    f"    ?fact <{MCB_NAMESPACE}rule> ?rule .",
                    f"    ?fact <{MCB_NAMESPACE}rationale> ?rationale .",
                    "  }",
                    "}",
                )
            ),
            block_name=block_name,
        )
        for field in specification.required_observations_when_passed:
            add_constraint(
                f"{block_name}-observation-{field}",
                "missing_required_observation",
                f"missing observation required for passed block: {block_name}.{field}",
                "\n".join(
                    (
                        "SELECT $this WHERE {",
                        f"  $this <{MCB_NAMESPACE}hasEvidence> ?fact .",
                        f"  ?fact <{MCB_NAMESPACE}blockName> {block_literal} .",
                        f"  ?fact <{MCB_NAMESPACE}declaredStatus> \"passed\" .",
                        "  FILTER NOT EXISTS {",
                        f"    ?fact <{MCB_NAMESPACE}observation/{quote(field, safe='')}> ?value .",
                        "  }",
                        "}",
                    )
                ),
                block_name=block_name,
            )
    return shapes


class StructuralDeficitCode(str, Enum):
    """Structural graph failures reported independently of domain status."""

    MISSING_OR_INVALID_PACKAGE_FIELD = "missing_or_invalid_package_field"
    MISSING_REQUIRED_BLOCK = "missing_required_block"
    INVALID_OR_MISSING_STATUS = "invalid_or_missing_status"
    MISSING_FACT_METADATA = "missing_fact_metadata"
    MISSING_REQUIRED_OBSERVATION = "missing_required_observation"


@dataclass(frozen=True)
class StructuralDeficit:
    """One SHACL structural violation and its optional evidence-block witness."""

    code: StructuralDeficitCode
    witness: str
    message: str

    def as_dict(self) -> dict[str, str]:
        return {
            "code": self.code.value,
            "witness": self.witness,
            "message": self.message,
        }


@dataclass(frozen=True)
class StructuralConformanceDecision:
    """Result of external SHACL validation of the graph contract only."""

    claim: str
    conforms: bool
    deficits: tuple[StructuralDeficit, ...]
    backend: str
    report_text: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "claim": self.claim,
            "conforms": self.conforms,
            "deficits": [row.as_dict() for row in self.deficits],
            "backend": self.backend,
            "report_text": self.report_text,
        }


# Backward-compatible type name. The object now represents structural
# conformance and deliberately exposes no scientific `authorized` property.
ShaclAuthorizationDecision = StructuralConformanceDecision


def validate_structure_with_shacl_v2(
    profile: ClaimAuthorizationProfile,
    claim: str,
    evidence_blocks: Mapping[str, EvidenceBlock],
    *,
    package_id: str = "authorization-case",
    manifest: Any | None = None,
) -> StructuralConformanceDecision:
    """Validate one claim package structurally without evaluating domain status."""

    requirement = profile.requirement(claim)
    if requirement is None:
        return StructuralConformanceDecision(
            claim=claim,
            conforms=False,
            deficits=(
                StructuralDeficit(
                    StructuralDeficitCode.MISSING_OR_INVALID_PACKAGE_FIELD,
                    "claimId",
                    "claim is outside the profile",
                ),
            ),
            backend="pyshacl",
            report_text="claim is outside the profile",
        )
    _, _, _, Namespace, _, _, RDF, SH, _ = _rdf_modules()
    mcb = Namespace(MCB_NAMESPACE)
    data_graph, package = evidence_package_to_rdf(
        profile,
        claim,
        evidence_blocks,
        package_id=package_id,
        manifest=manifest,
    )
    shapes = shacl_shapes_for_claim(profile, claim, package_node=package)
    conforms, report_graph, report_text = _pyshacl_validate()(
        data_graph,
        shacl_graph=shapes,
        inference="none",
        abort_on_first=False,
        allow_infos=False,
        allow_warnings=False,
        meta_shacl=False,
        advanced=True,
    )
    structural_deficits: set[StructuralDeficit] = set()
    for result in report_graph.subjects(RDF.type, SH.ValidationResult):
        source_shape = report_graph.value(result, SH.sourceShape)
        deficit_code = shapes.value(source_shape, mcb.deficitCode)
        block_name = shapes.value(source_shape, mcb.blockName)
        message = report_graph.value(result, SH.resultMessage)
        if deficit_code is None:
            raise RuntimeError("SHACL result cannot be mapped to a structural deficit")
        witness = str(block_name) if block_name is not None else "package"
        structural_deficits.add(
            StructuralDeficit(
                StructuralDeficitCode(str(deficit_code)),
                witness,
                str(message or deficit_code),
            )
        )
    deficits = tuple(sorted(structural_deficits, key=lambda row: (row.code.value, row.witness)))
    if bool(conforms) != (not deficits):
        raise RuntimeError("SHACL conformance and parsed deficit set disagree")
    return StructuralConformanceDecision(
        claim=claim,
        deficits=deficits,
        conforms=bool(conforms),
        backend="pyshacl",
        report_text=str(report_text),
    )


def authorize_with_shacl_v2(
    profile: ClaimAuthorizationProfile,
    claim: str,
    evidence_blocks: Mapping[str, EvidenceBlock],
    *,
    package_id: str = "authorization-case",
    manifest: Any | None = None,
) -> StructuralConformanceDecision:
    """Compatibility wrapper for the former, misleading authorization name."""

    return validate_structure_with_shacl_v2(
        profile,
        claim,
        evidence_blocks,
        package_id=package_id,
        manifest=manifest,
    )
