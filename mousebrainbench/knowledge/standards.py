"""PROV-O export and SHACL conformance for profile-v2 evidence packages.

The standards layer deliberately reuses W3C vocabularies instead of presenting
schema validation or provenance representation as MouseClaimBench inventions.
SHACL checks the structural authorization contract. The domain meaning of the
claims and the statistical evidence predicates remain outside SHACL itself.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any
from urllib.parse import quote

from mousebrainbench.knowledge.authorization import (
    ClaimAuthorizationProfile,
    ProfileAuthorizationStatus,
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
            "SHACL authorization requires the `standards-validation` dependencies"
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
):
    """Serialize declared facts without using the Python authorization decision."""

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
    graph.add((package, mcb.profileHash, Literal(profile.source_hash)))
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
    return graph, package


def shacl_shapes_for_claim(
    profile: ClaimAuthorizationProfile,
    claim: str,
    *,
    package_node,
):
    """Compile one profile requirement into SHACL-SPARQL constraints."""

    BNode, Graph, Literal, Namespace, _, _, RDF, SH, _ = _rdf_modules()
    requirement = profile.requirement(claim)
    if requirement is None:
        raise KeyError(f"claim is outside profile: {claim}")
    mcb = Namespace(MCB_NAMESPACE)
    shapes = Graph()
    shapes.bind("mcb", mcb)
    shapes.bind("sh", SH)
    for block_name in requirement.required_blocks:
        specification = profile.block_specification(block_name)
        shape = _identifier("shape", f"{claim}-{block_name}")
        constraint = BNode()
        required_patterns = [
            f'?fact <{MCB_NAMESPACE}blockName> {json.dumps(block_name)} .',
            f'?fact <{MCB_NAMESPACE}declaredStatus> "passed" .',
            f'?fact <{MCB_NAMESPACE}source> ?source .',
            f'?fact <{MCB_NAMESPACE}rule> ?rule .',
            f'?fact <{MCB_NAMESPACE}rationale> ?rationale .',
        ]
        required_patterns.extend(
            f"?fact <{MCB_NAMESPACE}observation/{quote(field, safe='')}> ?obs{index} ."
            for index, field in enumerate(
                specification.required_observations_when_passed
            )
        )
        query = "\n".join(
            (
                "SELECT $this WHERE {",
                "  FILTER NOT EXISTS {",
                f"    $this <{MCB_NAMESPACE}hasEvidence> ?fact .",
                *(f"    {pattern}" for pattern in required_patterns),
                "  }",
                "}",
            )
        )
        shapes.add((shape, RDF.type, SH.NodeShape))
        shapes.add((shape, SH.targetNode, package_node))
        shapes.add((shape, SH.sparql, constraint))
        shapes.add((shape, mcb.blockName, Literal(block_name)))
        shapes.add((constraint, SH.message, Literal(f"inadmissible required block: {block_name}")))
        shapes.add((constraint, SH.select, Literal(query)))
    return shapes


def _deficit_status(
    profile: ClaimAuthorizationProfile,
    name: str,
    evidence_blocks: Mapping[str, EvidenceBlock],
) -> EvidenceStatus:
    block = evidence_blocks.get(name)
    if block is None:
        return EvidenceStatus.UNKNOWN
    if block.status is not EvidenceStatus.PASSED:
        return block.status
    observations = dict(block.observations)
    specification = profile.block_specification(name)
    complete = all(
        field in observations and _present(observations[field])
        for field in specification.required_observations_when_passed
    )
    complete = complete and all(
        _present(value) for value in (block.source, block.rule, block.rationale)
    )
    return EvidenceStatus.PASSED if complete else EvidenceStatus.REQUIRES_REVIEW


@dataclass(frozen=True)
class ShaclAuthorizationDecision:
    """Profile disposition returned by a standards-compliant SHACL processor."""

    claim: str
    status: ProfileAuthorizationStatus
    deficits: tuple[tuple[str, EvidenceStatus], ...]
    conforms: bool
    backend: str
    report_text: str

    @property
    def authorized(self) -> bool:
        return self.status is ProfileAuthorizationStatus.AUTHORIZED


def authorize_with_shacl_v2(
    profile: ClaimAuthorizationProfile,
    claim: str,
    evidence_blocks: Mapping[str, EvidenceBlock],
    *,
    package_id: str = "authorization-case",
) -> ShaclAuthorizationDecision:
    """Validate one claim package with pySHACL and return all violating blocks."""

    requirement = profile.requirement(claim)
    if requirement is None:
        return ShaclAuthorizationDecision(
            claim=claim,
            status=ProfileAuthorizationStatus.OUTSIDE_PROFILE,
            deficits=(),
            conforms=False,
            backend="pyshacl",
            report_text="claim is outside the profile",
        )
    _, _, _, Namespace, _, _, RDF, SH, _ = _rdf_modules()
    mcb = Namespace(MCB_NAMESPACE)
    data_graph, package = evidence_package_to_rdf(
        profile, claim, evidence_blocks, package_id=package_id
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
    deficit_names: set[str] = set()
    for result in report_graph.subjects(RDF.type, SH.ValidationResult):
        source_shape = report_graph.value(result, SH.sourceShape)
        block_name = shapes.value(source_shape, mcb.blockName)
        if block_name is None:
            raise RuntimeError("SHACL result cannot be mapped to an evidence block")
        deficit_names.add(str(block_name))
    deficits = tuple(
        sorted(
            (
                (name, _deficit_status(profile, name, evidence_blocks))
                for name in deficit_names
            ),
            key=lambda item: item[0],
        )
    )
    if bool(conforms) != (not deficits):
        raise RuntimeError("SHACL conformance and parsed deficit set disagree")
    return ShaclAuthorizationDecision(
        claim=claim,
        status=(
            ProfileAuthorizationStatus.AUTHORIZED
            if conforms
            else ProfileAuthorizationStatus.NOT_AUTHORIZED
        ),
        deficits=deficits,
        conforms=bool(conforms),
        backend="pyshacl",
        report_text=str(report_text),
    )
