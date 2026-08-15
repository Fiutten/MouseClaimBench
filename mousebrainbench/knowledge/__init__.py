"""Knowledge representation and inference for scientific claim authorization."""

from mousebrainbench.knowledge.asp_engine import AspDecision, infer_with_clingo
from mousebrainbench.knowledge.authorization import (
    AuthorizationRequirement,
    ClaimAuthorizationProfile,
    ClaimAuthorizationSystem,
    EvaluatedEvidenceFact,
    EvidenceBlockSpecification,
    ProfileAuthorizationDecision,
    ProfileAuthorizationStatus,
    load_authorization_profile_v2,
    load_authorization_profile_v2_basis,
)
from mousebrainbench.knowledge.authorization_asp import (
    AspProfileAuthorizationDecision,
    authorize_with_clingo_v2,
)
from mousebrainbench.knowledge.engine import (
    ClaimKnowledgeSystem,
    InferenceStep,
    KnowledgeInference,
)
from mousebrainbench.knowledge.final_authorization import (
    FinalAuthorizationDecision,
    FinalAuthorizationSystem,
    compose_final_authorization,
)
from mousebrainbench.knowledge.integrity import (
    ArtifactRecord,
    DomainIntegrityAuthorizationSystem,
    EvidenceAttestation,
    EvidencePackageManifest,
    IntegrityAwareDecision,
    IntegrityDeficit,
    IntegrityDeficitCode,
    validate_evidence_manifest,
)
from mousebrainbench.knowledge.integrity import (
    IntegrityAwareAuthorizationSystem as IntegrityAwareAuthorizationSystem,
)
from mousebrainbench.knowledge.profile import (
    InferenceRule,
    KnowledgeProfile,
    load_default_profile,
    load_default_profile_basis,
)
from mousebrainbench.knowledge.standards import (
    ShaclAuthorizationDecision,
    StructuralConformanceDecision,
    StructuralDeficit,
    StructuralDeficitCode,
    authorize_with_shacl_v2,
    evidence_package_to_rdf,
    profile_to_rdf,
    shacl_shapes_for_claim,
    validate_structure_with_shacl_v2,
)

__all__ = [
    "ArtifactRecord",
    "AspDecision",
    "AspProfileAuthorizationDecision",
    "AuthorizationRequirement",
    "ClaimAuthorizationProfile",
    "ClaimAuthorizationSystem",
    "ClaimKnowledgeSystem",
    "DomainIntegrityAuthorizationSystem",
    "EvaluatedEvidenceFact",
    "EvidenceAttestation",
    "EvidenceBlockSpecification",
    "EvidencePackageManifest",
    "FinalAuthorizationDecision",
    "FinalAuthorizationSystem",
    "InferenceRule",
    "InferenceStep",
    "IntegrityAwareDecision",
    "IntegrityDeficit",
    "IntegrityDeficitCode",
    "KnowledgeInference",
    "KnowledgeProfile",
    "ProfileAuthorizationDecision",
    "ProfileAuthorizationStatus",
    "ShaclAuthorizationDecision",
    "StructuralConformanceDecision",
    "StructuralDeficit",
    "StructuralDeficitCode",
    "authorize_with_clingo_v2",
    "authorize_with_shacl_v2",
    "compose_final_authorization",
    "evidence_package_to_rdf",
    "infer_with_clingo",
    "load_authorization_profile_v2",
    "load_authorization_profile_v2_basis",
    "load_default_profile",
    "load_default_profile_basis",
    "profile_to_rdf",
    "shacl_shapes_for_claim",
    "validate_evidence_manifest",
    "validate_structure_with_shacl_v2",
]
