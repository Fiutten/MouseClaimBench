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
from mousebrainbench.knowledge.integrity import (
    ArtifactRecord,
    EvidenceAttestation,
    EvidencePackageManifest,
    IntegrityAwareAuthorizationSystem,
    IntegrityAwareDecision,
    IntegrityDeficit,
    IntegrityDeficitCode,
    validate_evidence_manifest,
)
from mousebrainbench.knowledge.profile import (
    InferenceRule,
    KnowledgeProfile,
    load_default_profile,
    load_default_profile_basis,
)
from mousebrainbench.knowledge.standards import (
    ShaclAuthorizationDecision,
    authorize_with_shacl_v2,
    evidence_package_to_rdf,
    profile_to_rdf,
    shacl_shapes_for_claim,
)

__all__ = [
    "ArtifactRecord",
    "AspDecision",
    "AspProfileAuthorizationDecision",
    "AuthorizationRequirement",
    "ClaimAuthorizationProfile",
    "ClaimAuthorizationSystem",
    "ClaimKnowledgeSystem",
    "EvaluatedEvidenceFact",
    "EvidenceAttestation",
    "EvidenceBlockSpecification",
    "EvidencePackageManifest",
    "InferenceRule",
    "InferenceStep",
    "IntegrityAwareAuthorizationSystem",
    "IntegrityAwareDecision",
    "IntegrityDeficit",
    "IntegrityDeficitCode",
    "KnowledgeInference",
    "KnowledgeProfile",
    "ProfileAuthorizationDecision",
    "ProfileAuthorizationStatus",
    "ShaclAuthorizationDecision",
    "authorize_with_clingo_v2",
    "authorize_with_shacl_v2",
    "evidence_package_to_rdf",
    "infer_with_clingo",
    "load_authorization_profile_v2",
    "load_authorization_profile_v2_basis",
    "load_default_profile",
    "load_default_profile_basis",
    "profile_to_rdf",
    "shacl_shapes_for_claim",
    "validate_evidence_manifest",
]
