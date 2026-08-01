from itertools import product

import pytest

from mousebrainbench.knowledge import ClaimKnowledgeSystem, KnowledgeProfile, load_default_profile
from mousebrainbench.validation.evidence_contract import (
    CLAIM_REQUIREMENTS_V3,
    DecisionStatus,
    EvidenceBlock,
    EvidenceContractEvaluator,
    EvidenceStatus,
)


def _block(name: str, status: EvidenceStatus) -> EvidenceBlock:
    return EvidenceBlock.from_mapping(
        name=name,
        status=status,
        source="test-source.json",
        rule="declared test predicate",
        rationale="controlled test fact",
        observations={"raw_value": 0.25},
    )


def test_packaged_profile_is_versioned_complete_and_hash_addressed() -> None:
    profile = load_default_profile()

    assert profile.profile_id == "mouse_brain_claims"
    assert profile.version == "1.0.0"
    assert profile.requirements == CLAIM_REQUIREMENTS_V3
    assert profile.source_hash.startswith("sha256:")
    assert len(profile.source_hash) == len("sha256:") + 64
    assert {rule.conclusion for rule in profile.rules} == set(DecisionStatus)


def test_profile_rejects_duplicate_claims() -> None:
    profile = load_default_profile().as_dict()
    profile["claims"].append(profile["claims"][0])

    with pytest.raises(ValueError, match="duplicate claim"):
        KnowledgeProfile.from_mapping(profile)


def test_profile_rejects_ambiguous_rule_priorities() -> None:
    profile = load_default_profile().as_dict()
    profile["inference_rules"][1]["priority"] = profile["inference_rules"][0]["priority"]

    with pytest.raises(ValueError, match="priorities must be unique"):
        KnowledgeProfile.from_mapping(profile)


def test_explanation_preserves_rule_witness_and_fact_provenance() -> None:
    profile = load_default_profile()
    blocks = {
        "prediction": _block("prediction", EvidenceStatus.PASSED),
        "internal_reproduction": _block("internal_reproduction", EvidenceStatus.UNKNOWN),
        "topology_specificity": _block("topology_specificity", EvidenceStatus.FAILED),
        "directed_identifiability": _block(
            "directed_identifiability", EvidenceStatus.REQUIRES_REVIEW
        ),
    }

    inference = ClaimKnowledgeSystem(profile, blocks).infer("mechanistic")

    assert inference.decision.status is DecisionStatus.BLOCKED
    assert inference.fired_rule == "failed_block_veto"
    assert inference.steps[0].witness_blocks == ("topology_specificity",)
    assert {fact.source for fact in inference.evidence_facts} == {"test-source.json"}
    assert inference.as_dict()["evidence_facts"][0]["observations"] == {"raw_value": 0.25}


def test_missing_fact_is_explicit_unknown_with_provenance() -> None:
    inference = ClaimKnowledgeSystem(load_default_profile(), {}).infer("predictive")

    assert inference.decision.status is DecisionStatus.UNCERTAIN
    assert inference.fired_rule == "missing_evidence_uncertainty"
    assert inference.evidence_facts[0].source == "missing"


def test_undeclared_claim_stays_outside_knowledge_boundary() -> None:
    inference = ClaimKnowledgeSystem(load_default_profile(), {}).infer("conscious")

    assert inference.decision.status is DecisionStatus.OUT_OF_SCOPE
    assert inference.fired_rule == "undeclared_claim_boundary"
    assert inference.decision.required_blocks == ()


def test_mechanistic_rule_precedence_is_total_over_all_625_state_assignments() -> None:
    profile = load_default_profile()
    names = (
        "prediction",
        "internal_reproduction",
        "topology_specificity",
        "directed_identifiability",
    )
    expected_by_precedence = (
        (EvidenceStatus.FAILED, DecisionStatus.BLOCKED),
        (EvidenceStatus.REQUIRES_REVIEW, DecisionStatus.NEEDS_EXTERNAL_REVIEW),
        (EvidenceStatus.UNKNOWN, DecisionStatus.UNCERTAIN),
        (EvidenceStatus.NOT_APPLICABLE, DecisionStatus.OUT_OF_SCOPE),
    )

    for states in product(EvidenceStatus, repeat=len(names)):
        blocks = {name: _block(name, status) for name, status in zip(names, states, strict=True)}
        inference = ClaimKnowledgeSystem(profile, blocks).infer("mechanistic")
        expected = DecisionStatus.SUPPORTED
        for evidence_status, decision_status in expected_by_precedence:
            if evidence_status in states:
                expected = decision_status
                break
        assert inference.decision.status is expected


def test_knowledge_graph_exposes_claim_rule_status_and_provenance_relations() -> None:
    graph = ClaimKnowledgeSystem(load_default_profile(), {}).knowledge_graph()
    node_types = {node["type"] for node in graph["nodes"]}
    relations = {edge["relation"] for edge in graph["edges"]}

    assert {"knowledge_profile", "claim", "evidence_block", "inference_rule"} <= node_types
    assert {"evidence_status", "decision_status"} <= node_types
    assert {"declares", "requires", "tests_status", "concludes"} <= relations
    assert graph == ClaimKnowledgeSystem(load_default_profile(), {}).knowledge_graph()


def test_legacy_facade_and_knowledge_system_are_decision_equivalent() -> None:
    blocks = {"prediction": _block("prediction", EvidenceStatus.PASSED)}
    legacy = EvidenceContractEvaluator()
    system = ClaimKnowledgeSystem(load_default_profile(), blocks)

    assert legacy.evaluate_claim("predictive", blocks) == system.infer("predictive").decision
    assert legacy.explain_claim("predictive", blocks)["fired_rule"] == (
        "all_requirements_satisfied"
    )
