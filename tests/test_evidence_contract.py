import pytest

from mousebrainbench.validation.evidence_contract import (
    DecisionStatus,
    EvidenceBlock,
    EvidenceContractEvaluator,
    EvidenceStatus,
    blocks_by_name,
)


def _block(name: str, status: EvidenceStatus) -> EvidenceBlock:
    return EvidenceBlock.from_mapping(
        name=name,
        status=status,
        source="test-artifact.json",
        rule="test rule",
        rationale="test rationale",
        observations={"raw_value": 3.7},
    )


def test_contract_preserves_original_observations_without_normalization() -> None:
    block = _block("prediction", EvidenceStatus.PASSED)

    assert block.as_dict()["observations"] == {"raw_value": 3.7}


def test_failed_block_decisively_blocks_mechanistic_claim() -> None:
    blocks = {
        "prediction": _block("prediction", EvidenceStatus.PASSED),
        "internal_reproduction": _block("internal_reproduction", EvidenceStatus.PASSED),
        "topology_specificity": _block("topology_specificity", EvidenceStatus.FAILED),
        "directed_identifiability": _block(
            "directed_identifiability", EvidenceStatus.UNKNOWN
        ),
    }

    decision = EvidenceContractEvaluator().evaluate_claim("mechanistic", blocks)

    assert decision.status is DecisionStatus.BLOCKED


def test_missing_evidence_is_uncertain_not_failed() -> None:
    decision = EvidenceContractEvaluator().evaluate_claim("predictive", {})

    assert decision.status is DecisionStatus.UNCERTAIN
    assert dict(decision.block_statuses)["prediction"] is EvidenceStatus.UNKNOWN


def test_not_applicable_protocol_produces_out_of_scope_decision() -> None:
    blocks = {
        "causal_intervention": _block(
            "causal_intervention", EvidenceStatus.NOT_APPLICABLE
        )
    }

    decision = EvidenceContractEvaluator().evaluate_claim("causal", blocks)

    assert decision.status is DecisionStatus.OUT_OF_SCOPE


def test_review_requirement_is_not_automatically_authorized() -> None:
    blocks = {
        "external_replication": _block(
            "external_replication", EvidenceStatus.REQUIRES_REVIEW
        )
    }

    decision = EvidenceContractEvaluator().evaluate_claim("externally_replicated", blocks)

    assert decision.status is DecisionStatus.NEEDS_EXTERNAL_REVIEW


def test_digital_twin_contract_requires_every_non_compensatory_block() -> None:
    names = (
        "prediction",
        "internal_reproduction",
        "topology_specificity",
        "directed_identifiability",
        "causal_intervention",
        "whole_brain_coverage",
        "independent_validation",
        "reproducible_compute",
    )
    blocks = {name: _block(name, EvidenceStatus.PASSED) for name in names}
    blocks["whole_brain_coverage"] = _block("whole_brain_coverage", EvidenceStatus.FAILED)

    decision = EvidenceContractEvaluator().evaluate_claim("digital_twin", blocks)

    assert decision.status is DecisionStatus.BLOCKED


def test_duplicate_evidence_blocks_are_rejected() -> None:
    block = _block("prediction", EvidenceStatus.PASSED)

    with pytest.raises(ValueError, match="duplicate evidence block"):
        blocks_by_name((block, block))
