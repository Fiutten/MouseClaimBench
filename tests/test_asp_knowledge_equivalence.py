from itertools import product

from mousebrainbench.knowledge import (
    ClaimKnowledgeSystem,
    infer_with_clingo,
    load_default_profile,
)
from mousebrainbench.validation.evidence_contract import EvidenceBlock, EvidenceStatus


def _block(name: str, status: EvidenceStatus) -> EvidenceBlock:
    return EvidenceBlock.from_mapping(
        name=name,
        status=status,
        source="controlled-equivalence-test",
        rule="enumerated state assignment",
        rationale="tests independent executable semantics",
    )


def test_asp_and_python_are_equivalent_for_all_625_mechanistic_assignments() -> None:
    profile = load_default_profile()
    names = profile.requirement("mechanistic").required_blocks

    for states in product(EvidenceStatus, repeat=len(names)):
        blocks = {
            name: _block(name, status)
            for name, status in zip(names, states, strict=True)
        }
        python_status = ClaimKnowledgeSystem(profile, blocks).infer("mechanistic").decision.status
        asp_status = infer_with_clingo(profile, "mechanistic", blocks).status
        assert asp_status is python_status


def test_asp_and_python_are_equivalent_for_every_single_block_claim() -> None:
    profile = load_default_profile()
    claims = [item.claim for item in profile.requirements if len(item.required_blocks) == 1]

    for claim in claims:
        block_name = profile.requirement(claim).required_blocks[0]
        for status in EvidenceStatus:
            blocks = {block_name: _block(block_name, status)}
            assert infer_with_clingo(profile, claim, blocks).status is (
                ClaimKnowledgeSystem(profile, blocks).infer(claim).decision.status
            )


def test_asp_and_python_share_missing_and_undeclared_boundaries() -> None:
    profile = load_default_profile()

    assert infer_with_clingo(profile, "predictive", {}).status is (
        ClaimKnowledgeSystem(profile, {}).infer("predictive").decision.status
    )
    assert infer_with_clingo(profile, "conscious", {}).status is (
        ClaimKnowledgeSystem(profile, {}).infer("conscious").decision.status
    )
