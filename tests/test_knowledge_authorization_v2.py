from itertools import combinations

from mousebrainbench.knowledge import (
    ClaimAuthorizationSystem,
    ProfileAuthorizationStatus,
    authorize_with_clingo_v2,
    load_authorization_profile_v2,
    load_authorization_profile_v2_basis,
)
from mousebrainbench.validation.evidence_contract import EvidenceBlock, EvidenceStatus


def _complete_observations(block_name: str) -> dict[str, object]:
    profile = load_authorization_profile_v2()
    fields = profile.block_specification(block_name).required_observations_when_passed
    return {field: 0 if field == "overlap" else f"declared-{field}" for field in fields}


def _block(
    name: str,
    status: EvidenceStatus = EvidenceStatus.PASSED,
    observations: dict[str, object] | None = None,
) -> EvidenceBlock:
    return EvidenceBlock.from_mapping(
        name=name,
        status=status,
        source="frozen-controlled-case.json",
        rule="prespecified block predicate",
        rationale="controlled profile-v2 test",
        observations=(
            _complete_observations(name) if observations is None else observations
        ),
    )


def _complete_case(claim: str) -> dict[str, EvidenceBlock]:
    requirement = load_authorization_profile_v2().requirement(claim)
    assert requirement is not None
    return {name: _block(name) for name in requirement.required_blocks}


def test_v2_profile_is_separate_scoped_and_hash_addressed() -> None:
    profile = load_authorization_profile_v2()
    claims = {item.claim for item in profile.requirements}

    assert profile.profile_id == "mouse_brain_claim_authorization"
    assert profile.version == "2.0.0"
    assert profile.source_hash.startswith("sha256:")
    assert "mechanistic" not in claims
    assert "causal" not in claims
    assert "digital_twin" not in claims
    assert "directed_topology_consistent_prediction" in claims
    assert "complete_entity_specific_mouse_brain_digital_twin" in claims


def test_v2_basis_exactly_covers_every_executable_relation() -> None:
    profile = load_authorization_profile_v2()
    basis = load_authorization_profile_v2_basis()
    expected = {
        (requirement.claim, block)
        for requirement in profile.requirements
        for block in requirement.required_blocks
    }
    observed = {
        (row["claim"], row["evidence_block"]) for row in basis["relations"]
    }

    assert basis["independent_expert_validation"] == (
        "not_performed_and_not_claimed_as_consensus"
    )
    assert observed == expected


def test_all_complete_passed_blocks_produce_profile_authorization() -> None:
    profile = load_authorization_profile_v2()
    for requirement in profile.requirements:
        decision = ClaimAuthorizationSystem(
            profile, _complete_case(requirement.claim)
        ).infer(requirement.claim)
        assert decision.status is ProfileAuthorizationStatus.AUTHORIZED
        assert decision.authorized is True
        assert decision.deficits == ()


def test_passed_block_with_missing_required_observation_is_not_authorized() -> None:
    profile = load_authorization_profile_v2()
    blocks = _complete_case("bounded_predictive_performance")
    prediction = blocks["prediction"]
    observations = dict(prediction.observations)
    observations.pop("comparator")
    blocks["prediction"] = _block("prediction", observations=observations)

    decision = ClaimAuthorizationSystem(profile, blocks).infer(
        "bounded_predictive_performance"
    )

    assert decision.status is ProfileAuthorizationStatus.NOT_AUTHORIZED
    deficit = next(item for item in decision.deficits if item.name == "prediction")
    assert deficit.declared_status is EvidenceStatus.PASSED
    assert deficit.effective_status is EvidenceStatus.REQUIRES_REVIEW
    assert deficit.missing_required_observations == ("comparator",)


def test_passed_block_with_missing_required_metadata_is_not_authorized() -> None:
    profile = load_authorization_profile_v2()
    blocks = _complete_case("bounded_predictive_performance")
    prediction = blocks["prediction"]
    blocks["prediction"] = EvidenceBlock.from_mapping(
        name=prediction.name,
        status=prediction.status,
        source="",
        rule=prediction.rule,
        rationale=prediction.rationale,
        observations=dict(prediction.observations),
    )

    decision = ClaimAuthorizationSystem(profile, blocks).infer(
        "bounded_predictive_performance"
    )

    deficit = next(item for item in decision.deficits if item.name == "prediction")
    assert deficit.effective_status is EvidenceStatus.REQUIRES_REVIEW
    assert deficit.missing_required_observations == ()
    assert deficit.missing_required_metadata == ("source",)


def test_every_single_status_defect_blocks_every_claim() -> None:
    profile = load_authorization_profile_v2()
    defects = tuple(status for status in EvidenceStatus if status is not EvidenceStatus.PASSED)
    for requirement in profile.requirements:
        for block_name in requirement.required_blocks:
            for status in defects:
                blocks = _complete_case(requirement.claim)
                blocks[block_name] = _block(block_name, status=status)
                decision = ClaimAuthorizationSystem(profile, blocks).infer(
                    requirement.claim
                )
                assert decision.authorized is False
                assert [(row.name, row.effective_status) for row in decision.deficits] == [
                    (block_name, status)
                ]


def test_mixed_deficits_are_all_preserved_without_priority_masking() -> None:
    profile = load_authorization_profile_v2()
    claim = "complete_entity_specific_mouse_brain_digital_twin"
    requirement = profile.requirement(claim)
    assert requirement is not None
    names = requirement.required_blocks[:4]
    statuses = (
        EvidenceStatus.FAILED,
        EvidenceStatus.REQUIRES_REVIEW,
        EvidenceStatus.UNKNOWN,
        EvidenceStatus.NOT_APPLICABLE,
    )
    blocks = _complete_case(claim)
    for name, status in zip(names, statuses, strict=True):
        blocks[name] = _block(name, status=status)

    decision = ClaimAuthorizationSystem(profile, blocks).infer(claim)

    assert [(item.name, item.effective_status) for item in decision.deficits] == list(
        zip(names, statuses, strict=True)
    )
    assert decision.as_dict()["deficit_counts"] == {
        "failed": 1,
        "unknown": 1,
        "not_applicable": 1,
        "requires_review": 1,
    }


def test_pairwise_deficits_are_never_lost() -> None:
    profile = load_authorization_profile_v2()
    claim = "directed_topology_consistent_prediction"
    requirement = profile.requirement(claim)
    assert requirement is not None
    for left, right in combinations(requirement.required_blocks, 2):
        blocks = _complete_case(claim)
        blocks[left] = _block(left, status=EvidenceStatus.FAILED)
        blocks[right] = _block(right, status=EvidenceStatus.UNKNOWN)
        decision = ClaimAuthorizationSystem(profile, blocks).infer(claim)
        assert {item.name for item in decision.deficits} == {left, right}


def test_undeclared_broad_claim_stays_outside_profile() -> None:
    decision = ClaimAuthorizationSystem(
        load_authorization_profile_v2(), {}
    ).infer("consciousness")

    assert decision.status is ProfileAuthorizationStatus.OUTSIDE_PROFILE
    assert decision.authorized is False
    assert decision.required_blocks == ()


def test_python_and_asp_agree_on_complete_and_single_defect_boundaries() -> None:
    profile = load_authorization_profile_v2()
    for requirement in profile.requirements:
        complete = _complete_case(requirement.claim)
        python = ClaimAuthorizationSystem(profile, complete).infer(requirement.claim)
        asp = authorize_with_clingo_v2(profile, requirement.claim, complete)
        assert asp.status is python.status
        assert asp.deficits == ()

        for block_name in requirement.required_blocks:
            defective = dict(complete)
            defective[block_name] = _block(
                block_name, status=EvidenceStatus.UNKNOWN
            )
            python = ClaimAuthorizationSystem(profile, defective).infer(
                requirement.claim
            )
            asp = authorize_with_clingo_v2(profile, requirement.claim, defective)
            assert asp.status is python.status
            assert asp.deficits == ((block_name, EvidenceStatus.UNKNOWN),)


def test_python_and_asp_agree_on_metadata_and_mixed_deficits() -> None:
    profile = load_authorization_profile_v2()
    claim = "directed_topology_consistent_prediction"
    blocks = _complete_case(claim)
    observations = dict(blocks["prediction"].observations)
    observations.pop("split_integrity")
    blocks["prediction"] = _block("prediction", observations=observations)
    blocks["distribution_shift"] = _block(
        "distribution_shift", status=EvidenceStatus.NOT_APPLICABLE
    )
    blocks["competing_mechanisms"] = _block(
        "competing_mechanisms", status=EvidenceStatus.FAILED
    )

    python = ClaimAuthorizationSystem(profile, blocks).infer(claim)
    asp = authorize_with_clingo_v2(profile, claim, blocks)

    assert asp.status is python.status
    assert set(asp.deficits) == {
        ("prediction", EvidenceStatus.REQUIRES_REVIEW),
        ("distribution_shift", EvidenceStatus.NOT_APPLICABLE),
        ("competing_mechanisms", EvidenceStatus.FAILED),
    }
