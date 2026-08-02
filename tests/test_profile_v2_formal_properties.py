from pathlib import Path

import yaml
from hypothesis import given, settings
from hypothesis import strategies as st

from mousebrainbench.benchmarks.profile_v2_contract_mutation import (
    _block,
    _complete_blocks,
)
from mousebrainbench.benchmarks.profile_v2_formal_properties import evaluate
from mousebrainbench.knowledge import (
    ClaimAuthorizationSystem,
    load_authorization_profile_v2,
)
from mousebrainbench.validation.evidence_contract import EvidenceStatus


def test_frozen_formal_property_protocol_has_no_violations() -> None:
    protocol = yaml.safe_load(
        Path("configs/benchmarks/profile_v2_formal_properties.yaml").read_text()
    )
    protocol["deterministic_random_packages"]["packages_per_claim"] = 10
    result = evaluate(protocol)

    assert result["random_packages"] == 100
    assert result["total_violations"] == 0
    assert result["all_properties_hold"] is True


@settings(max_examples=100, deadline=None)
@given(st.lists(st.sampled_from(tuple(EvidenceStatus)), min_size=10, max_size=10))
def test_authorization_is_exact_conjunction_for_directed_topology_claim(
    statuses: list[EvidenceStatus],
) -> None:
    profile = load_authorization_profile_v2()
    claim = "directed_topology_consistent_prediction"
    requirement = profile.requirement(claim)
    assert requirement is not None
    blocks = _complete_blocks(claim)
    for name, status in zip(requirement.required_blocks, statuses, strict=True):
        blocks[name] = _block(name, status)

    decision = ClaimAuthorizationSystem(profile, blocks).infer(claim)

    assert decision.authorized is all(
        status is EvidenceStatus.PASSED for status in statuses
    )
    assert {fact.name for fact in decision.deficits} == {
        name
        for name, status in zip(requirement.required_blocks, statuses, strict=True)
        if status is not EvidenceStatus.PASSED
    }
