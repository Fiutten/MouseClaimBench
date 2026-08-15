from hypothesis import given, settings
from hypothesis import strategies as st

from mousebrainbench.benchmarks.profile_v2_contract_mutation import (
    _block,
    _complete_blocks,
    generate_cases,
)
from mousebrainbench.benchmarks.profile_v2_standards import (
    _representative_round_trip_graphs,
)
from mousebrainbench.knowledge import load_authorization_profile_v2
from mousebrainbench.knowledge.authorization import ClaimAuthorizationSystem
from mousebrainbench.knowledge.standards import (
    profile_to_rdf,
    validate_structure_with_shacl_v2,
)
from mousebrainbench.validation.evidence_contract import EvidenceStatus


def test_prov_o_profile_and_representative_json_ld_round_trips_are_nonempty() -> None:
    from rdflib import Graph
    from rdflib.compare import isomorphic

    profile = load_authorization_profile_v2()
    profile_graph = profile_to_rdf(profile)
    assert len(profile_graph) > 100
    cases = _representative_round_trip_graphs(profile, generate_cases())
    assert len(cases) == 5
    for _, package_graph in cases:
        round_trip = Graph().parse(
            data=package_graph.serialize(format="json-ld"), format="json-ld"
        )
        assert len(package_graph) > 0
        assert isomorphic(package_graph, round_trip)


def test_shacl_matches_all_single_and_metadata_mutations_structurally() -> None:
    profile = load_authorization_profile_v2()
    families = {
        "pristine_complete",
        "single_status_defect",
        "omitted_required_block",
        "missing_required_observation",
        "missing_source",
        "missing_rule",
        "missing_rationale",
    }
    cases = tuple(case for case in generate_cases() if case.family in families)

    for case in cases:
        decision = validate_structure_with_shacl_v2(
            profile, case.claim, case.blocks, package_id=case.case_id
        )
        observed = tuple((row.code.value, row.witness) for row in decision.deficits)
        assert decision.conforms is case.expected_structural_conforms
        assert observed == case.expected_structural_deficits


@settings(max_examples=75, deadline=None)
@given(
    st.sampled_from(tuple(EvidenceStatus)),
    st.sampled_from(tuple(EvidenceStatus)),
)
def test_nonpassing_statuses_remain_structurally_valid(
    left_status: EvidenceStatus,
    right_status: EvidenceStatus,
) -> None:
    profile = load_authorization_profile_v2()
    claim = "directed_topology_consistent_prediction"
    blocks = _complete_blocks(claim)
    blocks["prediction"] = _block("prediction", left_status)
    blocks["robustness"] = _block("robustness", right_status)
    decision = validate_structure_with_shacl_v2(profile, claim, blocks)

    assert decision.conforms is True
    assert decision.deficits == ()


def test_canonical_structure_domain_separation() -> None:
    profile = load_authorization_profile_v2()
    claim = "bounded_predictive_performance"
    blocks = _complete_blocks(claim)

    passing_structure = validate_structure_with_shacl_v2(profile, claim, blocks)
    passing_domain = ClaimAuthorizationSystem(profile, blocks).infer(claim)
    assert passing_structure.conforms is True
    assert passing_domain.authorized is True

    blocks["prediction"] = _block("prediction", EvidenceStatus.FAILED)
    failed_structure = validate_structure_with_shacl_v2(profile, claim, blocks)
    failed_domain = ClaimAuthorizationSystem(profile, blocks).infer(claim)

    assert failed_structure.conforms is True
    assert failed_domain.authorized is False
