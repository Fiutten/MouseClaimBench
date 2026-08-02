from hypothesis import given, settings
from hypothesis import strategies as st

from mousebrainbench.benchmarks.profile_v2_contract_mutation import (
    _block,
    _complete_blocks,
    generate_cases,
)
from mousebrainbench.knowledge import load_authorization_profile_v2
from mousebrainbench.knowledge.standards import (
    authorize_with_shacl_v2,
    evidence_package_to_rdf,
    profile_to_rdf,
)
from mousebrainbench.validation.evidence_contract import EvidenceStatus


def test_prov_o_profile_and_json_ld_round_trip_are_nonempty() -> None:
    from rdflib import Graph
    from rdflib.compare import isomorphic

    profile = load_authorization_profile_v2()
    profile_graph = profile_to_rdf(profile)
    case = generate_cases()[0]
    package_graph, _ = evidence_package_to_rdf(
        profile, case.claim, case.blocks, package_id=case.case_id
    )
    round_trip = Graph().parse(
        data=package_graph.serialize(format="json-ld"), format="json-ld"
    )

    assert len(profile_graph) > 100
    assert isomorphic(package_graph, round_trip)


def test_shacl_matches_all_single_and_metadata_mutations() -> None:
    profile = load_authorization_profile_v2()
    families = {
        "pristine_complete",
        "single_status_defect",
        "omitted_required_block",
        "missing_required_observation",
    }
    cases = tuple(case for case in generate_cases() if case.family in families)

    for case in cases:
        decision = authorize_with_shacl_v2(
            profile, case.claim, case.blocks, package_id=case.case_id
        )
        assert decision.authorized is case.expected_authorized
        assert decision.deficits == case.expected_deficits


@settings(max_examples=75, deadline=None)
@given(
    st.sampled_from(tuple(EvidenceStatus)),
    st.sampled_from(tuple(EvidenceStatus)),
)
def test_shacl_preserves_two_independent_deficits(
    left_status: EvidenceStatus,
    right_status: EvidenceStatus,
) -> None:
    profile = load_authorization_profile_v2()
    claim = "directed_topology_consistent_prediction"
    blocks = _complete_blocks(claim)
    blocks["prediction"] = _block("prediction", left_status)
    blocks["robustness"] = _block("robustness", right_status)
    decision = authorize_with_shacl_v2(profile, claim, blocks)
    expected = tuple(
        sorted(
            (
                (name, status)
                for name, status in (
                    ("prediction", left_status),
                    ("robustness", right_status),
                )
                if status is not EvidenceStatus.PASSED
            ),
            key=lambda item: item[0],
        )
    )

    assert decision.deficits == expected
