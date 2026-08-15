from dataclasses import replace
from itertools import product

from mousebrainbench.benchmarks.profile_v2_contract_mutation import (
    _block,
    _complete_blocks,
)
from mousebrainbench.benchmarks.profile_v2_provenance_attacks import _base_manifest
from mousebrainbench.knowledge import (
    FinalAuthorizationSystem,
    compose_final_authorization,
    load_authorization_profile_v2,
)
from mousebrainbench.validation.evidence_contract import EvidenceStatus


def test_three_gate_truth_table_is_strictly_non_compensatory() -> None:
    for structural, domain, integrity in product((False, True), repeat=3):
        assert compose_final_authorization(structural, domain, integrity) is (
            structural and domain and integrity
        )


def test_final_system_preserves_layer_resolved_decisions() -> None:
    profile = load_authorization_profile_v2()
    claim = "bounded_predictive_performance"
    blocks = _complete_blocks(claim)
    manifest = _base_manifest(claim)

    pristine = FinalAuthorizationSystem(profile, blocks, manifest).infer(claim)
    assert pristine.structural.conforms is True
    assert pristine.domain.authorized is True
    assert pristine.integrity_conforms is True
    assert pristine.authorized is True

    blocks["prediction"] = _block("prediction", EvidenceStatus.FAILED)
    attestations = tuple(
        replace(row, status=EvidenceStatus.FAILED)
        if row.block_name == "prediction"
        else row
        for row in manifest.attestations
    )
    bounded_refusal = FinalAuthorizationSystem(
        profile, blocks, replace(manifest, attestations=attestations)
    ).infer(claim)
    assert bounded_refusal.structural.conforms is True
    assert bounded_refusal.domain.authorized is False
    assert bounded_refusal.integrity_conforms is True
    assert bounded_refusal.authorized is False
