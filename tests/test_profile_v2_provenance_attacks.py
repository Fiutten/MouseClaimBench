from pathlib import Path

import yaml

from mousebrainbench.benchmarks.profile_v2_contract_mutation import _complete_blocks
from mousebrainbench.benchmarks.profile_v2_provenance_attacks import (
    _apply_attack,
    _base_manifest,
    evaluate,
)
from mousebrainbench.knowledge import load_authorization_profile_v2
from mousebrainbench.knowledge.integrity import (
    IntegrityAwareAuthorizationSystem,
    IntegrityDeficitCode,
)


def test_each_integrity_attack_is_detected_without_masking() -> None:
    profile = load_authorization_profile_v2()
    claim = "directed_topology_consistent_prediction"
    manifest = _base_manifest(claim)
    expected = {
        "artifact_hash_tampering": IntegrityDeficitCode.ARTIFACT_HASH_MISMATCH,
        "profile_version_substitution": IntegrityDeficitCode.PROFILE_IDENTITY_MISMATCH,
        "dangling_provenance_reference": IntegrityDeficitCode.UNKNOWN_PROVENANCE_REFERENCE,
        "circular_provenance": IntegrityDeficitCode.PROVENANCE_CYCLE,
        "duplicate_independent_artifact": IntegrityDeficitCode.DUPLICATE_INDEPENDENT_ARTIFACT,
        "overlapping_independent_cohorts": IntegrityDeficitCode.OVERLAPPING_INDEPENDENT_COHORTS,
        "contradictory_attestation": IntegrityDeficitCode.CONTRADICTORY_ATTESTATION,
        "missing_block_lineage": IntegrityDeficitCode.MISSING_BLOCK_LINEAGE,
    }
    attacked = manifest
    for attack in expected:
        attacked = _apply_attack(attacked, attack)
    decision = IntegrityAwareAuthorizationSystem(
        profile, _complete_blocks(claim), attacked
    ).infer(claim)

    assert decision.authorized is False
    assert {row.code for row in decision.integrity_deficits} == set(expected.values())


def test_pristine_manifest_preserves_profile_authorization() -> None:
    profile = load_authorization_profile_v2()
    claim = "bounded_predictive_performance"
    decision = IntegrityAwareAuthorizationSystem(
        profile, _complete_blocks(claim), _base_manifest(claim)
    ).infer(claim)

    assert decision.core.authorized is True
    assert decision.integrity_deficits == ()
    assert decision.authorized is True


def test_frozen_attack_benchmark_has_exact_traces() -> None:
    protocol = yaml.safe_load(
        Path("configs/benchmarks/profile_v2_provenance_attacks.yaml").read_text()
    )
    result = evaluate(protocol)

    assert result["cases"] == 370
    assert result["attacked_cases"] == 360
    assert result["full_integrity_gate"]["false_authorizations"] == 0
    assert result["full_integrity_gate"]["exact_attack_trace_rate"] == 1.0
    assert all(result["endpoints"].values())
