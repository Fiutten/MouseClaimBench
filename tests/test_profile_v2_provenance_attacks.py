from dataclasses import replace
from pathlib import Path

import pytest
import yaml

from mousebrainbench.benchmarks.profile_v2_contract_mutation import _block, _complete_blocks
from mousebrainbench.benchmarks.profile_v2_provenance_attacks import (
    _apply_attack,
    _base_manifest,
    evaluate,
)
from mousebrainbench.knowledge import load_authorization_profile_v2
from mousebrainbench.knowledge.integrity import (
    DomainIntegrityAuthorizationSystem,
    EvidenceAttestation,
    IntegrityDeficitCode,
    validate_evidence_manifest,
)
from mousebrainbench.validation.evidence_contract import EvidenceStatus


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
    decision = DomainIntegrityAuthorizationSystem(
        profile, _complete_blocks(claim), attacked
    ).infer(claim)

    assert decision.authorized is False
    assert {row.code for row in decision.integrity_deficits} == {
        *expected.values(),
        IntegrityDeficitCode.ATTESTATION_BLOCK_STATUS_MISMATCH,
    }


def test_pristine_manifest_preserves_profile_authorization() -> None:
    profile = load_authorization_profile_v2()
    claim = "bounded_predictive_performance"
    decision = DomainIntegrityAuthorizationSystem(
        profile, _complete_blocks(claim), _base_manifest(claim)
    ).infer(claim)

    assert decision.core.authorized is True
    assert decision.integrity_deficits == ()
    assert decision.authorized is True


@pytest.mark.parametrize(
    ("relation", "pair"),
    (
        ("attestation", None),
        ("independence", ("missing-left", "existing")),
        ("independence", ("existing", "missing-right")),
        ("cohort", ("missing-left", "existing")),
        ("cohort", ("existing", "missing-right")),
    ),
)
def test_every_artifact_reference_must_resolve(
    relation: str, pair: tuple[str, str] | None
) -> None:
    profile = load_authorization_profile_v2()
    claim = "bounded_predictive_performance"
    blocks = _complete_blocks(claim)
    manifest = _base_manifest(claim)
    existing = manifest.artifacts[0].artifact_id
    if relation == "attestation":
        first = replace(manifest.attestations[0], artifact_id="missing-attestation-artifact")
        manifest = replace(manifest, attestations=(first, *manifest.attestations[1:]))
    elif relation == "independence":
        left, right = pair or ("", "")
        manifest = replace(
            manifest,
            independent_artifact_pairs=((existing if left == "existing" else left,
                                         existing if right == "existing" else right),),
        )
    else:
        left, right = pair or ("", "")
        manifest = replace(
            manifest,
            disjoint_cohort_pairs=((existing if left == "existing" else left,
                                    existing if right == "existing" else right),),
        )

    deficits = validate_evidence_manifest(profile, blocks, manifest)
    assert IntegrityDeficitCode.UNKNOWN_PROVENANCE_REFERENCE in {
        row.code for row in deficits
    }


def test_attestation_unknown_block_is_an_integrity_deficit() -> None:
    profile = load_authorization_profile_v2()
    claim = "bounded_predictive_performance"
    blocks = _complete_blocks(claim)
    manifest = _base_manifest(claim)
    unknown = EvidenceAttestation(
        "nonexistent-block", EvidenceStatus.PASSED, manifest.artifacts[0].artifact_id
    )

    deficits = validate_evidence_manifest(
        profile, blocks, replace(manifest, attestations=(*manifest.attestations, unknown))
    )
    assert IntegrityDeficitCode.UNKNOWN_BLOCK_REFERENCE in {row.code for row in deficits}


@pytest.mark.parametrize(
    ("block_status", "attestation_status"),
    (
        (EvidenceStatus.PASSED, EvidenceStatus.FAILED),
        (EvidenceStatus.FAILED, EvidenceStatus.PASSED),
    ),
)
def test_block_and_attestation_status_must_match(
    block_status: EvidenceStatus, attestation_status: EvidenceStatus
) -> None:
    profile = load_authorization_profile_v2()
    claim = "bounded_predictive_performance"
    blocks = _complete_blocks(claim)
    target = "prediction"
    blocks[target] = _block(target, block_status)
    manifest = _base_manifest(claim)
    attestations = tuple(
        replace(row, status=attestation_status) if row.block_name == target else row
        for row in manifest.attestations
    )

    deficits = validate_evidence_manifest(
        profile, blocks, replace(manifest, attestations=attestations)
    )
    assert IntegrityDeficitCode.ATTESTATION_BLOCK_STATUS_MISMATCH in {
        row.code for row in deficits
    }


def test_every_supplied_block_requires_an_attestation() -> None:
    profile = load_authorization_profile_v2()
    claim = "bounded_predictive_performance"
    blocks = _complete_blocks(claim)
    manifest = _base_manifest(claim)

    deficits = validate_evidence_manifest(
        profile, blocks, replace(manifest, attestations=manifest.attestations[1:])
    )
    assert IntegrityDeficitCode.MISSING_BLOCK_ATTESTATION in {
        row.code for row in deficits
    }


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
