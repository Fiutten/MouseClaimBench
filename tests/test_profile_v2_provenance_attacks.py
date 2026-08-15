from dataclasses import replace
from pathlib import Path

import pytest
import yaml

from mousebrainbench.benchmarks.profile_v2_contract_mutation import _block, _complete_blocks
from mousebrainbench.benchmarks.profile_v2_provenance_attacks import (
    ORIGINAL_ATTACK_FAMILIES,
    _apply_attack,
    _base_manifest,
    evaluate,
)
from mousebrainbench.knowledge import FinalAuthorizationSystem, load_authorization_profile_v2
from mousebrainbench.knowledge.integrity import (
    DomainIntegrityAuthorizationSystem,
    EvidenceAttestation,
    IntegrityDeficitCode,
    validate_evidence_manifest,
)
from mousebrainbench.validation.evidence_contract import EvidenceStatus


def test_integrity_taxonomy_declares_thirteen_deficit_types() -> None:
    assert len(IntegrityDeficitCode) == 13


def test_original_integrity_benchmark_has_eight_historical_families() -> None:
    assert len(ORIGINAL_ATTACK_FAMILIES) == 8


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


def test_duplicate_artifact_identifiers_return_a_structured_final_refusal() -> None:
    profile = load_authorization_profile_v2()
    claim = "bounded_predictive_performance"
    blocks = _complete_blocks(claim)
    manifest = _base_manifest(claim)
    duplicate = replace(
        manifest.artifacts[1], artifact_id=manifest.artifacts[0].artifact_id
    )
    malformed = replace(
        manifest, artifacts=(manifest.artifacts[0], duplicate, *manifest.artifacts[2:])
    )

    decision = FinalAuthorizationSystem(profile, blocks, malformed).infer(claim)

    assert decision.authorized is False
    assert IntegrityDeficitCode.DUPLICATE_ARTIFACT_ID in {
        row.code for row in decision.integrity_deficits
    }


def test_duplicate_block_lineage_returns_a_structured_final_refusal() -> None:
    profile = load_authorization_profile_v2()
    claim = "bounded_predictive_performance"
    blocks = _complete_blocks(claim)
    manifest = _base_manifest(claim)
    malformed = replace(
        manifest, block_artifacts=(*manifest.block_artifacts, manifest.block_artifacts[0])
    )

    decision = FinalAuthorizationSystem(profile, blocks, malformed).infer(claim)

    assert decision.authorized is False
    assert IntegrityDeficitCode.DUPLICATE_BLOCK_LINEAGE in {
        row.code for row in decision.integrity_deficits
    }


@pytest.mark.parametrize(
    ("relation", "expected"),
    (
        ("independence", IntegrityDeficitCode.DUPLICATE_INDEPENDENT_ARTIFACT),
        ("cohort", IntegrityDeficitCode.OVERLAPPING_INDEPENDENT_COHORTS),
    ),
)
def test_reflexive_relations_are_rejected_even_with_empty_cohorts(
    relation: str, expected: IntegrityDeficitCode
) -> None:
    profile = load_authorization_profile_v2()
    claim = "bounded_predictive_performance"
    blocks = _complete_blocks(claim)
    manifest = _base_manifest(claim)
    artifact_id = manifest.artifacts[0].artifact_id
    artifacts = (replace(manifest.artifacts[0], cohorts=()), *manifest.artifacts[1:])
    if relation == "independence":
        malformed = replace(
            manifest,
            artifacts=artifacts,
            independent_artifact_pairs=((artifact_id, artifact_id),),
        )
    else:
        malformed = replace(
            manifest,
            artifacts=artifacts,
            disjoint_cohort_pairs=((artifact_id, artifact_id),),
        )

    deficits = validate_evidence_manifest(profile, blocks, malformed)

    matching = tuple(row for row in deficits if row.code is expected)
    assert len(matching) == 1
    assert matching[0].witnesses == (f"{artifact_id}|{artifact_id}:reflexive",)


@pytest.mark.parametrize(("same_study", "rejected"), ((False, False), (True, True)))
def test_data_generation_identity_is_scoped_by_study(
    same_study: bool, rejected: bool
) -> None:
    profile = load_authorization_profile_v2()
    claim = "bounded_predictive_performance"
    blocks = _complete_blocks(claim)
    manifest = _base_manifest(claim)
    first, second = manifest.artifacts[:2]
    second = replace(
        second,
        study_id=first.study_id if same_study else "independent-study",
        data_generation_id=first.data_generation_id,
    )
    compared = replace(
        manifest,
        artifacts=(first, second, *manifest.artifacts[2:]),
        independent_artifact_pairs=((first.artifact_id, second.artifact_id),),
    )

    deficits = validate_evidence_manifest(profile, blocks, compared)
    codes = {row.code for row in deficits}

    assert (IntegrityDeficitCode.DUPLICATE_INDEPENDENT_ARTIFACT in codes) is rejected


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
