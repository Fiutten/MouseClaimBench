from dataclasses import replace
from pathlib import Path

import yaml

from mousebrainbench.benchmarks.profile_v2_scalability_ablation import (
    _expanded_manifest,
    evaluate_ablation,
    evaluate_scaling,
)
from mousebrainbench.benchmarks.profile_v2_contract_mutation import _complete_blocks
from mousebrainbench.knowledge import load_authorization_profile_v2
from mousebrainbench.knowledge.integrity import validate_evidence_manifest


def test_iterative_cycle_check_handles_thousands_of_artifacts() -> None:
    profile = load_authorization_profile_v2()
    manifest = _expanded_manifest(
        "complete_entity_specific_mouse_brain_digital_twin", 2500
    )

    blocks = _complete_blocks("complete_entity_specific_mouse_brain_digital_twin")
    assert validate_evidence_manifest(profile, blocks, manifest) == ()

    first = manifest.artifacts[0]
    last = manifest.artifacts[-1]
    artifacts = list(manifest.artifacts)
    artifacts[0] = replace(first, derived_from=(last.artifact_id,))
    artifacts[-1] = replace(last, derived_from=(first.artifact_id,))

    deficits = validate_evidence_manifest(
        profile, blocks, replace(manifest, artifacts=tuple(artifacts))
    )
    assert any(row.code.value == "provenance_cycle" for row in deficits)


def test_ablation_reports_declared_control_redundancy() -> None:
    protocol = yaml.safe_load(
        Path("configs/benchmarks/profile_v2_provenance_attacks.yaml").read_text()
    )

    result = evaluate_ablation(protocol)

    assert result["systems"]["full_integrity"]["false_authorizations"] == 0
    assert result["systems"]["profile_only"]["false_authorizations"] == 360
    assert result["systems"]["hash_only"]["false_authorizations"] == 280
    omissions = {
        name: row["false_authorizations"]
        for name, row in result["systems"].items()
        if name.startswith("without_")
    }
    assert omissions["without_contradictory_attestation"] == 0
    assert all(
        count == 10
        for name, count in omissions.items()
        if name != "without_contradictory_attestation"
    )


def test_small_scalability_protocol_preserves_pristine_decisions() -> None:
    protocol = yaml.safe_load(
        Path("configs/benchmarks/profile_v2_scalability_ablation.yaml").read_text()
    )
    protocol["batch_scaling"] = {
        "package_counts": [10, 50],
        "repetitions": 2,
        "warmup_packages": 5,
    }
    protocol["artifact_scaling"] = {
        "artifact_counts": [25, 50],
        "repetitions": 2,
        "claim": "complete_entity_specific_mouse_brain_digital_twin",
    }

    result = evaluate_scaling(protocol)

    assert result["all_pristine_decisions_authorized"] is True
    assert len(result["batch_scaling"]) == 2
    assert len(result["artifact_scaling"]) == 2
