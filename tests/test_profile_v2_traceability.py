import json
from pathlib import Path

import yaml

from mousebrainbench.benchmarks.profile_v2_traceability import evaluate
from mousebrainbench.knowledge import load_authorization_profile_v2_basis


def test_traceability_exactly_covers_profile_v2_without_claiming_consensus() -> None:
    protocol = yaml.safe_load(Path("configs/validation/profile_v2_traceability.yaml").read_text())
    result = evaluate(protocol, Path("references.bib").read_text())

    assert result["claims"] == 10
    assert result["evidence_blocks"] == 22
    assert result["claim_to_evidence_relations"] == 60
    assert result["claim_specific_justifications"] == 60
    assert result["unique_necessity_rationales"] == 60
    assert result["predicate_contracts"] == 22
    assert result["required_observation_slots"] == 124
    assert result["unresolved_source_ids"] == []
    assert result["all_conditions_passed"] is True
    assert result["independent_content_validity"] is False
    assert result["human_validation"] is False


def test_basis_loader_rejects_no_traceability_component() -> None:
    basis = load_authorization_profile_v2_basis()

    assert len(basis["relation_justifications"]) == 60
    assert len(basis["predicate_contracts"]) == 22
    assert basis["knowledge_acquisition"]["independent_content_validation"] == "not_performed"


def test_frozen_traceability_artifact_is_complete_and_clean() -> None:
    payload = json.loads(Path("results/profile_v2_traceability/summary.json").read_text())

    assert payload["all_conditions_passed"] is True
    assert payload["decision"] == "profile_v2_traceability_complete_author_policy_only"
    assert payload["independent_content_validity"] is False
    assert not payload["git_revision"].endswith("-dirty")
