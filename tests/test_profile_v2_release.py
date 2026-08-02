import json
from pathlib import Path

import yaml

from mousebrainbench.benchmarks.profile_v2_release import evaluate


def test_profile_v2_release_is_technical_not_consensus_or_acceptance_claim() -> None:
    protocol = yaml.safe_load(
        Path("configs/benchmarks/profile_v2_release.yaml").read_text()
    )
    result = evaluate(protocol)

    assert result["contract_mutation_cases"] > 5_000
    assert result["contract_mutation_false_authorizations"] == 0
    assert result["bounded_real_target_authorizations"] >= 2
    assert result["strict_twin_authorizations"] == 0
    assert result["ibl_locked_mice"] == 70
    assert result["orthogonal_shift_families"] >= 4
    assert result["orthogonal_shift_certificate_failures"] >= 2
    assert all(result["conditions"].values())
    assert not any(result["claim_boundaries"].values())
    assert result["technically_ready_for_manuscript_revision"] is True
    assert result["publication_assessment"] == (
        "bounded_methodological_submission_candidate_not_acceptance_guarantee"
    )


def test_frozen_profile_v2_release_is_clean_and_complete() -> None:
    payload = json.loads(Path("results/profile_v2_release/summary.json").read_text())

    assert payload["decision"] == "profile_v2_technical_release_complete"
    assert payload["technically_ready_for_manuscript_revision"] is True
    assert all(payload["conditions"].values())
    assert not payload["git_revision"].endswith("-dirty")
