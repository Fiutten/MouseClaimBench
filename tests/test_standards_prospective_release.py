import json
from pathlib import Path

import yaml

from mousebrainbench.benchmarks.standards_prospective_release import evaluate


def test_standards_prospective_release_is_bounded_and_complete() -> None:
    protocol = yaml.safe_load(
        Path("configs/benchmarks/standards_prospective_release.yaml").read_text()
    )

    result = evaluate(protocol)

    assert all(result["conditions"].values())
    assert result["evidence_counts"]["contract_cases"] == 5497
    assert result["evidence_counts"]["formal_checks"] >= 55_000
    assert result["evidence_counts"]["attacked_cases"] == 360
    assert result["evidence_counts"]["prospective_positive_authorizations"] == 1
    assert not any(result["claim_policy"].values())
    assert result["technically_ready_for_kbs_manuscript"] is True


def test_frozen_standards_prospective_release_is_clean() -> None:
    path = Path("results/standards_prospective_release/summary.json")
    if not path.exists():
        return
    payload = json.loads(path.read_text())

    assert payload["decision"] == "standards_prospective_release_complete"
    assert payload["technically_ready_for_kbs_manuscript"] is True
    assert not payload["git_revision"].endswith("-dirty")
