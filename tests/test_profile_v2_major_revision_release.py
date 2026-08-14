import json
from pathlib import Path

import yaml

from mousebrainbench.benchmarks.profile_v2_major_revision_release import evaluate


def test_major_revision_release_is_complete_and_bounded() -> None:
    protocol = yaml.safe_load(
        Path("configs/benchmarks/profile_v2_major_revision_release.yaml").read_text()
    )
    result = evaluate(protocol)

    assert all(result["conditions"].values())
    assert result["all_release_conditions_passed"] is True
    assert not any(result["claim_policy"].values())


def test_frozen_major_revision_release_is_clean() -> None:
    path = Path("results/profile_v2_major_revision_release/summary.json")
    if not path.exists():
        return
    payload = json.loads(path.read_text())

    assert payload["decision"] == "profile_v2_major_revision_release_complete"
    assert payload["all_release_conditions_passed"] is True
    assert not payload["git_revision"].endswith("-dirty")
