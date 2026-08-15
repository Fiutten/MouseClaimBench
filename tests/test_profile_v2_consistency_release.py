import json
from pathlib import Path


def test_consistency_release_artifact_is_complete_when_present() -> None:
    path = Path("results/profile_v2_consistency_release/summary.json")
    if not path.is_file():
        return
    payload = json.loads(path.read_text())
    assert payload["decision"] == "profile_v2_consistency_release_complete"
    assert payload["all_release_conditions_passed"] is True
    assert all(payload["conditions"].values())
    assert not payload["git_revision"].endswith("-dirty")
