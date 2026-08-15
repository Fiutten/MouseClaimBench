import json
from pathlib import Path

def test_second_review_release_is_complete_and_bounded() -> None:
    # This is a historical release. Validate its frozen decision rather than
    # re-evaluating its 5,497-case contract against later mutable artifacts.
    result = json.loads(
        Path("results/profile_v2_second_review_release/summary.json").read_text()
    )

    assert all(result["conditions"].values())
    assert result["all_release_conditions_passed"] is True
    assert not any(result["claim_policy"].values())


def test_frozen_second_review_release_is_clean() -> None:
    payload = json.loads(
        Path("results/profile_v2_second_review_release/summary.json").read_text()
    )

    assert payload["decision"] == "profile_v2_second_review_release_complete"
    assert payload["all_release_conditions_passed"] is True
    assert not payload["git_revision"].endswith("-dirty")
