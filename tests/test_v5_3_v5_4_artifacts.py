import json
from pathlib import Path


def test_ibl_external_population_artifact_respects_mouse_level_contract() -> None:
    payload = json.loads(
        Path("results/ibl_behavior_v5_confirmation/summary.json").read_text()
    )
    assert payload["selected_mice"] == 110
    assert payload["verified_tables"] == 110
    assert payload["inferential_unit"] == "mouse"
    assert payload["threshold_refitted"] is False
    assert payload["risk_lock"]["passed"] is True
    assert payload["final_evaluation"]["opened"] is True
    assert payload["final_evaluation"]["passed"] is True
    for role in (payload["risk_lock"], payload["final_evaluation"]):
        primary = role["comparators"]["frozen_v5_1_complete_authorizer"]
        assert primary["experiments"] == 35
        assert primary["failing_experiments"] == 0
        assert primary["certified"] is True


def test_orthogonal_shift_artifact_keeps_failures_visible() -> None:
    payload = json.loads(
        Path("results/semantic_risk_orthogonal_shifts_v5_4/summary.json").read_text()
    )
    assert payload["threshold_refitted"] is False
    assert payload["simultaneous_contract"]["total_levels"] == 13
    assert payload["all_levels_certified"] is False
    failed = {
        name
        for family in payload["families"].values()
        for name, level in family["levels"].items()
        if not level["certificate"]["certified"]
    }
    assert failed == {"tail_t2", "dimension_8"}
