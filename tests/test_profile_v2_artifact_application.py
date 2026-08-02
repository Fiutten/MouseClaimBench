from pathlib import Path

import yaml

from mousebrainbench.benchmarks.profile_v2_artifact_application import evaluate


def test_v2_artifact_application_is_bounded_and_semantically_equivalent() -> None:
    protocol = yaml.safe_load(
        Path("configs/benchmarks/profile_v2_artifact_application.yaml").read_text()
    )
    result = evaluate(protocol)
    targets = {
        row["case"]: row
        for row in result["decision_rows"]
        if row["claim"] != "complete_entity_specific_mouse_brain_digital_twin"
    }

    assert result["cases"] == 5
    assert result["decisions"] == 10
    assert result["strict_twin_authorizations"] == 0
    assert targets["sensorium_static_bounded_prediction"]["authorized"] is True
    assert targets["allen_vbn_negative"]["authorized"] is False
    assert targets["dynamic_sensorium_prediction_with_quality_deficit"][
        "authorized"
    ] is False
    microns = targets["microns_local_association_with_dependence_deficit"]
    assert microns["authorized"] is False
    assert any(
        row["name"] == "network_dependence_control" for row in microns["deficits"]
    )
    ibl = targets["ibl_behavior_topology_specific_prediction"]
    assert ibl["authorized"] is True
    assert ibl["deficits"] == []
    assert all(result["release_conditions"].values())
