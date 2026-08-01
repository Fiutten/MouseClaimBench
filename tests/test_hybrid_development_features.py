import json

import numpy as np

from mousebrainbench.benchmarks.hybrid_development_features import (
    development_records,
    hybrid_feature_names,
    run,
)


def _fake_direction(x, y, *, seed):
    del x, y, seed
    return {
        "p_forward": 0.8,
        "p_backward": 0.1,
        "signed_margin": 0.7,
        "absolute_margin": 0.7,
        "predicted_direction": "forward",
        "status": "passed",
        "execution_error": None,
    }


def test_development_rows_have_disjoint_deterministic_roles() -> None:
    rows = development_records(test_mode=True, direction_function=_fake_direction)

    assert len(rows) == 70
    assert {row["split_role"] for row in rows} == {
        "model_fit",
        "threshold_calibration",
        "locked_development_audit",
    }
    assert all(len(row["features"]) == len(hybrid_feature_names()) for row in rows)
    assert all(np.all(np.isfinite(row["features"])) for row in rows)


def test_feature_artifact_records_no_confirmatory_leakage(tmp_path) -> None:
    output = run(
        output=tmp_path / "summary.json",
        matrix=tmp_path / "cases.npz",
        test_mode=True,
        direction_function=_fake_direction,
    )
    payload = json.loads(output.read_text())
    matrix = np.load(tmp_path / "cases.npz")

    assert payload["confirmatory_v2_cases_used"] == 0
    assert payload["data_role"] == "consumed_development_only_not_confirmatory"
    assert matrix["features"].shape == (70, len(hybrid_feature_names()))
    assert matrix["labels"].shape == (70, 10)
