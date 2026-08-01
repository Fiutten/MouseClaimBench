import json

import numpy as np
import pytest

from mousebrainbench.benchmarks.prospective_probabilistic_baseline import (
    DEFAULT_PROTOCOL,
    development_records,
    load_frozen_model,
    predict_claims,
    train,
)


def test_development_records_are_deterministic_and_dgp_labeled() -> None:
    first = development_records(seeds=2, n_per_cohort=180)
    second = development_records(seeds=2, n_per_cohort=180)

    assert len(first) == 10
    assert [row["regime"] for row in first] == [row["regime"] for row in second]
    assert all(
        np.array_equal(left["features"], right["features"])
        for left, right in zip(first, second, strict=True)
    )
    assert first[0]["reference_claims"] == {"computationally_reproducible"}


def test_trained_model_records_development_only_provenance(tmp_path) -> None:
    output = train(
        output=tmp_path / "model.json",
        protocol_path=DEFAULT_PROTOCOL,
        seeds=3,
        n_per_cohort=180,
    )
    payload = json.loads(output.read_text())

    assert payload["training_partition"] == "development_partition_only"
    assert payload["prospective_data_used_for_training"] is False
    assert payload["training_cases"] == 15
    assert set(payload["models"]) == {
        "predictive",
        "computationally_reproducible",
        "internally_reproduced",
        "externally_replicated",
        "topology_specific",
        "directed",
        "structure_function",
        "mechanistic",
        "causal",
        "digital_twin",
    }
    model = load_frozen_model(output, protocol_path=DEFAULT_PROTOCOL)
    predictions = predict_claims(model, development_records(seeds=1)[0]["features"])
    assert "computationally_reproducible" in predictions
    assert "digital_twin" not in predictions


def test_model_loader_rejects_protocol_mismatch(tmp_path) -> None:
    output = train(
        output=tmp_path / "model.json",
        protocol_path=DEFAULT_PROTOCOL,
        seeds=2,
        n_per_cohort=180,
    )
    payload = json.loads(output.read_text())
    payload["protocol_hash"] = "sha256:invalid"
    output.write_text(json.dumps(payload))

    with pytest.raises(ValueError, match="protocol_hash"):
        load_frozen_model(output, protocol_path=DEFAULT_PROTOCOL)

