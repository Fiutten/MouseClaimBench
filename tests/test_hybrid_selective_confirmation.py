import json

import numpy as np
import yaml

from mousebrainbench.benchmarks.hybrid_selective_confirmation import (
    CONFIRMATORY_TRUTHS,
    run,
)


def _fake_direction(x, y, *, seed):
    del seed
    correlation = float(np.corrcoef(x, y)[0, 1])
    if abs(correlation) < 0.15:
        return {
            "status": "requires_review",
            "predicted_direction": "uncertain",
            "p_forward": 0.5,
            "p_backward": 0.5,
            "signed_margin": 0.0,
            "absolute_margin": 0.0,
            "execution_error": None,
        }
    return {
        "status": "passed",
        "predicted_direction": "forward",
        "p_forward": 0.8,
        "p_backward": 0.1,
        "signed_margin": 0.7,
        "absolute_margin": 0.7,
        "execution_error": None,
    }


def test_truth_map_matches_frozen_protocol():
    protocol = yaml.safe_load(
        open("configs/benchmarks/hybrid_selective_claim_validation_v2.yaml")
    )
    assert tuple(protocol["confirmatory_partition_v2"]["regimes"]) == tuple(
        CONFIRMATORY_TRUTHS
    )
    assert all("computationally_reproducible" in claims for claims in CONFIRMATORY_TRUTHS.values())
    assert all("digital_twin" not in claims for claims in CONFIRMATORY_TRUTHS.values())


def test_smoke_run_is_explicitly_ineligible(tmp_path):
    output = tmp_path / "summary.json"
    cases = tmp_path / "cases.npz"
    run(
        output=output,
        markdown=tmp_path / "summary.md",
        cases_path=cases,
        test_mode=True,
        direction_function=_fake_direction,
    )
    payload = json.loads(output.read_text())
    archive = np.load(cases)
    assert payload["num_cases"] == 10
    assert payload["scale_matches_frozen_protocol"] is False
    assert payload["primary_endpoint"]["passed"] is False
    assert payload["confirmatory_model_refitting_performed"] is False
    assert archive["labels"].shape == (10, 10)
    assert archive["policy_decisions"].shape == (7, 10, 10)
    assert next(
        row
        for row in payload["aggregate_by_policy"]
        if row["policy"] == "constrained_selective_hybrid"
    )["semantic_support_veto_violations"] == 0
