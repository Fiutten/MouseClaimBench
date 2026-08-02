import numpy as np

from mousebrainbench.benchmarks.timegraph_v5_1_topology_confirmation import (
    _frozen_v3_threshold,
    _limits,
)


def test_claim_specific_limits_are_read_from_frozen_contract() -> None:
    protocol = {
        "confirmatory_population": {
            "inferential_contract": {
                "target_seed_bundle_failure_probability": 0.10,
                "minimum_authorized_seed_bundle_coverage": 0.10,
                "minimum_positive_recovery": 0.05,
                "minimum_independent_units": 29,
                "confidence_level": 0.95,
            }
        }
    }
    assert _limits(protocol) == {
        "target_risk": 0.10,
        "minimum_coverage": 0.10,
        "minimum_positive_recovery": 0.05,
        "minimum_independent_units": 29,
        "confidence": 0.95,
    }


def test_frozen_v3_threshold_is_selected_by_claim_name() -> None:
    policy = {
        "semantic_policy": {
            "certificates": [
                {"claim": "predictive", "threshold": 0.1},
                {"claim": "topology_specific", "threshold": np.float64(0.22)},
            ]
        }
    }
    assert _frozen_v3_threshold(policy, "topology_specific") == 0.22
