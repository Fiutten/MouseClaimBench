from mousebrainbench.benchmarks.semantic_risk_shift_sweep_v5_2 import (
    _limits,
    paired_bundle_transition,
)


def test_sweep_uses_bonferroni_adjusted_confidence() -> None:
    protocol = {
        "simultaneous_contract": {
            "target_seed_bundle_failure_probability": 0.10,
            "minimum_authorized_seed_bundle_coverage": 0.10,
            "minimum_positive_recovery": 0.05,
            "minimum_independent_units": 29,
            "per_level_confidence": 1.0 - 0.05 / 6,
        }
    }
    assert _limits(protocol)["confidence"] == 1.0 - 0.05 / 6


def test_paired_transition_preserves_shared_seed_bundle_unit() -> None:
    before = {
        "seed-1": {"authorized": True, "failed": False},
        "seed-2": {"authorized": False, "failed": False},
    }
    after = {
        "seed-1": {"authorized": False, "failed": False},
        "seed-2": {"authorized": True, "failed": True},
    }
    result = paired_bundle_transition(before, after)
    assert result["shared_seed_bundles"] == 2
    assert result["authorization"]["on_to_off"] == 1
    assert result["authorization"]["off_to_on"] == 1
    assert result["failure"]["off_to_on"] == 1
