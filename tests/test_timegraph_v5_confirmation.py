import json
from pathlib import Path

import numpy as np
import pytest
import yaml

from mousebrainbench.benchmarks.timegraph_v5_confirmation import (
    _derived_seed,
    _load_official_source,
    _observed_edges,
    _role_data,
    _select_pairs,
)


def test_seed_derivation_is_deterministic_and_scenario_specific() -> None:
    first = _derived_seed(17, "a1", "gaussian", 2)
    assert first == _derived_seed(17, "a1", "gaussian", 2)
    assert first != _derived_seed(17, "a1c", "gaussian", 2)
    assert 0 <= first < 2**32


def test_pair_selection_is_balanced_and_deterministic() -> None:
    pairs = (
        ("X1", "X2", True),
        ("X2", "X3", True),
        ("X3", "X1", False),
        ("X1", "X3", False),
    )
    selected = _select_pairs(pairs, namespace="frozen", direct_count=1, control_count=1)
    assert selected == _select_pairs(
        pairs, namespace="frozen", direct_count=1, control_count=1
    )
    assert sum(item[2] for item in selected) == 1
    assert len(selected) == 2


def test_official_source_hash_and_generator_smoke() -> None:
    protocol = yaml.safe_load(
        Path("configs/benchmarks/semantic_risk_control_v5.yaml").read_text()
    )
    source = Path("data/external/timegraph_v5/Codes/a1.py")
    if not source.exists():
        pytest.skip("optional pinned TimeGraph source is not installed")
    expected = protocol["confirmatory_population"]["source_files"]["Codes/a1.py"]
    module = _load_official_source(source, expected_hash=expected)
    generator = module["LinearTimeSeriesGenerator"](random_state=7)
    frame = generator.generate_multivariate_ts(100, 6, 2)
    assert frame.shape == (100, 7)
    assert np.isfinite(frame.to_numpy()).all()
    assert _observed_edges(module, 6, 2)


def test_full_official_adapter_smoke() -> None:
    source_root = Path("data/external/timegraph_v5")
    if not source_root.exists():
        pytest.skip("optional pinned TimeGraph source is not installed")
    protocol = yaml.safe_load(
        Path("configs/benchmarks/semantic_risk_control_v5.yaml").read_text()
    )
    protocol["confirmatory_population"]["split_seeds"]["target_calibration"] = {
        "start": 2026085999,
        "count": 1,
    }
    protocol["confirmatory_population"]["factors"]["n_points"] = 200
    score_model = json.loads(Path("results/hybrid_selective_policy/model.json").read_text())
    risk_policy = json.loads(Path("results/semantic_risk_policy/model.json").read_text())
    data = _role_data(
        "target_calibration",
        protocol=protocol,
        source_root=source_root,
        score_model=score_model,
        variable_claims=tuple(risk_policy["variable_claims"]),
    )
    assert data.generated_scenarios == 8
    assert len(data.records) == 16
    assert len(np.unique(data.top_level_ids)) == 1
    assert data.scores.shape == (16, 6)


def test_paired_namespace_keeps_pair_identity_across_shift_roles() -> None:
    source_root = Path("data/external/timegraph_v5")
    if not source_root.exists():
        pytest.skip("optional pinned TimeGraph source is not installed")
    protocol = yaml.safe_load(
        Path("configs/benchmarks/semantic_risk_shift_sweep_v5_2.yaml").read_text()
    )
    for role in ("shift_0", "shift_1"):
        protocol["confirmatory_population"]["split_seeds"][role] = {
            "start": 2026087999,
            "count": 1,
        }
        protocol["confirmatory_population"]["role_factor_overrides"][role] = {
            "n_points": 200,
            "noise_scale": 0.1,
        }
    score_model = json.loads(Path("results/hybrid_selective_policy/model.json").read_text())
    first = _role_data(
        "shift_0",
        protocol=protocol,
        source_root=source_root,
        score_model=score_model,
        variable_claims=("topology_specific",),
    )
    second = _role_data(
        "shift_1",
        protocol=protocol,
        source_root=source_root,
        score_model=score_model,
        variable_claims=("topology_specific",),
    )
    first_pairs = [(row["source"], row["target"], row["direct_edge"]) for row in first.records]
    second_pairs = [(row["source"], row["target"], row["direct_edge"]) for row in second.records]
    assert first_pairs == second_pairs
