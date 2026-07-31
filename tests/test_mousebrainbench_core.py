"""End-to-end invariants for the MouseBrainBench Phase 1 core."""

from __future__ import annotations

import json

import numpy as np

from mousebrainbench.cli import execute
from mousebrainbench.config import load_config
from mousebrainbench.connectivity.graph_builder import build_directed_graph
from mousebrainbench.connectivity.normalization import Normalization, normalize_connectivity
from mousebrainbench.connectivity.synthetic import generate_synthetic_connectivity
from mousebrainbench.dynamics.linear_rate import LinearRateModel
from mousebrainbench.dynamics.wilson_cowan import WilsonCowanModel
from mousebrainbench.data.loaders import load_connectivity_npz, save_connectivity_npz
from mousebrainbench.schemas import BrainRegion, ConnectivityMatrix
from mousebrainbench.simulation.perturbations import Perturbation, PerturbationType
from mousebrainbench.simulation.runner import SimulationConfig, run_simulation
from mousebrainbench.simulation.stimuli import RegionalStimulus
from mousebrainbench.validation.functional_metrics import functional_connectivity


def test_schema_rejects_wrong_orientation_shape() -> None:
    with np.testing.assert_raises(ValueError):
        ConnectivityMatrix((1, 2), (1,), np.ones((2, 1)))


def test_graph_respects_target_source_orientation() -> None:
    regions = [BrainRegion(1, "A", "A"), BrainRegion(2, "B", "B")]
    connectivity = ConnectivityMatrix((1, 2), (1, 2), np.array([[0.0, 0.0], [0.7, 0.0]]))
    graph = build_directed_graph(regions, connectivity)
    assert graph.has_edge(1, 2)
    assert not graph.has_edge(2, 1)


def test_normalization_does_not_mutate_input() -> None:
    _, connectivity = generate_synthetic_connectivity(20, seed=3)
    original = connectivity.dense_weights()
    normalized = normalize_connectivity(connectivity, Normalization.SPECTRAL_RADIUS)
    np.testing.assert_array_equal(connectivity.dense_weights(), original)
    assert np.isclose(max(abs(np.linalg.eigvals(normalized.dense_weights()))), 1.0)


def test_native_connectivity_round_trip(tmp_path) -> None:
    _, connectivity = generate_synthetic_connectivity(20, seed=3)
    restored = load_connectivity_npz(save_connectivity_npz(connectivity, tmp_path / "matrix.npz"))
    np.testing.assert_array_equal(restored.dense_weights(), connectivity.dense_weights())
    assert restored.metadata == connectivity.metadata


def test_models_are_reproducible_and_finite() -> None:
    _, connectivity = generate_synthetic_connectivity(20, seed=4)
    connectivity = normalize_connectivity(connectivity, "spectral_radius")
    config = SimulationConfig(dt=0.2, t_max=4.0, seed=9, noise_std=0.001)
    for model in (LinearRateModel(coupling=0.2), WilsonCowanModel(coupling=0.2)):
        first = run_simulation(model, connectivity, config)
        second = run_simulation(model, connectivity, config)
        np.testing.assert_array_equal(first.activity, second.activity)
        assert np.all(np.isfinite(first.activity))


def test_lesion_is_clamped_and_does_not_mutate_connectivity() -> None:
    _, connectivity = generate_synthetic_connectivity(10, seed=5)
    original = connectivity.dense_weights()
    lesion = Perturbation(PerturbationType.LESION, (1,), onset=1.0, duration=2.0)
    state = run_simulation(
        LinearRateModel(coupling=0.2),
        connectivity,
        SimulationConfig(dt=0.1, t_max=4.0, seed=2),
        stimuli=(RegionalStimulus((1,), onset=0.0, duration=4.0, amplitude=0.2),),
        perturbations=(lesion,),
    )
    active = (state.time >= 1.0) & (state.time < 3.0)
    np.testing.assert_array_equal(state.activity[active, 1], 0.0)
    np.testing.assert_array_equal(connectivity.dense_weights(), original)


def test_functional_connectivity_has_no_nan() -> None:
    activity = np.ones((20, 4))
    assert np.all(np.isfinite(functional_connectivity(activity)))


def test_cli_creates_complete_reproducible_artifact(tmp_path) -> None:
    config = load_config("configs/default.yaml")
    config["run"]["output_dir"] = str(tmp_path)
    first = execute(config)
    first_results = np.load(first / "results.npz")["perturbed_activity"].copy()
    second = execute(config)
    second_results = np.load(second / "results.npz")["perturbed_activity"].copy()
    np.testing.assert_array_equal(first_results, second_results)
    for filename in ("config.yaml", "provenance.json", "metrics.json", "results.npz"):
        assert (first / filename).exists()
    metrics = json.loads((first / "metrics.json").read_text())
    assert metrics["interpretation"] == "engineering_validation_only"
