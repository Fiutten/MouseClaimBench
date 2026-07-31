import json

import numpy as np

from mousebrainbench.benchmarks.sensorium_predictive_mis import run_sensorium_benchmark
from mousebrainbench.benchmarks.sensorium_synthetic_smoke import run
from mousebrainbench.data.loaders.sensorium import SensoriumTrialTable


def test_sensorium_smoke_separates_prediction_from_mechanism(tmp_path) -> None:
    output = run(tmp_path / "sensorium_smoke.json")
    payload = json.loads(output.read_text())
    blocks = {block["name"]: block for block in payload["mis"]["blocks"]}

    assert payload["metrics"]["ridge_correlation"] > 0.9
    assert payload["metrics"]["ridge_minus_mean"] > 0.2
    assert blocks["reproducibility"]["passed"]
    assert blocks["topology_specificity"]["passed"]
    assert not blocks["directed_identifiability"]["passed"]
    assert payload["decision"] == "predictive_signal_requires_extra_mechanistic_evidence"


def test_sensorium_benchmark_marks_reliability_not_estimable_without_repeats(tmp_path) -> None:
    table = SensoriumTrialTable(
        root=tmp_path,
        modality="dynamic",
        stimulus_features=np.arange(20, dtype=float).reshape(4, 5),
        context_features=np.empty((4, 0), dtype=float),
        responses=np.arange(12, dtype=float).reshape(4, 3),
        tiers=np.asarray(["train", "train", "oracle", "oracle"]),
        trial_ids=np.arange(4),
        stimulus_ids=np.arange(4),
        neuron_ids=np.arange(3),
    )

    output = run_sensorium_benchmark(
        table, output=tmp_path / "no_repeats.json", eval_tiers=("oracle",)
    )
    payload = json.loads(output.read_text())

    assert payload["metrics"]["reliability"] == 0.0
    assert not payload["metrics"]["reliability_estimable"]
    assert payload["metrics"]["reliability_repeated_stimuli"] == 0.0


def test_calibrated_residual_adapter_can_improve_over_mean(tmp_path) -> None:
    rng = np.random.default_rng(123)
    stimulus_features = rng.normal(size=(36, 6))
    weights = rng.normal(size=(6, 4))
    baseline = np.asarray([[3.0, 4.0, 5.0, 6.0]])
    responses = baseline + 0.35 * stimulus_features @ weights
    responses += rng.normal(scale=0.02, size=responses.shape)
    tiers = np.asarray(["train"] * 28 + ["oracle"] * 8)
    stimulus_ids = np.concatenate([np.arange(28), np.repeat([28, 29, 30, 31], 2)])
    table = SensoriumTrialTable(
        root=tmp_path,
        modality="dynamic",
        stimulus_features=stimulus_features,
        context_features=np.empty((36, 0), dtype=float),
        responses=responses,
        tiers=tiers,
        trial_ids=np.arange(36),
        stimulus_ids=stimulus_ids,
        neuron_ids=np.arange(4),
    )

    output = run_sensorium_benchmark(
        table,
        output=tmp_path / "calibrated_adapter.json",
        eval_tiers=("oracle",),
        adapter="calibrated_residual_ridge",
    )
    payload = json.loads(output.read_text())

    assert payload["metrics"]["calibrated_residual_ridge_minus_mean"] > 0.05
    assert payload["adapter"]["diagnostics"]["adapter_beta"] > 0.0
    assert "temporal_svd_residual_ridge_correlation" not in payload["metrics"]
    assert "random_feature_residual_ridge_correlation" not in payload["metrics"]
    assert "temporal_svd_residual_ridge" not in payload["baselines"]
    assert "random_feature_residual_ridge" not in payload["baselines"]


def test_temporal_svd_adapter_can_improve_over_mean(tmp_path) -> None:
    rng = np.random.default_rng(321)
    latent = rng.normal(size=(48, 5))
    mixing = rng.normal(size=(5, 40))
    stimulus_features = latent @ mixing + rng.normal(scale=0.03, size=(48, 40))
    neural_weights = rng.normal(size=(5, 6))
    baseline = np.asarray([[2.0, 2.5, 3.0, 3.5, 4.0, 4.5]])
    responses = baseline + 0.45 * latent @ neural_weights
    responses += rng.normal(scale=0.02, size=responses.shape)
    tiers = np.asarray(["train"] * 36 + ["oracle"] * 12)
    stimulus_ids = np.concatenate([np.arange(36), np.repeat(np.arange(36, 42), 2)])
    table = SensoriumTrialTable(
        root=tmp_path,
        modality="dynamic",
        stimulus_features=stimulus_features,
        context_features=np.empty((48, 0), dtype=float),
        responses=responses,
        tiers=tiers,
        trial_ids=np.arange(48),
        stimulus_ids=stimulus_ids,
        neuron_ids=np.arange(6),
        feature_mode="temporal_filterbank",
    )

    output = run_sensorium_benchmark(
        table,
        output=tmp_path / "temporal_svd_adapter.json",
        eval_tiers=("oracle",),
        adapter="temporal_svd_residual_ridge",
        adapter_alpha_grid=(0.1, 1.0, 10.0),
    )
    payload = json.loads(output.read_text())

    assert payload["metrics"]["temporal_svd_residual_ridge_minus_mean"] > 0.05
    assert payload["adapter"]["diagnostics"]["adapter_components"] > 0.0


def test_random_feature_adapter_can_capture_nonlinear_signal(tmp_path) -> None:
    rng = np.random.default_rng(456)
    stimulus_features = rng.normal(size=(54, 8))
    nonlinear = np.column_stack(
        [
            np.sin(stimulus_features[:, 0]),
            stimulus_features[:, 1] * stimulus_features[:, 2],
            np.cos(stimulus_features[:, 3]),
        ]
    )
    weights = rng.normal(size=(3, 5))
    baseline = np.asarray([[1.0, 1.5, 2.0, 2.5, 3.0]])
    responses = baseline + 0.65 * nonlinear @ weights
    responses += rng.normal(scale=0.02, size=responses.shape)
    tiers = np.asarray(["train"] * 42 + ["oracle"] * 12)
    table = SensoriumTrialTable(
        root=tmp_path,
        modality="dynamic",
        stimulus_features=stimulus_features,
        context_features=np.empty((54, 0), dtype=float),
        responses=responses,
        tiers=tiers,
        trial_ids=np.arange(54),
        stimulus_ids=np.arange(54),
        neuron_ids=np.arange(5),
    )

    output = run_sensorium_benchmark(
        table,
        output=tmp_path / "random_feature_adapter.json",
        eval_tiers=("oracle",),
        adapter="random_feature_residual_ridge",
        adapter_alpha_grid=(0.1, 1.0, 10.0),
    )
    payload = json.loads(output.read_text())

    assert payload["metrics"]["random_feature_residual_ridge_minus_mean"] > 0.05
    assert payload["adapter"]["diagnostics"]["adapter_components"] > 0.0
    assert payload["adapter"]["diagnostics"]["adapter_gamma"] > 0.0


def test_sensorium_benchmark_records_ood_gate(tmp_path) -> None:
    table = SensoriumTrialTable(
        root=tmp_path,
        modality="dynamic",
        stimulus_features=np.arange(40, dtype=float).reshape(8, 5),
        context_features=np.empty((8, 0), dtype=float),
        responses=np.arange(24, dtype=float).reshape(8, 3),
        tiers=np.asarray(["train"] * 4 + ["live_test_main"] * 4),
        trial_ids=np.arange(8),
        stimulus_ids=np.arange(8),
        neuron_ids=np.arange(3),
    )

    output = run_sensorium_benchmark(
        table,
        output=tmp_path / "ood_gate.json",
        eval_tiers=("live_test_main",),
        has_ood_generalization_gate=True,
    )
    payload = json.loads(output.read_text())

    assert payload["metrics"]["has_ood_generalization_gate"]
    assert payload["dataset"]["eval_tiers"] == ["live_test_main"]
