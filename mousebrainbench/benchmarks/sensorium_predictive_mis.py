"""Sensorium-style prediction benchmark with mechanistic-identifiability gates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from mousebrainbench import __version__
from mousebrainbench.artifacts import code_revision
from mousebrainbench.data.loaders.sensorium import SensoriumTrialTable, load_sensorium_directory
from mousebrainbench.surrogate.sensorium_adapters import (
    calibrated_residual_ridge_adapter,
    random_feature_residual_ridge_adapter,
    temporal_svd_residual_ridge_adapter,
)
from mousebrainbench.validation.mechanistic_identifiability import (
    Criterion,
    MechanisticIdentifiabilityScore,
    build_mis_from_blocks,
)


DEFAULT_OUTPUT = Path("results/sensorium_predictive_mis_benchmark.json")


def _safe_correlation(left: np.ndarray, right: np.ndarray) -> float:
    left_flat = np.asarray(left, dtype=float).ravel()
    right_flat = np.asarray(right, dtype=float).ravel()
    if left_flat.size != right_flat.size:
        raise ValueError("correlation inputs must have equal size")
    if np.std(left_flat) == 0 or np.std(right_flat) == 0:
        return 0.0
    return float(np.corrcoef(left_flat, right_flat)[0, 1])


def _median_neuron_correlation(prediction: np.ndarray, response: np.ndarray) -> float:
    values = []
    for neuron in range(response.shape[1]):
        values.append(_safe_correlation(prediction[:, neuron], response[:, neuron]))
    return float(np.median(values))


def _standardize_train_test(
    train: np.ndarray, test: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    mean = train.mean(axis=0, keepdims=True)
    std = train.std(axis=0, keepdims=True)
    std[std == 0] = 1.0
    return (train - mean) / std, (test - mean) / std, mean, std


def _ridge_fit_predict(
    train_x: np.ndarray,
    train_y: np.ndarray,
    test_x: np.ndarray,
    *,
    alpha: float,
) -> np.ndarray:
    train_x_std, test_x_std, _, _ = _standardize_train_test(train_x, test_x)
    design = np.column_stack([np.ones(len(train_x_std)), train_x_std])
    test_design = np.column_stack([np.ones(len(test_x_std)), test_x_std])
    penalty = np.eye(design.shape[1]) * alpha
    penalty[0, 0] = 0.0
    weights = np.linalg.solve(design.T @ design + penalty, design.T @ train_y)
    return test_design @ weights


def _tier_mask(tiers: np.ndarray, names: tuple[str, ...]) -> np.ndarray:
    return np.asarray([tier in names for tier in tiers], dtype=bool)


def _response_reliability(table: SensoriumTrialTable, eval_mask: np.ndarray) -> tuple[float, int]:
    """Estimate repeat reliability from repeated stimulus ids in held-out tiers."""

    correlations = []
    stimulus_ids = table.stimulus_ids[eval_mask]
    responses = table.responses[eval_mask]
    for stimulus_id in sorted(set(stimulus_ids.tolist())):
        indices = np.flatnonzero(stimulus_ids == stimulus_id)
        if len(indices) < 2:
            continue
        half = len(indices) // 2
        first = responses[indices[:half]].mean(axis=0)
        second = responses[indices[half:]].mean(axis=0)
        correlations.append(_safe_correlation(first, second))
    return (float(np.median(correlations)) if correlations else 0.0, len(correlations))


def predictive_mis_from_metrics(metrics: dict[str, float | bool]) -> MechanisticIdentifiabilityScore:
    """Build a Sensorium-oriented MIS without overstating mechanism.

    The first two blocks ask whether the neural target is reproducible and
    stimulus-specific. The final block is intentionally strict: predictive
    performance alone is not treated as mechanistic identifiability.
    """

    return build_mis_from_blocks(
        reproducibility=(
            Criterion("heldout_repeat_response_reliability", float(metrics["reliability"]), 0.30),
        ),
        topology_specificity=(
            Criterion(
                "best_predictive_minus_mean_correlation",
                float(metrics["best_predictive_minus_mean"]),
                0.05,
            ),
            Criterion(
                "best_predictive_minus_scrambled_correlation",
                float(metrics["best_predictive_minus_scrambled"]),
                0.05,
            ),
            Criterion(
                "best_predictive_test_correlation",
                float(metrics["best_predictive_correlation"]),
                0.10,
            ),
        ),
        directed_identifiability=(
            Criterion(
                "has_structural_or_interventional_constraint",
                1.0 if bool(metrics["has_structural_or_interventional_constraint"]) else 0.0,
                1.0,
                "gte",
            ),
            Criterion(
                "has_ood_generalization_gate",
                1.0 if bool(metrics["has_ood_generalization_gate"]) else 0.0,
                1.0,
                "gte",
            ),
        ),
    )


def run_sensorium_benchmark(
    table: SensoriumTrialTable,
    *,
    output: str | Path = DEFAULT_OUTPUT,
    alpha: float = 1.0,
    seed: int = 17,
    eval_tiers: tuple[str, ...] = ("validation", "val"),
    git_revision: str | None = None,
    has_structural_or_interventional_constraint: bool = False,
    has_ood_generalization_gate: bool = False,
    adapter: str = "ridge",
    adapter_alpha: float = 10.0,
    adapter_alpha_grid: tuple[float, ...] | None = None,
) -> Path:
    """Evaluate transparent predictive baselines and write MIS diagnostics."""

    train_mask = _tier_mask(table.tiers, ("train",))
    eval_mask = _tier_mask(table.tiers, eval_tiers)
    if not np.any(train_mask) or not np.any(eval_mask):
        raise ValueError("Sensorium benchmark requires train and held-out tiers")
    train_x = table.stimulus_features[train_mask]
    train_y = table.responses[train_mask]
    eval_x = table.stimulus_features[eval_mask]
    eval_y = table.responses[eval_mask]

    mean_prediction = np.repeat(train_y.mean(axis=0, keepdims=True), len(eval_y), axis=0)
    ridge_prediction = _ridge_fit_predict(train_x, train_y, eval_x, alpha=alpha)
    rng = np.random.default_rng(seed)
    scrambled_prediction = _ridge_fit_predict(
        rng.permutation(train_x), train_y, eval_x, alpha=alpha
    )
    context_available = bool(
        table.context_features.shape[1] > 0
        and np.any(np.std(table.context_features[train_mask], axis=0) > 0)
        and np.any(np.std(table.context_features[eval_mask], axis=0) > 0)
    )
    if context_available:
        train_context_x = np.column_stack([train_x, table.context_features[train_mask]])
        eval_context_x = np.column_stack([eval_x, table.context_features[eval_mask]])
        context_prediction = _ridge_fit_predict(train_context_x, train_y, eval_context_x, alpha=alpha)
        scrambled_context_prediction = _ridge_fit_predict(
            np.column_stack([rng.permutation(train_x), table.context_features[train_mask]]),
            train_y,
            eval_context_x,
            alpha=alpha,
        )
        context_correlation = _safe_correlation(context_prediction, eval_y)
        scrambled_context_correlation = _safe_correlation(scrambled_context_prediction, eval_y)
        context_median_neuron = _median_neuron_correlation(context_prediction, eval_y)
    else:
        context_correlation = 0.0
        scrambled_context_correlation = 0.0
        context_median_neuron = 0.0

    stimulus_correlation = _safe_correlation(ridge_prediction, eval_y)
    stimulus_scrambled_correlation = _safe_correlation(scrambled_prediction, eval_y)
    if context_available and context_correlation > stimulus_correlation:
        best_model = "stimulus_context_ridge"
        best_correlation = context_correlation
        best_scrambled = scrambled_context_correlation
    else:
        best_model = "stimulus_ridge"
        best_correlation = stimulus_correlation
        best_scrambled = stimulus_scrambled_correlation

    adapter_diagnostics: dict[str, float] = {}
    adapter_correlation = 0.0
    adapter_scrambled_correlation = 0.0
    adapter_median_neuron = 0.0
    temporal_svd_correlation = 0.0
    temporal_svd_scrambled_correlation = 0.0
    temporal_svd_median_neuron = 0.0
    random_feature_correlation = 0.0
    random_feature_scrambled_correlation = 0.0
    random_feature_median_neuron = 0.0
    if adapter == "calibrated_residual_ridge":
        adapter_result = calibrated_residual_ridge_adapter(
            train_x,
            train_y,
            eval_x,
            alpha=adapter_alpha,
            alpha_grid=adapter_alpha_grid,
            seed=seed,
        )
        adapter_diagnostics = adapter_result.diagnostics
        adapter_correlation = _safe_correlation(adapter_result.prediction, eval_y)
        adapter_scrambled_correlation = _safe_correlation(
            adapter_result.scrambled_prediction, eval_y
        )
        adapter_median_neuron = _median_neuron_correlation(adapter_result.prediction, eval_y)
        if adapter_correlation > best_correlation:
            best_model = adapter_result.name
            best_correlation = adapter_correlation
            best_scrambled = adapter_scrambled_correlation
    elif adapter == "temporal_svd_residual_ridge":
        adapter_result = temporal_svd_residual_ridge_adapter(
            train_x,
            train_y,
            eval_x,
            alpha_grid=adapter_alpha_grid,
            seed=seed,
        )
        adapter_diagnostics = adapter_result.diagnostics
        temporal_svd_correlation = _safe_correlation(adapter_result.prediction, eval_y)
        temporal_svd_scrambled_correlation = _safe_correlation(
            adapter_result.scrambled_prediction, eval_y
        )
        temporal_svd_median_neuron = _median_neuron_correlation(
            adapter_result.prediction, eval_y
        )
        if temporal_svd_correlation > best_correlation:
            best_model = adapter_result.name
            best_correlation = temporal_svd_correlation
            best_scrambled = temporal_svd_scrambled_correlation
    elif adapter == "random_feature_residual_ridge":
        adapter_result = random_feature_residual_ridge_adapter(
            train_x,
            train_y,
            eval_x,
            alpha_grid=adapter_alpha_grid,
            seed=seed,
        )
        adapter_diagnostics = adapter_result.diagnostics
        random_feature_correlation = _safe_correlation(adapter_result.prediction, eval_y)
        random_feature_scrambled_correlation = _safe_correlation(
            adapter_result.scrambled_prediction, eval_y
        )
        random_feature_median_neuron = _median_neuron_correlation(
            adapter_result.prediction, eval_y
        )
        if random_feature_correlation > best_correlation:
            best_model = adapter_result.name
            best_correlation = random_feature_correlation
            best_scrambled = random_feature_scrambled_correlation
    elif adapter != "ridge":
        raise ValueError(f"unknown Sensorium adapter: {adapter}")

    reliability, repeated_stimuli = _response_reliability(table, eval_mask)
    metrics: dict[str, Any] = {
        "mean_correlation": _safe_correlation(mean_prediction, eval_y),
        "ridge_correlation": stimulus_correlation,
        "scrambled_correlation": stimulus_scrambled_correlation,
        "ridge_median_neuron_correlation": _median_neuron_correlation(ridge_prediction, eval_y),
        "context_available": context_available,
        "stimulus_context_ridge_correlation": context_correlation,
        "stimulus_context_scrambled_correlation": scrambled_context_correlation,
        "stimulus_context_ridge_median_neuron_correlation": context_median_neuron,
        "best_model_is_contextual": best_model == "stimulus_context_ridge",
        "best_model": best_model,
        "best_predictive_correlation": best_correlation,
        "best_predictive_scrambled_correlation": best_scrambled,
        "reliability": reliability,
        "reliability_estimable": repeated_stimuli > 0,
        "reliability_repeated_stimuli": float(repeated_stimuli),
        "has_structural_or_interventional_constraint": has_structural_or_interventional_constraint,
        "has_ood_generalization_gate": has_ood_generalization_gate,
    }
    metrics["ridge_minus_mean"] = float(metrics["ridge_correlation"]) - float(
        metrics["mean_correlation"]
    )
    metrics["ridge_minus_scrambled"] = float(metrics["ridge_correlation"]) - float(
        metrics["scrambled_correlation"]
    )
    metrics["stimulus_context_ridge_minus_mean"] = float(
        metrics["stimulus_context_ridge_correlation"]
    ) - float(metrics["mean_correlation"])
    metrics["stimulus_context_ridge_minus_scrambled"] = float(
        metrics["stimulus_context_ridge_correlation"]
    ) - float(metrics["stimulus_context_scrambled_correlation"])
    adapter_metrics = {
        "calibrated_residual_ridge": (
            adapter_correlation,
            adapter_scrambled_correlation,
            adapter_median_neuron,
        ),
        "temporal_svd_residual_ridge": (
            temporal_svd_correlation,
            temporal_svd_scrambled_correlation,
            temporal_svd_median_neuron,
        ),
        "random_feature_residual_ridge": (
            random_feature_correlation,
            random_feature_scrambled_correlation,
            random_feature_median_neuron,
        ),
    }
    if adapter in adapter_metrics:
        correlation, scrambled_correlation, median_neuron_correlation = adapter_metrics[adapter]
        metrics[f"{adapter}_correlation"] = correlation
        metrics[f"{adapter}_scrambled_correlation"] = scrambled_correlation
        metrics[f"{adapter}_median_neuron_correlation"] = median_neuron_correlation
        metrics[f"{adapter}_minus_mean"] = correlation - float(metrics["mean_correlation"])
        metrics[f"{adapter}_minus_scrambled"] = correlation - scrambled_correlation
    metrics["best_predictive_minus_mean"] = float(metrics["best_predictive_correlation"]) - float(
        metrics["mean_correlation"]
    )
    metrics["best_predictive_minus_scrambled"] = float(
        metrics["best_predictive_correlation"]
    ) - float(metrics["best_predictive_scrambled_correlation"])
    mis = predictive_mis_from_metrics(metrics)
    baselines = {
        "mean_response": "train-set neuron mean",
        "stimulus_ridge": "ridge regression on deterministic stimulus descriptors",
        "stimulus_context_ridge": "stimulus ridge plus behavior and pupil covariates when available",
        "scrambled_stimulus_ridge": "same model with permuted train stimuli",
        "scrambled_stimulus_context_ridge": (
            "context-preserving control with permuted train stimuli"
        ),
    }
    adapter_descriptions = {
        "calibrated_residual_ridge": (
            "train-calibrated mean response plus shrinkage-scaled stimulus residual"
        ),
        "temporal_svd_residual_ridge": (
            "train-only SVD temporal subspace plus calibrated residual ridge"
        ),
        "random_feature_residual_ridge": (
            "train-only random Fourier feature map plus calibrated residual ridge"
        ),
    }
    if adapter in adapter_descriptions:
        baselines[adapter] = adapter_descriptions[adapter]

    payload: dict[str, Any] = {
        "version": __version__,
        "git_revision": git_revision or code_revision(),
        "interpretation": "sensorium_predictive_case_not_mechanistic_claim",
        "dataset": {
            "root": str(table.root),
            "modality": table.modality,
            "feature_mode": table.feature_mode,
            "n_trials": table.n_trials,
            "n_neurons": table.n_neurons,
            "eval_tiers": list(eval_tiers),
            "tiers": {tier: int(np.sum(table.tiers == tier)) for tier in sorted(set(table.tiers))},
        },
        "baselines": baselines,
        "adapter": {"name": adapter, "diagnostics": adapter_diagnostics},
        "metrics": metrics,
        "mis": mis.as_dict(),
        "decision": (
            "predictive_and_mechanistically_identifiable"
            if mis.passed
            else "predictive_signal_requires_extra_mechanistic_evidence"
        ),
        "limitations": [
            "The baseline is intentionally transparent and not a Sensorium SOTA model.",
            "Prediction from images/videos is not by itself mechanistic explanation.",
            "A mechanistic claim requires structure, perturbation, OOD, or causal constraints.",
            "Reliability is only estimable when held-out tiers contain repeated stimulus ids.",
        ],
    }
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))
    return path


def run(
    root: str | Path,
    *,
    output: str | Path = DEFAULT_OUTPUT,
    modality: str | None = None,
    max_trials: int | None = None,
    feature_mode: str = "summary",
    alpha: float = 1.0,
    eval_tiers: tuple[str, ...] = ("validation", "val"),
    git_revision: str | None = None,
    adapter: str = "ridge",
    adapter_alpha: float = 10.0,
    adapter_alpha_grid: tuple[float, ...] | None = None,
    has_structural_or_interventional_constraint: bool = False,
    has_ood_generalization_gate: bool = False,
) -> Path:
    table = load_sensorium_directory(
        root, modality=modality, max_trials=max_trials, feature_mode=feature_mode
    )
    return run_sensorium_benchmark(
        table,
        output=output,
        alpha=alpha,
        eval_tiers=eval_tiers,
        git_revision=git_revision,
        adapter=adapter,
        adapter_alpha=adapter_alpha,
        adapter_alpha_grid=adapter_alpha_grid,
        has_structural_or_interventional_constraint=has_structural_or_interventional_constraint,
        has_ood_generalization_gate=has_ood_generalization_gate,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--modality", choices=("static", "dynamic"), default=None)
    parser.add_argument("--max-trials", type=int, default=None)
    parser.add_argument(
        "--feature-mode",
        choices=("summary", "temporal_filterbank"),
        default="summary",
        help="Stimulus descriptor family. temporal_filterbank keeps coarse video timing.",
    )
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument(
        "--adapter",
        choices=(
            "ridge",
            "calibrated_residual_ridge",
            "temporal_svd_residual_ridge",
            "random_feature_residual_ridge",
        ),
        default="ridge",
        help="Optional predictive adapter. Defaults to the original transparent ridge baseline.",
    )
    parser.add_argument(
        "--adapter-alpha",
        type=float,
        default=10.0,
        help="Ridge alpha used by calibrated residual adapters.",
    )
    parser.add_argument(
        "--adapter-alpha-grid",
        default=None,
        help="Comma-separated alpha candidates selected by train-only CV for calibrated adapters.",
    )
    parser.add_argument(
        "--eval-tier",
        action="append",
        dest="eval_tiers",
        default=None,
        help="Held-out tier to evaluate; repeat for multiple tiers. Defaults to validation/val.",
    )
    parser.add_argument(
        "--git-revision",
        default=None,
        help="Override provenance revision when regenerating several tracked artifacts.",
    )
    parser.add_argument(
        "--has-structural-or-interventional-constraint",
        action="store_true",
        help="Mark MIS directed-identifiability structure/causal evidence as present.",
    )
    parser.add_argument(
        "--has-ood-generalization-gate",
        action="store_true",
        help="Mark that the held-out tier is an explicit OOD generalization gate.",
    )
    args = parser.parse_args()
    output = run(
        args.root,
        output=args.output,
        modality=args.modality,
        max_trials=args.max_trials,
        feature_mode=args.feature_mode,
        alpha=args.alpha,
        eval_tiers=tuple(args.eval_tiers) if args.eval_tiers else ("validation", "val"),
        git_revision=args.git_revision,
        adapter=args.adapter,
        adapter_alpha=args.adapter_alpha,
        adapter_alpha_grid=(
            tuple(float(value) for value in args.adapter_alpha_grid.split(","))
            if args.adapter_alpha_grid
            else None
        ),
        has_structural_or_interventional_constraint=(
            args.has_structural_or_interventional_constraint
        ),
        has_ood_generalization_gate=args.has_ood_generalization_gate,
    )
    print(json.dumps({"output": str(output.resolve())}))


if __name__ == "__main__":
    main()
