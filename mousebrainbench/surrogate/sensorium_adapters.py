"""Transparent Sensorium prediction adapters.

The adapters in this module are deliberately lightweight. They are not intended
to replace official Sensorium models; they provide reproducible baselines that
can be audited inside the MouseBrainBench/MIS framework.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SensoriumAdapterResult:
    """Predictions and diagnostics emitted by a Sensorium adapter."""

    name: str
    prediction: np.ndarray
    scrambled_prediction: np.ndarray
    diagnostics: dict[str, float]


def _safe_correlation(left: np.ndarray, right: np.ndarray) -> float:
    left_flat = np.asarray(left, dtype=float).ravel()
    right_flat = np.asarray(right, dtype=float).ravel()
    if left_flat.size != right_flat.size:
        raise ValueError("correlation inputs must have equal size")
    if np.std(left_flat) == 0 or np.std(right_flat) == 0:
        return 0.0
    return float(np.corrcoef(left_flat, right_flat)[0, 1])


def _standardize_train_test(
    train: np.ndarray, test: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    mean = train.mean(axis=0, keepdims=True)
    std = train.std(axis=0, keepdims=True)
    std[std == 0] = 1.0
    return (train - mean) / std, (test - mean) / std, mean, std


def ridge_fit_predict(
    train_x: np.ndarray,
    train_y: np.ndarray,
    test_x: np.ndarray,
    *,
    alpha: float,
) -> np.ndarray:
    """Fit closed-form ridge regression and predict on held-out features."""

    train_x_std, test_x_std, _, _ = _standardize_train_test(train_x, test_x)
    design = np.column_stack([np.ones(len(train_x_std)), train_x_std])
    test_design = np.column_stack([np.ones(len(test_x_std)), test_x_std])
    penalty = np.eye(design.shape[1]) * alpha
    penalty[0, 0] = 0.0
    weights = np.linalg.solve(design.T @ design + penalty, design.T @ train_y)
    return test_design @ weights


def calibrated_residual_ridge_adapter(
    train_x: np.ndarray,
    train_y: np.ndarray,
    eval_x: np.ndarray,
    *,
    alpha: float = 10.0,
    alpha_grid: tuple[float, ...] | None = None,
    seed: int = 17,
    folds: int = 5,
    beta_grid: np.ndarray | None = None,
) -> SensoriumAdapterResult:
    """Predict ``mean response + beta * stimulus residual`` with train-only calibration.

    Dynamic Sensorium has a strong population mean component. A plain ridge
    residual can overfit and degrade held-out correlation even when it contains
    stimulus information. This adapter estimates the residual shrinkage
    coefficient ``beta`` by cross-validation within the training split, then
    applies the calibrated residual to the held-out tier.
    """

    if folds < 2:
        raise ValueError("folds must be at least 2")
    if len(train_x) != len(train_y):
        raise ValueError("train_x and train_y must have the same number of trials")
    if beta_grid is None:
        beta_grid = np.linspace(0.0, 1.5, 61)
    alphas = tuple(float(value) for value in (alpha_grid or (alpha,)))

    indices = np.arange(len(train_x))
    rng = np.random.default_rng(seed)
    rng.shuffle(indices)
    fold_indices = [fold for fold in np.array_split(indices, min(folds, len(indices))) if len(fold)]
    cv_predictions: dict[tuple[float, float], np.ndarray] = {
        (float(alpha_value), float(beta)): np.zeros_like(train_y, dtype=float)
        for alpha_value in alphas
        for beta in beta_grid
    }

    for fold in fold_indices:
        fit_indices = np.setdiff1d(indices, fold, assume_unique=False)
        train_mean = train_y[fit_indices].mean(axis=0, keepdims=True)
        residual = train_y[fit_indices] - train_mean
        for alpha_value in alphas:
            residual_prediction = ridge_fit_predict(
                train_x[fit_indices], residual, train_x[fold], alpha=alpha_value
            )
            for beta in beta_grid:
                cv_predictions[(float(alpha_value), float(beta))][fold] = (
                    train_mean + float(beta) * residual_prediction
                )

    cv_scores = {
        params: _safe_correlation(prediction, train_y)
        for params, prediction in cv_predictions.items()
    }
    best_alpha, best_beta = max(cv_scores, key=cv_scores.get)

    full_mean = train_y.mean(axis=0, keepdims=True)
    full_residual = train_y - full_mean
    residual_prediction = ridge_fit_predict(train_x, full_residual, eval_x, alpha=best_alpha)
    prediction = np.repeat(full_mean, len(eval_x), axis=0) + best_beta * residual_prediction

    scrambled_train_x = rng.permutation(train_x)
    scrambled_residual = ridge_fit_predict(
        scrambled_train_x, full_residual, eval_x, alpha=best_alpha
    )
    scrambled_prediction = (
        np.repeat(full_mean, len(eval_x), axis=0) + best_beta * scrambled_residual
    )
    return SensoriumAdapterResult(
        name="calibrated_residual_ridge",
        prediction=prediction,
        scrambled_prediction=scrambled_prediction,
        diagnostics={
            "adapter_alpha": float(best_alpha),
            "adapter_beta": float(best_beta),
            "adapter_cv_correlation": float(cv_scores[(best_alpha, best_beta)]),
            "adapter_cv_folds": float(len(fold_indices)),
            "adapter_alpha_candidates": float(len(alphas)),
        },
    )


def _project_with_train_svd(
    train_x: np.ndarray, eval_x: np.ndarray, *, components: int
) -> tuple[np.ndarray, np.ndarray]:
    """Project features onto train-only right singular vectors.

    This is intentionally a small numerical building block rather than a new
    deep model. The projection is fit only on the training split, so held-out
    and OOD tiers cannot influence the temporal subspace.
    """

    train_x_std, eval_x_std, _, _ = _standardize_train_test(train_x, eval_x)
    _, _, vt = np.linalg.svd(train_x_std, full_matrices=False)
    width = max(1, min(int(components), vt.shape[0]))
    basis = vt[:width].T
    return train_x_std @ basis, eval_x_std @ basis


def temporal_svd_residual_ridge_adapter(
    train_x: np.ndarray,
    train_y: np.ndarray,
    eval_x: np.ndarray,
    *,
    alpha_grid: tuple[float, ...] | None = None,
    component_grid: tuple[int, ...] = (8, 16, 32, 64),
    seed: int = 17,
    folds: int = 5,
    beta_grid: np.ndarray | None = None,
) -> SensoriumAdapterResult:
    """Predict neural responses with train-only SVD temporal components.

    The temporal filterbank can be high-dimensional relative to the number of
    trials. This adapter tests a more serious but still auditable model:

    1. standardize stimulus descriptors using the training split;
    2. fit an SVD subspace on training descriptors only;
    3. regress neural residuals in that low-dimensional temporal subspace;
    4. choose components, ridge alpha, and residual shrinkage by train-only CV.

    It is a stronger temporal baseline, not a mechanistic model and not a
    replacement for official Sensorium deep networks.
    """

    if folds < 2:
        raise ValueError("folds must be at least 2")
    if len(train_x) != len(train_y):
        raise ValueError("train_x and train_y must have the same number of trials")
    if beta_grid is None:
        beta_grid = np.linspace(0.0, 1.5, 61)
    alphas = tuple(float(value) for value in (alpha_grid or (1.0, 3.0, 10.0, 30.0, 100.0)))
    components = tuple(
        sorted(
            {
                max(1, min(int(value), train_x.shape[0] - 1, train_x.shape[1]))
                for value in component_grid
            }
        )
    )

    indices = np.arange(len(train_x))
    rng = np.random.default_rng(seed)
    rng.shuffle(indices)
    fold_indices = [fold for fold in np.array_split(indices, min(folds, len(indices))) if len(fold)]
    best_score = -np.inf
    best_components = components[0]
    best_alpha = alphas[0]
    best_beta = float(beta_grid[0])

    for component in components:
        for alpha in alphas:
            cv_mean = np.zeros_like(train_y, dtype=float)
            cv_residual_prediction = np.zeros_like(train_y, dtype=float)
            for fold in fold_indices:
                fit_indices = np.setdiff1d(indices, fold, assume_unique=False)
                train_mean = train_y[fit_indices].mean(axis=0, keepdims=True)
                residual = train_y[fit_indices] - train_mean
                cv_mean[fold] = train_mean
                # The SVD subspace is refit inside each fold, so model selection
                # does not leak held-out training-fold information.
                fit_x, heldout_x = _project_with_train_svd(
                    train_x[fit_indices], train_x[fold], components=component
                )
                cv_residual_prediction[fold] = ridge_fit_predict(
                    fit_x, residual, heldout_x, alpha=alpha
                )
            for beta in beta_grid:
                score = _safe_correlation(
                    cv_mean + float(beta) * cv_residual_prediction, train_y
                )
                if score > best_score:
                    best_score = score
                    best_components = component
                    best_alpha = alpha
                    best_beta = float(beta)

    full_mean = train_y.mean(axis=0, keepdims=True)
    full_residual = train_y - full_mean
    train_svd_x, eval_svd_x = _project_with_train_svd(
        train_x, eval_x, components=best_components
    )
    residual_prediction = ridge_fit_predict(
        train_svd_x, full_residual, eval_svd_x, alpha=best_alpha
    )
    prediction = np.repeat(full_mean, len(eval_x), axis=0) + best_beta * residual_prediction

    scrambled_train_x = rng.permutation(train_x)
    scrambled_svd_x, scrambled_eval_svd_x = _project_with_train_svd(
        scrambled_train_x, eval_x, components=best_components
    )
    scrambled_residual = ridge_fit_predict(
        scrambled_svd_x, full_residual, scrambled_eval_svd_x, alpha=best_alpha
    )
    scrambled_prediction = (
        np.repeat(full_mean, len(eval_x), axis=0) + best_beta * scrambled_residual
    )
    return SensoriumAdapterResult(
        name="temporal_svd_residual_ridge",
        prediction=prediction,
        scrambled_prediction=scrambled_prediction,
        diagnostics={
            "adapter_alpha": float(best_alpha),
            "adapter_beta": float(best_beta),
            "adapter_components": float(best_components),
            "adapter_cv_correlation": float(best_score),
            "adapter_cv_folds": float(len(fold_indices)),
            "adapter_alpha_candidates": float(len(alphas)),
            "adapter_component_candidates": float(len(components)),
        },
    )


def _random_fourier_features(
    train_x: np.ndarray,
    eval_x: np.ndarray,
    *,
    components: int,
    gamma: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Build train-standardized RBF random Fourier features.

    This is a compact approximation to an RBF-kernel ridge model. The random
    projection is generated from a fixed seed and the standardization is fit on
    train only, so held-out tiers do not influence the feature map.
    """

    train_x_std, eval_x_std, _, _ = _standardize_train_test(train_x, eval_x)
    width = max(1, int(components))
    rng = np.random.default_rng(seed)
    weights = rng.normal(scale=np.sqrt(2.0 * gamma), size=(train_x_std.shape[1], width))
    phase = rng.uniform(0.0, 2.0 * np.pi, size=width)
    scale = np.sqrt(2.0 / width)
    return (
        scale * np.cos(train_x_std @ weights + phase),
        scale * np.cos(eval_x_std @ weights + phase),
    )


def random_feature_residual_ridge_adapter(
    train_x: np.ndarray,
    train_y: np.ndarray,
    eval_x: np.ndarray,
    *,
    alpha_grid: tuple[float, ...] | None = None,
    component_grid: tuple[int, ...] = (32, 64, 128),
    gamma_grid: tuple[float, ...] = (0.01, 0.03, 0.1),
    seed: int = 17,
    folds: int = 5,
    beta_grid: np.ndarray | None = None,
) -> SensoriumAdapterResult:
    """Approximate nonlinear kernel residual ridge with train-only selection.

    This is a stronger local baseline than the linear temporal models, but it
    is not an official Sensorium/SOTA model. It asks whether a modest nonlinear
    stimulus map can explain held-out responses beyond mean and transparent
    temporal descriptors while preserving the same MIS contract.
    """

    if folds < 2:
        raise ValueError("folds must be at least 2")
    if len(train_x) != len(train_y):
        raise ValueError("train_x and train_y must have the same number of trials")
    if beta_grid is None:
        beta_grid = np.linspace(0.0, 1.5, 61)
    alphas = tuple(float(value) for value in (alpha_grid or (1.0, 3.0, 10.0, 30.0, 100.0)))
    components = tuple(
        sorted({max(1, min(int(value), max(1, train_x.shape[0] - 1))) for value in component_grid})
    )
    gammas = tuple(float(value) for value in gamma_grid)

    indices = np.arange(len(train_x))
    rng = np.random.default_rng(seed)
    rng.shuffle(indices)
    fold_indices = [fold for fold in np.array_split(indices, min(folds, len(indices))) if len(fold)]
    best_score = -np.inf
    best_alpha = alphas[0]
    best_beta = float(beta_grid[0])
    best_components = components[0]
    best_gamma = gammas[0]

    for component in components:
        for gamma in gammas:
            for alpha in alphas:
                cv_mean = np.zeros_like(train_y, dtype=float)
                cv_residual_prediction = np.zeros_like(train_y, dtype=float)
                for fold_number, fold in enumerate(fold_indices):
                    fit_indices = np.setdiff1d(indices, fold, assume_unique=False)
                    train_mean = train_y[fit_indices].mean(axis=0, keepdims=True)
                    residual = train_y[fit_indices] - train_mean
                    cv_mean[fold] = train_mean
                    fit_x, heldout_x = _random_fourier_features(
                        train_x[fit_indices],
                        train_x[fold],
                        components=component,
                        gamma=gamma,
                        seed=seed + 1009 * fold_number + 37 * component,
                    )
                    cv_residual_prediction[fold] = ridge_fit_predict(
                        fit_x, residual, heldout_x, alpha=alpha
                    )
                for beta in beta_grid:
                    score = _safe_correlation(
                        cv_mean + float(beta) * cv_residual_prediction, train_y
                    )
                    if score > best_score:
                        best_score = score
                        best_alpha = alpha
                        best_beta = float(beta)
                        best_components = component
                        best_gamma = gamma

    full_mean = train_y.mean(axis=0, keepdims=True)
    full_residual = train_y - full_mean
    train_features, eval_features = _random_fourier_features(
        train_x,
        eval_x,
        components=best_components,
        gamma=best_gamma,
        seed=seed + 7919,
    )
    residual_prediction = ridge_fit_predict(
        train_features, full_residual, eval_features, alpha=best_alpha
    )
    prediction = np.repeat(full_mean, len(eval_x), axis=0) + best_beta * residual_prediction

    scrambled_train_x = rng.permutation(train_x)
    scrambled_train_features, scrambled_eval_features = _random_fourier_features(
        scrambled_train_x,
        eval_x,
        components=best_components,
        gamma=best_gamma,
        seed=seed + 7919,
    )
    scrambled_residual = ridge_fit_predict(
        scrambled_train_features, full_residual, scrambled_eval_features, alpha=best_alpha
    )
    scrambled_prediction = (
        np.repeat(full_mean, len(eval_x), axis=0) + best_beta * scrambled_residual
    )
    return SensoriumAdapterResult(
        name="random_feature_residual_ridge",
        prediction=prediction,
        scrambled_prediction=scrambled_prediction,
        diagnostics={
            "adapter_alpha": float(best_alpha),
            "adapter_beta": float(best_beta),
            "adapter_components": float(best_components),
            "adapter_gamma": float(best_gamma),
            "adapter_cv_correlation": float(best_score),
            "adapter_cv_folds": float(len(fold_indices)),
            "adapter_alpha_candidates": float(len(alphas)),
            "adapter_component_candidates": float(len(components)),
            "adapter_gamma_candidates": float(len(gammas)),
        },
    )
