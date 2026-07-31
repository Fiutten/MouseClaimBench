"""Run a compact PyTorch MLP baseline on Dynamic Sensorium features.

This is a real neural-network baseline implemented with PyTorch, but it is not
the official Sensorium starter-kit model. It exists to test whether a modest
nonlinear NN can beat the transparent SVD baseline under the same train-only
selection discipline.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import median
from typing import Any

import numpy as np
import torch
from torch import nn

from mousebrainbench.data.loaders.sensorium import load_sensorium_directory


def _safe_corr(left: np.ndarray, right: np.ndarray) -> float:
    left = np.asarray(left, dtype=float).ravel()
    right = np.asarray(right, dtype=float).ravel()
    if left.size != right.size or np.std(left) == 0 or np.std(right) == 0:
        return 0.0
    return float(np.corrcoef(left, right)[0, 1])


def _tier_mask(tiers: np.ndarray, names: tuple[str, ...]) -> np.ndarray:
    return np.asarray([tier in names for tier in tiers], dtype=bool)


def _standardize(train: np.ndarray, other: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = train.mean(axis=0, keepdims=True)
    std = train.std(axis=0, keepdims=True)
    std[std == 0] = 1.0
    return (train - mean) / std, (other - mean) / std


class ResidualMLP(nn.Module):
    """Small MLP mapping stimulus features to neural residuals."""

    def __init__(self, in_features: int, out_features: int, hidden: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, out_features),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def _fit_one(
    train_x: np.ndarray,
    train_y: np.ndarray,
    eval_x: np.ndarray,
    *,
    hidden: int,
    weight_decay: float,
    beta_grid: tuple[float, ...],
    seed: int,
    epochs: int,
    batch_size: int,
) -> tuple[np.ndarray, dict[str, float]]:
    rng = np.random.default_rng(seed)
    indices = np.arange(len(train_x))
    rng.shuffle(indices)
    split = max(1, int(0.8 * len(indices)))
    fit_idx = indices[:split]
    val_idx = indices[split:]
    if len(val_idx) == 0:
        val_idx = indices[-1:]
        fit_idx = indices[:-1]

    fit_x_std, val_x_std = _standardize(train_x[fit_idx], train_x[val_idx])
    _, eval_x_std = _standardize(train_x[fit_idx], eval_x)
    train_mean = train_y[fit_idx].mean(axis=0, keepdims=True)
    fit_residual = train_y[fit_idx] - train_mean
    val_y = train_y[val_idx]

    torch.manual_seed(seed)
    model = ResidualMLP(fit_x_std.shape[1], train_y.shape[1], hidden=hidden, dropout=0.05)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=weight_decay)
    loss_fn = nn.MSELoss()
    x_tensor = torch.as_tensor(fit_x_std, dtype=torch.float32)
    y_tensor = torch.as_tensor(fit_residual, dtype=torch.float32)
    for _ in range(epochs):
        order = torch.randperm(len(x_tensor))
        for start in range(0, len(order), batch_size):
            batch = order[start : start + batch_size]
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(x_tensor[batch]), y_tensor[batch])
            loss.backward()
            optimizer.step()

    model.eval()
    with torch.no_grad():
        val_residual = model(torch.as_tensor(val_x_std, dtype=torch.float32)).numpy()
        eval_residual = model(torch.as_tensor(eval_x_std, dtype=torch.float32)).numpy()
    best_beta = 0.0
    best_score = -np.inf
    val_mean = np.repeat(train_mean, len(val_y), axis=0)
    for beta in beta_grid:
        score = _safe_corr(val_mean + beta * val_residual, val_y)
        if score > best_score:
            best_score = score
            best_beta = float(beta)
    prediction = np.repeat(train_mean, len(eval_x), axis=0) + best_beta * eval_residual
    return prediction, {
        "hidden": float(hidden),
        "weight_decay": float(weight_decay),
        "beta": float(best_beta),
        "validation_correlation": float(best_score),
        "epochs": float(epochs),
    }


def run_one(
    root: Path,
    *,
    eval_tiers: tuple[str, ...],
    hidden_grid: tuple[int, ...],
    weight_decay_grid: tuple[float, ...],
    seed: int,
    epochs: int,
    batch_size: int,
    ood_gate: bool,
) -> dict[str, Any]:
    table = load_sensorium_directory(root, modality="dynamic", feature_mode="temporal_filterbank")
    train_mask = _tier_mask(table.tiers, ("train",))
    eval_mask = _tier_mask(table.tiers, eval_tiers)
    train_x = table.stimulus_features[train_mask]
    train_y = table.responses[train_mask]
    eval_x = table.stimulus_features[eval_mask]
    eval_y = table.responses[eval_mask]
    mean_prediction = np.repeat(train_y.mean(axis=0, keepdims=True), len(eval_y), axis=0)
    mean_corr = _safe_corr(mean_prediction, eval_y)

    beta_grid = tuple(float(value) for value in np.linspace(0.0, 1.5, 31))
    best_prediction: np.ndarray | None = None
    best_diag: dict[str, float] | None = None
    best_val = -np.inf
    for hidden in hidden_grid:
        for weight_decay in weight_decay_grid:
            prediction, diag = _fit_one(
                train_x,
                train_y,
                eval_x,
                hidden=hidden,
                weight_decay=weight_decay,
                beta_grid=beta_grid,
                seed=seed,
                epochs=epochs,
                batch_size=batch_size,
            )
            if diag["validation_correlation"] > best_val:
                best_val = diag["validation_correlation"]
                best_prediction = prediction
                best_diag = diag
    if best_prediction is None or best_diag is None:
        raise RuntimeError("MLP selection produced no candidate")
    corr = _safe_corr(best_prediction, eval_y)
    return {
        "mouse": root.name.split("-Video-")[0],
        "root": str(root),
        "eval_tiers": list(eval_tiers),
        "n_train_trials": int(train_mask.sum()),
        "n_eval_trials": int(eval_mask.sum()),
        "n_neurons": int(train_y.shape[1]),
        "mean_correlation": mean_corr,
        "torch_mlp_correlation": corr,
        "torch_mlp_minus_mean": float(corr - mean_corr),
        "diagnostics": best_diag,
        "has_ood_generalization_gate": bool(ood_gate),
        "interpretation": (
            "Compact PyTorch MLP baseline; not official Sensorium SOTA. "
            "Hyperparameters are selected inside train only."
        ),
    }


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    minus_mean = [float(row["torch_mlp_minus_mean"]) for row in rows]
    return {
        "dataset": "Dynamic Sensorium compact PyTorch MLP baseline",
        "adapter": "torch_residual_mlp",
        "n_mice": len(rows),
        "positive_torch_mlp_minus_mean_count": int(sum(value > 0 for value in minus_mean)),
        "median_torch_mlp_minus_mean": median(minus_mean) if minus_mean else None,
        "rows": rows,
        "interpretation": (
            "This is the strongest local NN control available without installing "
            "the official Sensorium starter-kit stack. It is not a SOTA claim."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--extracted-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--summary-output", type=Path, required=True)
    parser.add_argument("--eval-tier", action="append", dest="eval_tiers", required=True)
    parser.add_argument("--hidden-grid", default="64,128")
    parser.add_argument("--weight-decay-grid", default="0.0001,0.001")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--ood-gate", action="store_true")
    args = parser.parse_args()

    roots = sorted(path for path in args.extracted_root.glob("dynamic*") if path.is_dir())
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for index, root in enumerate(roots):
        row = run_one(
            root,
            eval_tiers=tuple(args.eval_tiers),
            hidden_grid=tuple(int(value) for value in args.hidden_grid.split(",")),
            weight_decay_grid=tuple(float(value) for value in args.weight_decay_grid.split(",")),
            seed=args.seed + index,
            epochs=args.epochs,
            batch_size=args.batch_size,
            ood_gate=args.ood_gate,
        )
        output = args.output_dir / f"{row['mouse']}_torch_mlp_mis.json"
        output.write_text(json.dumps(row, indent=2))
        row["output"] = str(output)
        rows.append(row)
    payload = summarize(rows)
    args.summary_output.parent.mkdir(parents=True, exist_ok=True)
    args.summary_output.write_text(json.dumps(payload, indent=2))
    print(json.dumps({"output": str(args.summary_output.resolve())}))


if __name__ == "__main__":
    main()
