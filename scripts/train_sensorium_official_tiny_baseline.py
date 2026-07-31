"""Train a bounded official-architecture Sensorium baseline on local mice.

This script uses the official Sensorium loader and model factory, but keeps the
training budget intentionally small so it can run as a reproducible integration
test on a laptop. The output is a MouseBrainBench artifact, not a leaderboard
submission and not a state-of-the-art claim.

Run with the isolated official environment:

    MPLBACKEND=Agg MPLCONFIGDIR=.cache/matplotlib XDG_CACHE_HOME=.cache/xdg \
    PYTHONPATH=external/neuralpredictors_43fa:external/sensorium_2023 \
    .venv-sensorium-official/bin/python \
      scripts/train_sensorium_official_tiny_baseline.py
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Iterable

from smoke_sensorium_official_model import _patch_python312_collections


DEFAULT_DATA_ROOT = Path("data/dynamic_sensorium/extracted")
DEFAULT_OUTPUT = Path(
    "results/sensorium_official_baseline_audit/official_trained_baseline_summary.json"
)


def _mouse_roots(data_root: Path, limit: int | None) -> list[Path]:
    """Return Dynamic Sensorium mouse directories in deterministic order."""

    roots = sorted(path for path in data_root.glob("dynamic*-Video-*") if path.is_dir())
    return roots[:limit] if limit is not None else roots


def _safe_corr(prediction, target) -> float:
    """Compute Pearson correlation with a conservative zero-variance guard."""

    import numpy as np  # noqa: PLC0415

    pred = np.asarray(prediction, dtype=float).reshape(-1)
    true = np.asarray(target, dtype=float).reshape(-1)
    if pred.size == 0 or true.size == 0:
        return float("nan")
    pred_std = float(pred.std())
    true_std = float(true.std())
    if pred_std == 0.0 or true_std == 0.0:
        return 0.0
    return float(np.corrcoef(pred, true)[0, 1])


def _select_device(requested: str):
    """Select CPU or Apple Metal/MPS for local official Sensorium training."""

    import torch  # noqa: PLC0415

    if requested == "auto":
        return torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    if requested == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("Requested MPS device, but torch.backends.mps.is_available() is False")
    return torch.device(requested)


def _data_key(loaders: dict[str, Any]) -> str:
    """Extract the single data key created by the official mouse loader."""

    keys = sorted(loaders["train"].keys())
    if len(keys) != 1:
        raise ValueError(f"Expected one official data key, found {keys}")
    return str(keys[0])


def _build_official_tiny_model(
    loaders: dict[str, Any],
    seed: int,
    hidden_channels: list[int],
    spatial_kernel: int,
    temporal_kernel: int,
):
    """Create a minimal Factorized3dCore + factorized readout Sensorium model."""

    from sensorium.models.make_model import make_video_model  # noqa: PLC0415

    # This is the same official model family exercised by the smoke test. The
    # small channel count keeps the benchmark bounded; it is not the published
    # competition configuration.
    core = dict(
        input_channels=1,
        hidden_channels=hidden_channels,
        spatial_input_kernel=(spatial_kernel, spatial_kernel),
        temporal_input_kernel=temporal_kernel,
        spatial_hidden_kernel=(spatial_kernel, spatial_kernel),
        temporal_hidden_kernel=temporal_kernel,
        stride=1,
        layers=len(hidden_channels),
        gamma_input_spatial=0,
        gamma_input_temporal=0,
        bias=True,
        hidden_nonlinearities="elu",
        x_shift=0,
        y_shift=0,
        batch_norm=False,
        laplace_padding=None,
        input_regularizer="LaplaceL2norm",
        padding=True,
        final_nonlin=True,
        momentum=0.7,
    )
    readout = dict(
        bias=False,
        gamma_readout=0.0,
        spatial_and_feature_reg_weight=0.0,
        positive_weights=False,
        normalize=False,
        init_noise=0.001,
    )
    return make_video_model(
        loaders,
        seed=seed,
        core_dict=core,
        core_type="3D_factorised",
        readout_dict=readout,
        readout_type="factorised",
        use_gru=False,
        gru_dict=None,
        use_shifter=False,
        shifter_dict=None,
        shifter_type="MLP",
        deeplake_ds=False,
    )


def _aligned_target(responses, n_time: int):
    """Align official responses to the model output time axis."""

    # Official responses are [batch, neurons, time]; model output is
    # [batch, model_time, neurons]. We compare the final frames because the
    # 3D convolutional core shortens the temporal axis.
    return responses[:, :, -n_time:].transpose(1, 2)


def _limited_batches(loader: Iterable[Any], max_batches: int):
    """Yield at most ``max_batches`` batches from an official dataloader."""

    for index, batch in enumerate(loader):
        if index >= max_batches:
            break
        yield batch


def _mean_baseline(train_loader, data_key: str, model, max_batches: int, device):
    """Estimate a per-neuron mean response baseline on the same aligned frames."""

    import torch  # noqa: PLC0415

    chunks = []
    model.eval()
    with torch.no_grad():
        for batch in _limited_batches(train_loader, max_batches):
            output = model(batch.videos.float().to(device), data_key=data_key)
            target = _aligned_target(batch.responses.float().to(device), output.shape[1])
            chunks.append(target.reshape(-1, target.shape[-1]).cpu())
    if not chunks:
        raise RuntimeError("No batches available for mean baseline estimation")
    return torch.cat(chunks, dim=0).mean(dim=0)


def _evaluate(loader, data_key: str, model, mean_response, max_batches: int, device) -> dict[str, Any]:
    """Evaluate model and matched mean baseline on held-out/oracle batches."""

    import numpy as np  # noqa: PLC0415
    import torch  # noqa: PLC0415

    model_predictions = []
    mean_predictions = []
    targets = []
    model.eval()
    with torch.no_grad():
        for batch in _limited_batches(loader, max_batches):
            output = model(batch.videos.float().to(device), data_key=data_key)
            target = _aligned_target(batch.responses.float().to(device), output.shape[1])
            expanded_mean = mean_response.to(device).view(1, 1, -1).expand_as(target)
            model_predictions.append(output.cpu().numpy())
            mean_predictions.append(expanded_mean.cpu().numpy())
            targets.append(target.cpu().numpy())
    if not targets:
        return {
            "n_eval_batches": 0,
            "official_sensorium_correlation": float("nan"),
            "mean_response_correlation": float("nan"),
        }
    prediction = np.concatenate(model_predictions, axis=0)
    mean_prediction = np.concatenate(mean_predictions, axis=0)
    target = np.concatenate(targets, axis=0)
    return {
        "n_eval_batches": len(targets),
        "official_sensorium_correlation": _safe_corr(prediction, target),
        "mean_response_correlation": _safe_corr(mean_prediction, target),
    }


def train_one_mouse(
    *,
    mouse_root: Path,
    max_train_batches: int,
    max_eval_batches: int,
    mean_batches: int,
    learning_rate: float,
    seed: int,
    device_name: str,
    hidden_channels: list[int],
    spatial_kernel: int,
    temporal_kernel: int,
) -> dict[str, Any]:
    """Train and evaluate one bounded official-architecture Sensorium baseline."""

    _patch_python312_collections()

    import torch  # noqa: PLC0415
    from sensorium.datasets.mouse_video_loaders import mouse_video_loader  # noqa: PLC0415

    torch.manual_seed(seed)
    device = _select_device(device_name)
    loaders = mouse_video_loader(
        # The trailing slash preserves the mouse directory as official data key.
        [str(mouse_root) + "/"],
        batch_size=1,
        frames=51,
        max_frame=60,
        include_behavior=False,
        include_pupil_centers=False,
        scale=0.25,
        cuda=False,
    )
    data_key = _data_key(loaders)
    model = _build_official_tiny_model(
        loaders,
        seed=seed,
        hidden_channels=hidden_channels,
        spatial_kernel=spatial_kernel,
        temporal_kernel=temporal_kernel,
    )
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    loss_fn = torch.nn.MSELoss()

    losses: list[float] = []
    model.train()
    for batch in _limited_batches(loaders["train"][data_key], max_train_batches):
        optimizer.zero_grad(set_to_none=True)
        output = model(batch.videos.float().to(device), data_key=data_key)
        target = _aligned_target(batch.responses.float().to(device), output.shape[1])
        loss = loss_fn(output, target)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach().cpu()))

    mean_response = _mean_baseline(loaders["train"][data_key], data_key, model, mean_batches, device)
    tier = "oracle" if "oracle" in loaders else "validation"
    eval_result = _evaluate(
        loaders[tier][data_key],
        data_key,
        model,
        mean_response,
        max_eval_batches,
        device,
    )
    official_corr = eval_result["official_sensorium_correlation"]
    mean_corr = eval_result["mean_response_correlation"]
    return {
        "mouse": mouse_root.name,
        "data_key": data_key,
        "evaluation_tier": tier,
        "n_train_batches": len(losses),
        "n_eval_batches": eval_result["n_eval_batches"],
        "final_train_loss": losses[-1] if losses else None,
        "median_train_loss": sorted(losses)[len(losses) // 2] if losses else None,
        "official_sensorium_correlation": official_corr,
        "mean_response_correlation": mean_corr,
        "official_sensorium_minus_mean": (
            official_corr - mean_corr
            if math.isfinite(float(official_corr)) and math.isfinite(float(mean_corr))
            else None
        ),
        "model_family": "official_sensorium_factorized3d_core_factorized_readout_bounded",
        "hidden_channels": hidden_channels,
        "spatial_kernel": spatial_kernel,
        "temporal_kernel": temporal_kernel,
        "device": str(device),
    }


def run(
    *,
    data_root: Path,
    output: Path,
    limit_mice: int | None,
    max_train_batches: int,
    max_eval_batches: int,
    mean_batches: int,
    learning_rate: float,
    seed: int,
    q1_min_mice: int,
    device_name: str,
    hidden_channels: list[int],
    spatial_kernel: int,
    temporal_kernel: int,
) -> dict[str, Any]:
    """Run the bounded official baseline over the selected local mice."""

    roots = _mouse_roots(data_root, limit_mice)
    rows = [
        train_one_mouse(
            mouse_root=root,
            max_train_batches=max_train_batches,
            max_eval_batches=max_eval_batches,
            mean_batches=mean_batches,
            learning_rate=learning_rate,
            seed=seed + index,
            device_name=device_name,
            hidden_channels=hidden_channels,
            spatial_kernel=spatial_kernel,
            temporal_kernel=temporal_kernel,
        )
        for index, root in enumerate(roots)
    ]
    usable_rows = [
        row
        for row in rows
        if row["n_train_batches"] > 0
        and row["n_eval_batches"] > 0
        and math.isfinite(float(row["official_sensorium_correlation"]))
    ]
    q1_baseline_qualified = False
    payload = {
        "benchmark": "official_sensorium_bounded_trained_baseline",
        "data_root": str(data_root),
        "trained_baseline": True,
        "official_loader": True,
        "official_model_factory": True,
        "official_architecture_family": "Factorized3dCore + factorized readout",
        "bounded_training": True,
        "sota_claim": False,
        "q1_baseline_qualified": q1_baseline_qualified,
        "q1_qualification_rule": (
            f"This artifact has {len(usable_rows)} usable mice. It intentionally "
            "does not qualify as a Q1-level official baseline because it uses a "
            "bounded local training budget. Q1 qualification requires the published "
            "official budget/configuration or a documented external leaderboard "
            "checkpoint evaluated through MouseBrainBench."
        ),
        "n_mice": len(rows),
        "n_usable_mice": len(usable_rows),
        "max_train_batches": max_train_batches,
        "max_eval_batches": max_eval_batches,
        "mean_batches": mean_batches,
        "learning_rate": learning_rate,
        "seed": seed,
        "device_requested": device_name,
        "hidden_channels": hidden_channels,
        "spatial_kernel": spatial_kernel,
        "temporal_kernel": temporal_kernel,
        "rows": rows,
        "interpretation": (
            "This artifact proves that official Sensorium code can be trained and "
            "evaluated inside MouseBrainBench. It is a bounded local control, not "
            "a SOTA Sensorium benchmark result."
        ),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2))
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--limit-mice", type=int, default=5)
    parser.add_argument("--max-train-batches", type=int, default=24)
    parser.add_argument("--max-eval-batches", type=int, default=16)
    parser.add_argument("--mean-batches", type=int, default=24)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--q1-min-mice", type=int, default=5)
    parser.add_argument("--device", choices=("auto", "cpu", "mps"), default="auto")
    parser.add_argument("--hidden-channels", default="4")
    parser.add_argument("--spatial-kernel", type=int, default=3)
    parser.add_argument("--temporal-kernel", type=int, default=3)
    args = parser.parse_args()
    hidden_channels = [int(value) for value in args.hidden_channels.split(",") if value]
    payload = run(
        data_root=args.data_root,
        output=args.output,
        limit_mice=args.limit_mice,
        max_train_batches=args.max_train_batches,
        max_eval_batches=args.max_eval_batches,
        mean_batches=args.mean_batches,
        learning_rate=args.learning_rate,
        seed=args.seed,
        q1_min_mice=args.q1_min_mice,
        device_name=args.device,
        hidden_channels=hidden_channels,
        spatial_kernel=args.spatial_kernel,
        temporal_kernel=args.temporal_kernel,
    )
    print(json.dumps({"output": str(args.output.resolve()), "n_usable_mice": payload["n_usable_mice"]}))


if __name__ == "__main__":
    main()
