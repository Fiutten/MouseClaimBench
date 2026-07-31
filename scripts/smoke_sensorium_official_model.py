"""Smoke-test the official Sensorium 2023 model stack on local data.

This script must be run with the isolated official environment:

    MPLBACKEND=Agg MPLCONFIGDIR=.cache/matplotlib XDG_CACHE_HOME=.cache/xdg \
    PYTHONPATH=external/neuralpredictors_43fa:external/sensorium_2023 \
    .venv-sensorium-official/bin/python scripts/smoke_sensorium_official_model.py

It does not claim official leaderboard performance. It verifies that official
Sensorium loader/model code can instantiate and run a forward pass on a local
Dynamic Sensorium mouse. Training/evaluation remains a separate Q1 gate.
"""

from __future__ import annotations

import argparse
import collections
import collections.abc
import json
from pathlib import Path
from typing import Any


def _patch_python312_collections() -> None:
    """Patch legacy imports used by older neuralpredictors on Python 3.12."""

    collections.Iterable = collections.abc.Iterable  # type: ignore[attr-defined]
    collections.Mapping = collections.abc.Mapping  # type: ignore[attr-defined]
    collections.MutableMapping = collections.abc.MutableMapping  # type: ignore[attr-defined]


def run_smoke(mouse_root: Path) -> dict[str, Any]:
    """Instantiate an official Sensorium model and run one local forward pass."""

    _patch_python312_collections()

    # Imports intentionally happen after the Python 3.12 compatibility patch.
    import neuralpredictors  # noqa: PLC0415
    import torch  # noqa: PLC0415
    from sensorium.datasets.mouse_video_loaders import mouse_video_loader  # noqa: PLC0415
    from sensorium.models.make_model import make_video_model  # noqa: PLC0415

    loaders = mouse_video_loader(
        [str(mouse_root)],
        batch_size=1,
        frames=51,
        max_frame=60,
        include_behavior=False,
        include_pupil_centers=False,
        scale=0.25,
        cuda=False,
    )

    # Minimal official architecture: Factorized3dCore plus factorized readout.
    # The tiny channel count is for integration smoke testing only; it is not
    # the published benchmark configuration and must not be reported as SOTA.
    core = dict(
        input_channels=1,
        hidden_channels=[4],
        spatial_input_kernel=(3, 3),
        temporal_input_kernel=3,
        spatial_hidden_kernel=(3, 3),
        temporal_hidden_kernel=3,
        stride=1,
        layers=1,
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
    model = make_video_model(
        loaders,
        seed=1,
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
    model.eval()
    batch = next(iter(loaders["train"]["extracted"]))
    with torch.no_grad():
        output = model(batch.videos, data_key="extracted")

    return {
        "official_stack_forward_ok": True,
        "mouse_root": str(mouse_root),
        "neuralpredictors_source": str(Path(neuralpredictors.__file__).resolve()),
        "available_tiers": sorted(str(key) for key in loaders.keys()),
        "input_shape": list(batch.videos.shape),
        "response_shape": list(batch.responses.shape),
        "output_shape": list(output.shape),
        "model_family": "official_sensorium_factorized3d_core_factorized_readout_smoke",
        "trained_baseline": False,
        "sota_claim": False,
        "interpretation": (
            "Official Sensorium loader/model code runs on local Dynamic Sensorium "
            "data, but this smoke test is not an official trained baseline."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mouse-root",
        type=Path,
        default=Path(
            "data/dynamic_sensorium/extracted/"
            "dynamic29515-10-12-Video-9b4f6a1a067fe51e15306b9628efea20"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/sensorium_official_baseline_audit/official_model_smoke.json"),
    )
    args = parser.parse_args()
    payload = run_smoke(args.mouse_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2))
    print(json.dumps({"output": str(args.output.resolve()), "official_stack_forward_ok": True}))


if __name__ == "__main__":
    main()
