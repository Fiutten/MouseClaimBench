"""Check whether PyTorch can use the Apple Metal/MPS GPU backend.

Run this outside restricted sandboxes when possible. Some execution sandboxes
can hide the Metal device even when the same Python environment can use it from
a normal terminal.
"""

from __future__ import annotations

import json
import platform

import torch


def main() -> None:
    """Print a compact machine-readable MPS diagnostic."""

    payload = {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "mps_built": torch.backends.mps.is_built() if hasattr(torch.backends, "mps") else False,
        "mps_available": (
            torch.backends.mps.is_available() if hasattr(torch.backends, "mps") else False
        ),
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_count": torch.cuda.device_count(),
    }
    if payload["mps_available"]:
        tensor = torch.ones(4, device="mps")
        payload["mps_tensor_sum"] = float((tensor + 1).sum().cpu())
        payload["mps_tensor_device"] = str(tensor.device)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
