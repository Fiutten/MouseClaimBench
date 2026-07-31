"""Portable, machine-readable experiment artifact persistence."""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from mousebrainbench import __version__
from mousebrainbench.schemas import ConnectivityMatrix, SimulationState


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def code_revision() -> str:
    """Return the source revision used to generate an artifact.

    Publication pipelines may set ``MOUSEBRAINBENCH_GIT_REVISION`` once, before
    writing a sequence of tracked outputs. This keeps every artifact tied to the
    same clean source commit even though earlier outputs make the working tree
    dirty during the run. The override accepts only a hexadecimal Git object ID.
    """

    override = os.environ.get("MOUSEBRAINBENCH_GIT_REVISION")
    if override:
        if not re.fullmatch(r"[0-9a-fA-F]{7,40}", override):
            raise ValueError(
                "MOUSEBRAINBENCH_GIT_REVISION must be a 7-40 character hexadecimal Git ID"
            )
        try:
            resolved = subprocess.check_output(
                ["git", "rev-parse", "--verify", f"{override}^{{commit}}"],
                text=True,
                stderr=subprocess.DEVNULL,
            ).strip()
            head = subprocess.check_output(
                ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
            ).strip()
        except (OSError, subprocess.CalledProcessError) as exc:
            raise ValueError(
                "MOUSEBRAINBENCH_GIT_REVISION must resolve to a local Git commit"
            ) from exc
        if resolved.lower() != head.lower():
            raise ValueError(
                "MOUSEBRAINBENCH_GIT_REVISION must resolve to the checked-out HEAD"
            )
        return resolved.lower()

    try:
        return subprocess.check_output(
            ["git", "describe", "--always", "--dirty"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def config_hash(config: dict[str, Any]) -> str:
    payload = yaml.safe_dump({key: value for key, value in config.items() if key != "_config_path"})
    return hashlib.sha256(payload.encode()).hexdigest()[:12]


def save_run(
    output_root: str | Path,
    run_name: str,
    config: dict[str, Any],
    connectivity: ConnectivityMatrix,
    baseline: SimulationState,
    perturbed: SimulationState,
    metrics: dict[str, Any],
) -> Path:
    """Persist all inputs and outputs required to inspect or repeat one run."""

    run_dir = Path(output_root) / f"{run_name}-{config_hash(config)}"
    run_dir.mkdir(parents=True, exist_ok=True)
    clean_config = {key: value for key, value in config.items() if key != "_config_path"}
    (run_dir / "config.yaml").write_text(yaml.safe_dump(clean_config, sort_keys=False))
    provenance = {
        "mousebrainbench_version": __version__,
        "git_revision": code_revision(),
        "source_config": config.get("_config_path"),
        "connectivity_metadata": connectivity.metadata,
        "baseline_metadata": baseline.metadata,
        "perturbed_metadata": perturbed.metadata,
    }
    (run_dir / "provenance.json").write_text(json.dumps(_json_safe(provenance), indent=2))
    (run_dir / "metrics.json").write_text(json.dumps(_json_safe(metrics), indent=2))
    np.savez_compressed(
        run_dir / "results.npz",
        time=baseline.time,
        region_ids=np.asarray(baseline.region_ids),
        weights=connectivity.dense_weights(copy=False),
        baseline_activity=baseline.activity,
        perturbed_activity=perturbed.activity,
    )
    return run_dir
