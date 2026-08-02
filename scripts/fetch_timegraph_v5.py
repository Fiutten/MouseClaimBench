"""Fetch the exact TimeGraph source files frozen by the v5 protocol."""

from __future__ import annotations

import argparse
import hashlib
import subprocess
from pathlib import Path

import yaml

DEFAULT_PROTOCOL = Path("configs/benchmarks/semantic_risk_control_v5.yaml")
DEFAULT_ROOT = Path("data/external/timegraph_v5")


def _sha256(path: Path) -> str:
    return f"sha256:{hashlib.sha256(path.read_bytes()).hexdigest()}"


def _run(*args: str, cwd: Path | None = None) -> None:
    subprocess.run(args, cwd=cwd, check=True)


def fetch(protocol_path: Path, root: Path) -> Path:
    """Clone a sparse, pinned source checkout and verify every numerical file."""

    protocol = yaml.safe_load(protocol_path.read_text())
    population = protocol["confirmatory_population"]
    revision = str(population["source_revision"])
    source_files = dict(population["source_files"])
    if not root.exists():
        root.parent.mkdir(parents=True, exist_ok=True)
        _run(
            "git",
            "clone",
            "--filter=blob:none",
            "--no-checkout",
            str(population["repository"]),
            str(root),
        )
        _run("git", "sparse-checkout", "init", "--no-cone", cwd=root)
        _run(
            "git",
            "sparse-checkout",
            "set",
            *source_files,
            "README.md",
            "LICENSE",
            "CITATION.cff",
            cwd=root,
        )
    _run("git", "fetch", "--depth", "1", "origin", revision, cwd=root)
    _run("git", "checkout", "--detach", revision, cwd=root)
    observed_revision = subprocess.check_output(
        ("git", "rev-parse", "HEAD"), cwd=root, text=True
    ).strip()
    if observed_revision != revision:
        raise RuntimeError(f"TimeGraph revision mismatch: {observed_revision}")
    for relative, expected in source_files.items():
        observed = _sha256(root / relative)
        if observed != expected:
            raise RuntimeError(f"TimeGraph hash mismatch for {relative}: {observed}")
    return root


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    args = parser.parse_args()
    print(fetch(args.protocol, args.root).resolve())


if __name__ == "__main__":
    main()
