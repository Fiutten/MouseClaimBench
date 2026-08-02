#!/usr/bin/env python3
"""Validate exact v3 dependencies and optional external-data checksums."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
from pathlib import Path
from typing import Any

EXPECTED_DISTRIBUTIONS = {
    "anndata": "0.13.2",
    "causal-learn": "0.1.4.8",
    "causalchamber": "0.2.8",
    "clingo": "5.8.0",
    "MAPIE": "1.4.1",
    "numpy": "2.3.5",
    "ONE-api": "3.5.2",
    "openpyxl": "3.1.5",
    "pandas": "2.3.3",
    "scikit-learn": "1.9.0",
    "scipy": "1.18.0",
}


def file_digest(path: Path, algorithm: str) -> str:
    """Hash large files in bounded-memory chunks."""

    digest = hashlib.new(algorithm)
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _check(path: Path, expected: str) -> dict[str, Any]:
    algorithm, digest = expected.split(":", maxsplit=1)
    exists = path.is_file()
    observed = file_digest(path, algorithm) if exists else None
    return {
        "path": str(path),
        "exists": exists,
        "algorithm": algorithm,
        "expected": digest,
        "observed": observed,
        "valid": exists and observed == digest,
    }


def external_data_checks(root: Path) -> list[dict[str, Any]]:
    """Read committed manifests and verify every external source file."""

    checks: list[dict[str, Any]] = []
    causalbench = json.loads(
        (root / "configs/benchmarks/causalbench_v3_selection.json").read_text()
    )
    for source in causalbench["source_files"].values():
        checks.append(_check(root / source["path"], f"md5:{source['expected_md5']}"))
    checks.append(
        _check(
            root / "data/external/causalbench/summary_stats.xlsx",
            f"sha256:{causalbench['summary_stats_sha256']}",
        )
    )

    ibl = json.loads((root / "configs/benchmarks/ibl_bwm_v3_selection.json").read_text())
    ibl_root = root / "data/external/ibl"
    for name, metadata in ibl["source_files"].items():
        expected = (
            f"md5:{metadata['md5']}"
            if "md5" in metadata
            else f"sha256:{metadata['sha256']}"
        )
        checks.append(_check(ibl_root / name, expected))

    chambers = json.loads(
        (root / "results/causal_chambers_transport/summary.json").read_text()
    )
    for relative, expected in chambers["dataset_file_hashes"].items():
        checks.append(_check(root / relative, expected))
    return checks


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--require-external-data", action="store_true")
    args = parser.parse_args()

    versions = {
        name: importlib.metadata.version(name) for name in EXPECTED_DISTRIBUTIONS
    }
    version_mismatches = {
        name: {"expected": expected, "observed": versions[name]}
        for name, expected in EXPECTED_DISTRIBUTIONS.items()
        if versions[name] != expected
    }
    checks = external_data_checks(args.root) if args.require_external_data else []
    invalid_data = [item for item in checks if not item["valid"]]
    payload = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "versions": versions,
        "version_mismatches": version_mismatches,
        "external_data_checked": args.require_external_data,
        "external_file_count": len(checks),
        "invalid_external_files": invalid_data,
        "valid": not version_mismatches and not invalid_data,
    }
    print(json.dumps(payload, indent=2))
    if not payload["valid"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
