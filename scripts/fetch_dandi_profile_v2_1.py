#!/usr/bin/env python3
"""Fetch the two prospectively frozen DANDI profile-v2.1 resources.

Raw NWB files are stored under the gitignored data tree. Every selected asset is
verified against the published DANDI SHA-256 digest before the manifest is
updated. Existing verified files are reused.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import requests

API = "https://api.dandiarchive.org/api"
CHUNK_BYTES = 8 * 1024 * 1024


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(CHUNK_BYTES):
            digest.update(chunk)
    return digest.hexdigest()


def _request_json(url: str) -> dict[str, Any]:
    response = requests.get(url, timeout=120)
    response.raise_for_status()
    return response.json()


def _catalog(dandiset: str, version: str) -> list[dict[str, Any]]:
    url = f"{API}/dandisets/{dandiset}/versions/{version}/assets/?page_size=100"
    rows: list[dict[str, Any]] = []
    while url:
        page = _request_json(url)
        rows.extend(page["results"])
        url = page.get("next")
    return rows


def _select_ach(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    # Published filenames containing `-Ach-` are single-sensor sessions. The
    # remaining behavior+ophys files are schema candidates for simultaneous
    # rACh and GCaMP signals and are verified by the analysis benchmark.
    return sorted(
        (
            row
            for row in rows
            if row["path"].endswith("behavior+ophys.nwb")
            and "-Ach-" not in row["path"]
        ),
        key=lambda row: row["path"],
    )


def _select_contrast(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    eligible = [row for row in rows if row["path"].endswith("behavior+ophys.nwb")]
    subjects = sorted({row["path"].split("/")[0] for row in eligible})
    return [
        min(
            (row for row in eligible if row["path"].split("/")[0] == subject),
            key=lambda row: row["path"],
        )
        for subject in subjects
    ]


def _asset_detail(asset_id: str) -> dict[str, Any]:
    return _request_json(f"{API}/assets/{asset_id}/")


def _download_one(
    row: dict[str, Any],
    root: Path,
) -> dict[str, Any]:
    detail = _asset_detail(str(row["asset_id"]))
    expected = str(detail["digest"]["dandi:sha2-256"])
    relative = Path(str(row["path"]))
    destination = root / relative
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and destination.stat().st_size == int(row["size"]):
        observed = _sha256(destination)
        if observed == expected:
            return {
                **row,
                "local_path": str(destination),
                "official_sha256": expected,
                "observed_sha256": observed,
                "verified": True,
                "reused": True,
            }
    partial = destination.with_suffix(destination.suffix + ".part")
    partial.unlink(missing_ok=True)
    response = requests.get(
        f"{API}/assets/{row['asset_id']}/download/",
        stream=True,
        timeout=(120, 600),
    )
    response.raise_for_status()
    digest = hashlib.sha256()
    written = 0
    with partial.open("wb") as handle:
        for chunk in response.iter_content(chunk_size=CHUNK_BYTES):
            if not chunk:
                continue
            handle.write(chunk)
            digest.update(chunk)
            written += len(chunk)
    observed = digest.hexdigest()
    if written != int(row["size"]):
        raise RuntimeError(f"download size mismatch for {row['path']}")
    if observed != expected:
        raise RuntimeError(f"download SHA-256 mismatch for {row['path']}")
    partial.replace(destination)
    return {
        **row,
        "local_path": str(destination),
        "official_sha256": expected,
        "observed_sha256": observed,
        "verified": True,
        "reused": False,
    }


def run(*, resource: str, root: Path, workers: int) -> Path:
    if resource == "ach":
        dandiset, version = "001176", "0.260610.2204"
        selector = _select_ach
    elif resource == "contrast":
        dandiset, version = "000039", "0.230223.1216"
        selector = _select_contrast
    else:
        raise ValueError(f"unknown resource: {resource}")
    rows = _catalog(dandiset, version)
    selected = selector(rows)
    output_root = root / f"dandi_{resource}_profile_v2_1"
    output_root.mkdir(parents=True, exist_ok=True)
    completed: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(_download_one, row, output_root): row for row in selected
        }
        for future in as_completed(futures):
            completed.append(future.result())
            print(f"verified {len(completed)}/{len(selected)}: {futures[future]['path']}")
    completed.sort(key=lambda row: row["path"])
    payload = {
        "resource": resource,
        "dandiset": dandiset,
        "version": version,
        "catalog_assets": len(rows),
        "selected_assets": len(selected),
        "selected_subjects": len(
            {row["path"].split("/")[0] for row in selected}
        ),
        "selected_bytes": sum(int(row["size"]) for row in selected),
        "selection_rule": (
            "non-Ach behavior+ophys schema candidates"
            if resource == "ach"
            else "lexicographically first behavior+ophys asset per subject"
        ),
        "assets": completed,
    }
    manifest = output_root / "manifest.json"
    manifest.write_text(json.dumps(payload, indent=2) + "\n")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("resource", choices=("ach", "contrast"))
    parser.add_argument("--root", type=Path, default=Path("data/external"))
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    print(run(resource=args.resource, root=args.root, workers=args.workers).resolve())


if __name__ == "__main__":
    main()
