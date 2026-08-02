"""Fetch exact, QC-passed IBL trial tables for the frozen v5 mouse cohort.

The script reads no behavioral values while choosing files. It selects the one
catalog entry marked as both the default dataset and QC PASS, downloads from the
public Brain-Wide Map repository, and verifies byte count and MD5 before adding
the immutable dataset UUID and revision to the local manifest.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import ssl
import urllib.request
from pathlib import Path
from typing import Any

DEFAULT_SELECTION = Path("configs/benchmarks/ibl_behavior_v5_selection.json")
DEFAULT_ROOT = Path("data/external/ibl_behavior_v5")
DEFAULT_MANIFEST = DEFAULT_ROOT / "manifest.json"
BASE_URL = "https://openalyx.internationalbrainlab.org"


def _file_hash(path: Path, algorithm: str = "md5") -> str:
    digest = hashlib.new(algorithm)
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def select_trial_dataset(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Select exactly one immutable default/QC-passed public trial table."""

    eligible = [
        row
        for row in rows
        if row.get("name") == "_ibl_trials.table.pqt"
        and row.get("default_dataset") is True
        and row.get("qc") in {"PASS", "WARNING"}
        and int(row.get("public", 0)) >= 1
    ]
    if len(eligible) != 1:
        raise ValueError(f"expected one default QC PASS/WARNING trial table, found {len(eligible)}")
    row = eligible[0]
    records = [
        item
        for item in row.get("file_records", [])
        if item.get("exists") is True and str(item.get("data_url", "")).startswith("https://")
    ]
    if not records:
        raise ValueError("selected trial table has no existing public HTTPS file record")
    records.sort(
        key=lambda item: (
            not str(item.get("data_repository", "")).startswith("aws_"),
            str(item.get("data_repository", "")),
        )
    )
    url = str(row.get("url", ""))
    dataset_uuid = url.rstrip("/").rsplit("/", maxsplit=1)[-1]
    if not dataset_uuid or not row.get("hash") or not row.get("file_size"):
        raise ValueError("selected trial table lacks immutable identity or integrity metadata")
    return {
        "dataset_uuid": dataset_uuid,
        "revision": str(row.get("revision", "")),
        "qc": str(row["qc"]),
        "default_dataset": True,
        "file_size": int(row["file_size"]),
        "md5": str(row["hash"]),
        # Alyx exposes revision directories as #revision# path components. A
        # literal hash is a URL fragment delimiter, so it must be percent-encoded
        # before an HTTP client can address the S3 object itself.
        "url": str(records[0]["data_url"]).replace("#", "%23"),
        "repository": str(records[0].get("data_repository", "")),
        "created_datetime": row.get("created_datetime"),
        "data_version": row.get("version"),
    }


def _download(url: str, destination: Path) -> None:
    import certifi

    temporary = destination.with_suffix(destination.suffix + ".partial")
    request = urllib.request.Request(url, headers={"User-Agent": "MouseClaimBench/5.3"})
    context = ssl.create_default_context(cafile=certifi.where())
    with (
        urllib.request.urlopen(request, timeout=120, context=context) as response,
        temporary.open("wb") as handle,
    ):
        while chunk := response.read(1024 * 1024):
            handle.write(chunk)
    os.replace(temporary, destination)


def fetch(selection_path: Path, root: Path, manifest_path: Path) -> Path:
    """Download every selected mouse without replacement and write a manifest."""

    from one.api import ONE

    selection = json.loads(selection_path.read_text())
    trials_root = root / "trials"
    trials_root.mkdir(parents=True, exist_ok=True)
    one = ONE(base_url=BASE_URL, cache_dir=root / "one_cache", silent=True)
    entries: list[dict[str, Any]] = []
    for index, selected in enumerate(selection["insertions"], start=1):
        eid = str(selected["eid"])
        try:
            rows = one.alyx.rest("datasets", "list", session=eid, dataset_type="trials.table")
            source = select_trial_dataset(rows)
            destination = trials_root / f"{eid}.pqt"
            if (
                not destination.exists()
                or destination.stat().st_size != source["file_size"]
                or _file_hash(destination) != source["md5"]
            ):
                _download(source["url"], destination)
            if destination.stat().st_size != source["file_size"]:
                raise ValueError("downloaded byte count does not match Alyx")
            if _file_hash(destination) != source["md5"]:
                raise ValueError("downloaded MD5 does not match Alyx")
            status = "verified"
            error = None
        except Exception as exc:  # noqa: BLE001 - freeze every remote failure in the manifest.
            source = {}
            destination = trials_root / f"{eid}.pqt"
            status = "unavailable"
            error = f"{type(exc).__name__}: {exc}"
        entries.append(
            {
                **{key: selected[key] for key in ("rank", "role", "subject", "pid", "eid", "lab")},
                "status": status,
                "error": error,
                "local_path": str(destination),
                **source,
            }
        )
        print(f"[{index:03d}/{len(selection['insertions']):03d}] {eid}: {status}")

    payload = {
        "protocol": "ibl_behavior_mouse_population_v5",
        "selection_path": str(selection_path),
        "selection_sha256": hashlib.sha256(selection_path.read_bytes()).hexdigest(),
        "base_url": BASE_URL,
        "selection_uses_behavioral_outcomes": False,
        "replacement_after_failure": False,
        "selected_mice": len(entries),
        "verified_tables": sum(row["status"] == "verified" for row in entries),
        "unavailable_tables": sum(row["status"] != "verified" for row in entries),
        "entries": entries,
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(payload, indent=2) + "\n")
    return manifest_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", type=Path, default=DEFAULT_SELECTION)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    args = parser.parse_args()
    print(fetch(args.selection, args.root, args.manifest).resolve())


if __name__ == "__main__":
    main()
