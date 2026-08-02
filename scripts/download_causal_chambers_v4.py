"""Download and checksum only the datasets frozen in the v4 population manifest."""

from __future__ import annotations

import argparse
import hashlib
import shutil
import zipfile
from pathlib import Path

import requests
import yaml

BASE_URL = "https://causalchamber.s3.eu-central-1.amazonaws.com/downloadables"


def _md5(path: Path) -> str:
    digest = hashlib.md5(usedforsecurity=False)
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def run(manifest: Path, root: Path) -> None:
    payload = yaml.safe_load(manifest.read_text())
    root.mkdir(parents=True, exist_ok=True)
    for item in payload["datasets"]:
        name = str(item["name"])
        expected = str(item["md5"])
        archive = root / f"{name}.zip"
        destination = root / name
        if not archive.exists() or _md5(archive) != expected:
            temporary = archive.with_suffix(".zip.part")
            with requests.get(
                f"{BASE_URL}/{name}.zip", stream=True, timeout=(30, 300)
            ) as response:
                response.raise_for_status()
                with temporary.open("wb") as stream:
                    shutil.copyfileobj(response.raw, stream)
            temporary.replace(archive)
        observed = _md5(archive)
        if observed != expected:
            raise ValueError(f"checksum mismatch for {name}: {observed} != {expected}")
        if not destination.exists():
            with zipfile.ZipFile(archive) as compressed:
                compressed.extractall(root)
        if not destination.exists():
            raise FileNotFoundError(f"archive did not create expected directory: {destination}")
        print(f"verified {name} ({item['role']})")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("configs/benchmarks/causal_chambers_v4_population.yaml"),
    )
    parser.add_argument(
        "--root", type=Path, default=Path("data/external/causal_chambers_v4")
    )
    args = parser.parse_args()
    run(args.manifest, args.root)


if __name__ == "__main__":
    main()
