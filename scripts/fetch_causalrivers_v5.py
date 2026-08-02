"""Download and verify the CausalRivers files frozen by the v5 protocol."""

from __future__ import annotations

import argparse
import hashlib
import shutil
import urllib.request
import zipfile
from pathlib import Path, PurePosixPath
from typing import Any

import yaml

DEFAULT_PROTOCOL = Path("configs/benchmarks/causalrivers_v5_transport.yaml")
DEFAULT_ROOT = Path("data/external/causalrivers_v5")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _expected_files(protocol: dict[str, Any]) -> dict[str, str]:
    return {str(name): str(digest) for name, digest in protocol["source"]["file_sha256"].items()}


def verify_product(product: Path, expected: dict[str, str]) -> None:
    """Reject missing, substituted, or corrupted source files."""

    for name, digest in expected.items():
        path = product / name
        if not path.is_file():
            raise FileNotFoundError(f"missing CausalRivers file: {path}")
        observed = _sha256(path)
        if observed != digest:
            raise RuntimeError(f"CausalRivers hash mismatch for {name}: {observed}")


def extract_product(archive: Path, root: Path, expected: dict[str, str]) -> Path:
    """Extract only protocol-listed members and verify their bytes."""

    product = root / "product"
    product.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as bundle:
        members: dict[str, str] = {}
        for member in bundle.namelist():
            parts = PurePosixPath(member).parts
            if len(parts) == 2 and parts[0] == "product" and parts[1] in expected:
                members[parts[1]] = member
        missing = sorted(set(expected) - set(members))
        if missing:
            raise RuntimeError(f"CausalRivers archive lacks frozen members: {', '.join(missing)}")
        for name, member in members.items():
            with bundle.open(member) as source, (product / name).open("wb") as target:
                shutil.copyfileobj(source, target)
    verify_product(product, expected)
    return product


def fetch(protocol_path: Path, root: Path) -> Path:
    """Reuse a verified product or download, hash-check, and safely extract it."""

    protocol = yaml.safe_load(protocol_path.read_text())
    source = protocol["source"]
    expected = _expected_files(protocol)
    product = root / "product"
    try:
        verify_product(product, expected)
        return product
    except FileNotFoundError:
        pass

    root.mkdir(parents=True, exist_ok=True)
    archive = root / "product.zip"
    if not archive.exists():
        partial = root / "product.zip.partial"
        request = urllib.request.Request(
            str(source["release_asset"]),
            headers={"User-Agent": "MouseClaimBench-reproducibility-fetcher/1.0"},
        )
        with urllib.request.urlopen(request) as response, partial.open("wb") as target:
            shutil.copyfileobj(response, target)
        partial.replace(archive)
    observed_archive = _sha256(archive)
    if observed_archive != str(source["archive_sha256"]):
        raise RuntimeError(f"CausalRivers archive hash mismatch: {observed_archive}")
    return extract_product(archive, root, expected)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    args = parser.parse_args()
    print(fetch(args.protocol, args.root).resolve())


if __name__ == "__main__":
    main()
