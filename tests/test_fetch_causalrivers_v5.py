import hashlib
import zipfile
from pathlib import Path

import pytest

from scripts.fetch_causalrivers_v5 import extract_product, verify_product


def _digest(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def test_extract_product_accepts_only_frozen_verified_members(tmp_path: Path) -> None:
    expected = {"graph.p": _digest(b"graph"), "series.csv": _digest(b"time,value\n")}
    archive = tmp_path / "source.zip"
    with zipfile.ZipFile(archive, "w") as bundle:
        bundle.writestr("product/graph.p", b"graph")
        bundle.writestr("product/series.csv", b"time,value\n")
        bundle.writestr("unrelated/ignored.txt", b"ignore")

    product = extract_product(archive, tmp_path / "data", expected)

    verify_product(product, expected)
    assert sorted(path.name for path in product.iterdir()) == ["graph.p", "series.csv"]


def test_verify_product_rejects_substituted_bytes(tmp_path: Path) -> None:
    product = tmp_path / "product"
    product.mkdir()
    (product / "graph.p").write_bytes(b"substituted")

    with pytest.raises(RuntimeError, match="hash mismatch"):
        verify_product(product, {"graph.p": _digest(b"expected")})


def test_extract_product_rejects_incomplete_archive(tmp_path: Path) -> None:
    archive = tmp_path / "source.zip"
    with zipfile.ZipFile(archive, "w") as bundle:
        bundle.writestr("product/graph.p", b"graph")

    with pytest.raises(RuntimeError, match="lacks frozen members"):
        extract_product(
            archive,
            tmp_path / "data",
            {"graph.p": _digest(b"graph"), "series.csv": _digest(b"series")},
        )
