"""Synchronize the canonical Overleaf-root manuscript into ``paper/``."""

from __future__ import annotations

import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MIRROR = ROOT / "paper"


def run() -> None:
    """Copy canonical manuscript files without changing their contents."""

    for filename in ("main.tex", "references.bib"):
        (MIRROR / filename).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / filename, MIRROR / filename)
    for directory in ("sections", "tables", "figures"):
        destination = MIRROR / directory
        destination.mkdir(parents=True, exist_ok=True)
        for source in (ROOT / directory).iterdir():
            if source.is_file():
                shutil.copy2(source, destination / source.name)


if __name__ == "__main__":
    run()
