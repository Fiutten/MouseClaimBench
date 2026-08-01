from pathlib import Path


def test_overleaf_root_and_structured_paper_mirror_are_identical() -> None:
    pairs = [
        (Path("main.tex"), Path("paper/main.tex")),
        (Path("references.bib"), Path("paper/references.bib")),
    ]
    for directory in ("sections", "tables", "figures"):
        for root_path in sorted(Path(directory).glob("*")):
            if root_path.is_file():
                pairs.append((root_path, Path("paper") / root_path))

    mismatches = [
        str(left)
        for left, right in pairs
        if not right.exists() or left.read_bytes() != right.read_bytes()
    ]

    assert mismatches == []
