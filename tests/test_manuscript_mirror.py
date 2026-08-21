from pathlib import Path


def test_overleaf_root_and_structured_paper_mirror_are_identical() -> None:
    pairs = [
        (Path("main.tex"), Path("paper/main.tex")),
        (Path("highlights.txt"), Path("paper/highlights.txt")),
        (Path("figure_captions.txt"), Path("paper/figure_captions.txt")),
        (
            Path("related_manuscript_statement.txt"),
            Path("paper/related_manuscript_statement.txt"),
        ),
        (Path("references.bib"), Path("paper/references.bib")),
        (Path("elsarticle-num.bst"), Path("paper/elsarticle-num.bst")),
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
