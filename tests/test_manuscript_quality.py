import re
from pathlib import Path


ROOT = Path(__file__).parents[1]
MANUSCRIPT_PATHS = [
    ROOT / "main.tex",
    *sorted((ROOT / "sections").glob("*.tex")),
    *sorted((ROOT / "tables").glob("*.tex")),
    *sorted((ROOT / "figures").glob("*.tex")),
]


def _without_comments(text: str) -> str:
    return "\n".join(re.sub(r"(?<!\\)%.*$", "", line) for line in text.splitlines())


def _citation_keys(text: str) -> set[str]:
    keys: set[str] = set()
    for match in re.finditer(r"\\citep(?:\[[^]]*\]){0,2}\{([^}]+)\}", text):
        keys.update(key.strip() for key in match.group(1).split(","))
    return keys


def test_abstract_is_one_paragraph_and_within_elsevier_limit() -> None:
    abstract = (ROOT / "sections" / "abstract.tex").read_text(encoding="utf-8").strip()
    assert "\n\n" not in abstract

    plain = re.sub(r"\\(?:ac|acs)\{([^}]+)\}", r"\1", abstract)
    plain = re.sub(r"\\[%&]", " ", plain)
    words = re.findall(r"[A-Za-z0-9]+(?:[-'][A-Za-z0-9]+)*", plain)
    assert len(words) <= 250


def test_all_bibliography_entries_are_cited_and_all_citations_exist() -> None:
    manuscript = "\n".join(
        _without_comments(path.read_text(encoding="utf-8")) for path in MANUSCRIPT_PATHS
    )
    cited = _citation_keys(manuscript)
    bibliography = (ROOT / "references.bib").read_text(encoding="utf-8")
    available = set(re.findall(r"@\w+\s*\{\s*([^,\s]+)", bibliography))

    assert cited == available


def test_manuscript_style_and_scope_guards() -> None:
    manuscript = "\n".join(path.read_text(encoding="utf-8") for path in MANUSCRIPT_PATHS)
    assert ";" not in manuscript
    assert "Mouse-Brain" in (ROOT / "main.tex").read_text(encoding="utf-8")
    assert "Human decision benefit remains untested" in manuscript


def test_numbered_sections_and_subsections_have_descriptive_labels() -> None:
    for path in sorted((ROOT / "sections").glob("*.tex")):
        lines = path.read_text(encoding="utf-8").splitlines()
        for index, line in enumerate(lines):
            if re.match(r"\\(?:sub)*section\{", line):
                following = "\n".join(lines[index + 1 : index + 3])
                assert re.search(r"\\label\{(?:sec|subsec):[a-z0-9-]+\}", following)
