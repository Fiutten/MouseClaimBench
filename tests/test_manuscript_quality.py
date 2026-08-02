import re
from pathlib import Path

from mousebrainbench.knowledge import load_authorization_profile_v2_basis
from scripts.build_elsevier_submission import _resolve_tex_dependencies

ROOT = Path(__file__).parents[1]
MANUSCRIPT_PATHS, _ = _resolve_tex_dependencies(ROOT / "main.tex")


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

    main = (ROOT / "main.tex").read_text(encoding="utf-8")
    acronyms = {
        key: f"{long_name} ({short_name})"
        for key, short_name, long_name in re.findall(
            r"\\acrodef\{([^}]+)\}\[([^]]+)\]\{([^}]+)\}", main
        )
    }
    plain = re.sub(
        r"\\ac\{([^}]+)\}",
        lambda match: acronyms[match.group(1)],
        abstract,
    )
    plain = re.sub(r"\\acs\{([^}]+)\}", lambda match: match.group(1), plain)
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


def test_profile_curation_sources_exist_in_the_manuscript_bibliography() -> None:
    bibliography = (ROOT / "references.bib").read_text(encoding="utf-8")
    available = set(re.findall(r"@\w+\s*\{\s*([^,\s]+)", bibliography))
    basis_sources = {
        source_id
        for relation in load_authorization_profile_v2_basis()["relations"]
        for source_id in relation["source_ids"]
    }

    assert basis_sources <= available


def test_manuscript_style_and_scope_guards() -> None:
    manuscript = "\n".join(path.read_text(encoding="utf-8") for path in MANUSCRIPT_PATHS)
    main = (ROOT / "main.tex").read_text(encoding="utf-8")
    assert ";" not in manuscript
    assert "Mouse-Brain" in main
    assert "\\journal{Knowledge-Based Systems}" in main
    assert "Decision Support Systems" not in manuscript
    assert "do not support a claim that the system determines scientific truth" in manuscript
    assert "External validity is not established outside computational mouse-brain evidence" in manuscript


def test_kbs_keywords_and_highlights_follow_elsevier_limits() -> None:
    main = (ROOT / "main.tex").read_text(encoding="utf-8")
    keyword_block = re.search(
        r"\\begin\{keyword\}(.*?)\\end\{keyword\}", main, flags=re.DOTALL
    )
    assert keyword_block is not None
    keywords = [value.strip() for value in keyword_block.group(1).split(r"\sep")]
    assert 1 <= len(keywords) <= 7
    assert all(keyword and keyword[0].isupper() for keyword in keywords)

    highlights = [
        line.removeprefix("- ").strip()
        for line in (ROOT / "highlights.txt").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert 3 <= len(highlights) <= 5
    assert all(len(highlight) <= 85 for highlight in highlights)


def test_submission_declarations_and_title_are_present() -> None:
    main = (ROOT / "main.tex").read_text(encoding="utf-8")
    title = re.search(r"\\title\{([^}]+)\}", main)
    assert title is not None
    assert len(title.group(1).split()) <= 15
    assert "Declaration of Competing Interest" in main
    assert "CRediT Authorship Contribution Statement" in main
    assert "Data and Code Availability" in main
    assert "Declaration of Generative AI and AI-Assisted Technologies" in main


def test_every_used_acronym_is_defined() -> None:
    main = (ROOT / "main.tex").read_text(encoding="utf-8")
    manuscript = "\n".join(path.read_text(encoding="utf-8") for path in MANUSCRIPT_PATHS)
    defined = set(re.findall(r"\\acrodef\{([^}]+)\}", main))
    used = set(re.findall(r"\\ac(?:s)?\{([^}]+)\}", manuscript))
    assert used <= defined


def test_numbered_sections_and_subsections_have_descriptive_labels() -> None:
    for path in sorted((ROOT / "sections").glob("*.tex")):
        lines = path.read_text(encoding="utf-8").splitlines()
        for index, line in enumerate(lines):
            if re.match(r"\\(?:sub)*section\{", line):
                following = "\n".join(lines[index + 1 : index + 3])
                assert re.search(r"\\label\{(?:sec|subsec):[a-z0-9-]+\}", following)
