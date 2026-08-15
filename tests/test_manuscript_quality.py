import csv
import re
import tomllib
from collections import Counter
from pathlib import Path

from mousebrainbench.knowledge import (
    load_authorization_profile_v2,
    load_authorization_profile_v2_basis,
)
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


def test_engine_scope_and_property_claims_are_calibrated() -> None:
    abstract = (ROOT / "sections" / "abstract.tex").read_text(encoding="utf-8")
    manuscript = "\n".join(
        path.read_text(encoding="utf-8") for path in MANUSCRIPT_PATHS
    )
    assert (
        "Python and a separately implemented \\ac{asp} path agree on domain "
        "authorization and deficits across 5,677 contract cases."
    ) in abstract
    assert "Python, \\ac{shacl}, and the separately implemented" not in abstract
    assert "formal verification results" not in manuscript.lower()
    assert "The evaluation is organized into four evidence classes." in manuscript
    assert "under the declared deterministic threat model" in manuscript.lower()


def test_second_review_formal_and_interpretive_corrections_are_present() -> None:
    manuscript = "\n".join(
        path.read_text(encoding="utf-8") for path in MANUSCRIPT_PATHS
    )
    method = (ROOT / "sections" / "method.tex").read_text(encoding="utf-8")
    experiments = (ROOT / "sections" / "experiments.tex").read_text(
        encoding="utf-8"
    )

    assert r"\mathrm{1}[I_{\Pi}(M,B)=\varnothing]" in method
    assert r"S_{\Pi}(G_{\Pi}(c,B,M))A_{\Pi}(c,B)" in method
    assert r"G_{\Pi}(c,B,M)=\operatorname{RDF}_{\Pi}(c,B,M)" in method
    assert r"T_{\Delta_{\Pi}(c,B)}" in method
    assert "2,550 non-empty compositions" not in manuscript
    assert "one-edge structural policy perturbation" in manuscript.lower()
    assert "core authorization-engine microbenchmark" in manuscript.lower()
    assert "p=1/1001\\approx0.001" in experiments
    assert "Relationship to the preceding MouseBrainBench study" in manuscript
    assert "Research contributions.." not in manuscript
    assert "Lessons learned.." not in manuscript


def test_bibliography_protects_reviewed_acronyms() -> None:
    bibliography = (ROOT / "references.bib").read_text(encoding="utf-8")

    assert "{{W3C} Recommendation" in bibliography
    assert "{{ASME} V\\&V 40-2018}" in bibliography
    assert "{{ICLR} 2026 submission" in bibliography


def test_claim_table_is_complete_and_case_counts_match_artifact() -> None:
    profile = load_authorization_profile_v2()
    table = (ROOT / "tables" / "profile_v2_claims.tex").read_text(encoding="utf-8")
    labels = (
        "Bounded predictive performance",
        "Clean computational reproduction",
        "Within-resource reproduction",
        "Independent-study replication",
        "Topology-specific prediction",
        "Assumption-conditional direction",
        "Local observational structure--function association",
        "Intervention-supported effect",
        "Directed topology-consistent prediction",
        "Complete entity-specific mouse-brain digital twin",
    )
    with (ROOT / "results/profile_v2_contract_mutation/cases.csv").open(
        encoding="utf-8", newline=""
    ) as handle:
        counts = Counter(row["claim"] for row in csv.DictReader(handle))

    assert len(profile.requirements) == len(labels) == 10
    assert sum(len(row.required_blocks) for row in profile.requirements) == 60
    assert sum(counts.values()) == 5_677
    for index, (requirement, label) in enumerate(
        zip(profile.requirements, labels, strict=True), start=1
    ):
        expected = (
            f"C{index} & {label} & {len(requirement.required_blocks)} & "
            f"{counts[requirement.claim]:,}"
        )
        assert expected in table


def test_integrity_threat_table_covers_all_eight_attacks() -> None:
    table = (ROOT / "tables" / "integrity_threat_model.tex").read_text(
        encoding="utf-8"
    )
    attacks = (
        "Profile-version substitution",
        "Artifact-hash tampering",
        "Dangling provenance reference",
        "Circular provenance",
        "Duplicate independent artifact",
        "Overlapping independent cohorts",
        "Contradictory attestation",
        "Missing block lineage",
    )
    assert all(attack in table for attack in attacks)
    assert "Block--attestation mismatch" in table
    assert "Missing block attestation" in table
    assert "Duplicate artifact identifier" in table
    assert "Duplicate block lineage" in table
    assert "Thirteen-type integrity model and evaluated coverage" in table
    assert "adaptive robustness" in table


def test_minor_revision_terminology_and_microns_roles_are_consistent() -> None:
    manuscript = "\n".join(
        path.read_text(encoding="utf-8") for path in MANUSCRIPT_PATHS
    )
    figure_source = (ROOT / "scripts" / "build_standards_prospective_figures.py").read_text(
        encoding="utf-8"
    )

    assert "Eight mandatory controls" not in manuscript
    assert "2,550 attacked compositions" not in manuscript
    assert "360 original attacks" not in manuscript
    assert "Full gate" not in figure_source
    assert "Full integrity gate" in figure_source
    assert "The discovery window fixes the positive direction" in manuscript
    assert "both pre-fixed hold-outs" in manuscript


def test_integrity_evaluation_units_use_canonical_nonconflated_terms() -> None:
    manuscript = "\n".join(
        path.read_text(encoding="utf-8") for path in MANUSCRIPT_PATHS
    )
    threat_table = (ROOT / "tables/integrity_threat_model.tex").read_text(
        encoding="utf-8"
    )
    design_table = (ROOT / "tables/experiment_design_v2.tex").read_text(
        encoding="utf-8"
    )

    assert "first eight attacks" not in manuscript.lower()
    assert "eight historical attack families" in manuscript.lower()
    assert "370-case original benchmark" in manuscript.lower()
    assert "2,550 attacked packages spanning all 255 compositions" in manuscript
    assert "100-package nine-family regression suite" in manuscript.lower()
    assert "six direct API edge cases" in manuscript
    assert "20 coherent-forgery controls" in manuscript
    assert "Evaluation" in threat_table
    assert "O: 370-case original benchmark" in threat_table
    assert "C: 2,550-package compositional stress test" in threat_table
    assert "E: 100-package extended regression suite" in threat_table
    assert "D: six direct API edge-case regressions" in threat_table
    assert "six direct API edge cases" in design_table


def test_release_version_is_consistent_across_user_facing_metadata() -> None:
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    version = project["project"]["version"]
    citation = (ROOT / "CITATION.cff").read_text(encoding="utf-8")
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    main = (ROOT / "main.tex").read_text(encoding="utf-8")

    assert version == "0.12.3"
    assert f"version: {version}" in citation
    assert f"v{version}" in readme
    assert f"v{version}" in main


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
