import json
from pathlib import Path

from mousebrainbench.benchmarks.profile_v2_contract_mutation import (
    _asp_selection,
    evaluate,
    generate_cases,
)


def _protocol():
    return {
        "asp_conformance": {
            "selection": "deterministic_per_family_subset",
            "maximum_cases_per_family": 8,
        },
        "interpretation": "contract conformance only",
    }


def test_mutation_design_covers_every_declared_family() -> None:
    cases = generate_cases()
    families = {case.family for case in cases}

    assert families == {
        "pristine_complete",
        "single_status_defect",
        "omitted_required_block",
        "missing_required_observation",
        "pairwise_mixed_status_defects",
    }
    assert len(cases) > 5_000
    assert len({case.case_id for case in cases}) == len(cases)


def test_asp_selection_is_deterministic_and_stratified() -> None:
    cases = generate_cases()
    first = _asp_selection(cases, 8)
    second = _asp_selection(tuple(reversed(cases)), 8)

    assert first == second
    for family in {case.family for case in cases}:
        available = sum(case.family == family for case in cases)
        selected = sum(
            case.case_id in first and case.family == family for case in cases
        )
        assert selected == min(available, 8)


def test_asp_selection_can_cover_every_generated_case() -> None:
    cases = generate_cases()

    assert _asp_selection(cases) == {case.case_id for case in cases}


def test_v2_passes_mutation_endpoints_and_shortcuts_fail() -> None:
    summary, _ = evaluate(generate_cases(), _protocol())

    assert summary["profile_v2"]["false_authorizations"] == 0
    assert summary["profile_v2"]["false_rejections"] == 0
    assert summary["profile_v2"]["exact_deficit_rate"] == 1.0
    assert summary["asp_conformance"]["rate"] == 1.0
    assert all(summary["primary_endpoints"].values())
    assert all(
        value > 0 for value in summary["comparator_false_authorizations"].values()
    )
    assert summary["prioritized_single_reason_trace"]["exact_deficit_rate"] < 1.0


def test_frozen_mutation_artifact_has_exact_counts_and_clean_revision() -> None:
    payload = json.loads(
        Path("results/profile_v2_contract_mutation/summary.json").read_text()
    )

    assert payload["cases"] == 5_497
    assert payload["mutation_cases"] == 5_487
    assert payload["profile_v2"]["false_authorizations"] == 0
    assert payload["profile_v2"]["exact_deficit_rate"] == 1.0
    assert payload["asp_conformance"] == {
        "cases": 5_497,
        "exact_status_and_deficit_matches": 5_497,
        "rate": 1.0,
    }
    assert not payload["git_revision"].endswith("-dirty")
