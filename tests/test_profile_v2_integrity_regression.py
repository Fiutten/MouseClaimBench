import json
from pathlib import Path

from mousebrainbench.benchmarks.profile_v2_integrity_regression import (
    EXTENDED_REGRESSION_FAMILIES,
    evaluate,
    generate_cases,
)


def test_extended_integrity_regression_is_complete_and_exact() -> None:
    cases = generate_cases()
    assert len(cases) == 100
    assert len(EXTENDED_REGRESSION_FAMILIES) == 9
    assert {case.attack for case in cases if case.attack is not None} == set(
        EXTENDED_REGRESSION_FAMILIES
    )

    result = evaluate(
        {
            "design": {"expected_cases": 100},
            "interpretation": "declared integrity regressions only",
        }
    )
    assert result["exact_integrity_trace_rate"] == 1.0
    assert result["false_authorizations"] == 0
    assert result["false_rejections"] == 0
    assert result["regression_families"] == list(EXTENDED_REGRESSION_FAMILIES)
    assert "attack_families" not in result
    assert all(result["endpoints"].values())


def test_frozen_extended_regression_uses_canonical_units_and_clean_source() -> None:
    payload = json.loads(
        Path("results/profile_v2_integrity_regression/summary.json").read_text()
    )

    assert payload["cases"] == 100
    assert payload["pristine_cases"] == 10
    assert payload["attacked_cases"] == 90
    assert payload["regression_families"] == list(EXTENDED_REGRESSION_FAMILIES)
    assert "attack_families" not in payload
    assert not payload["git_revision"].endswith("-dirty")
