from mousebrainbench.benchmarks.profile_v2_integrity_regression import (
    ATTACKS,
    evaluate,
    generate_cases,
)


def test_extended_integrity_regression_is_complete_and_exact() -> None:
    cases = generate_cases()
    assert len(cases) == 100
    assert {case.attack for case in cases if case.attack is not None} == set(ATTACKS)

    result = evaluate(
        {
            "design": {"expected_cases": 100},
            "interpretation": "declared integrity regressions only",
        }
    )
    assert result["exact_integrity_trace_rate"] == 1.0
    assert result["false_authorizations"] == 0
    assert result["false_rejections"] == 0
    assert all(result["endpoints"].values())
