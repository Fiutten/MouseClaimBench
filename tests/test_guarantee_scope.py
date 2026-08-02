import numpy as np

from mousebrainbench.validation.guarantee_scope import (
    PopulationScope,
    assess_guarantee_scope,
    enforce_guarantee_scope,
)


def _scope(identifier: str, family: str = "synthetic_sem") -> PopulationScope:
    return PopulationScope(
        scope_id=identifier,
        population_family=family,
        independent_unit="complete_case",
        evidence_protocol="evidence-v3",
        reference_protocol="oracle-v3",
    )


def test_identical_scope_preserves_authorizations() -> None:
    calibration = _scope("calibration")
    target = _scope("target")
    assessment = assess_guarantee_scope(calibration, target, detected_shift=False)
    decisions = np.asarray([[1, 0], [0, 1]], dtype=np.int8)

    assert assessment.valid is True
    assert np.array_equal(enforce_guarantee_scope(decisions, assessment), decisions)


def test_cross_domain_scope_forces_abstention() -> None:
    assessment = assess_guarantee_scope(
        _scope("calibration"),
        _scope("target", family="causalbench_rpe1"),
    )
    decisions = np.ones((3, 2), dtype=np.int8)

    assert assessment.valid is False
    assert "scope_mismatch:population_family" in assessment.blockers
    assert not enforce_guarantee_scope(decisions, assessment).any()


def test_detected_shift_invalidates_matching_population_scope() -> None:
    assessment = assess_guarantee_scope(
        _scope("calibration"),
        _scope("target"),
        detected_shift=True,
    )

    assert assessment.valid is False
    assert assessment.blockers == ("distribution_shift_detected",)
