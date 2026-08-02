import numpy as np

from mousebrainbench.validation.shift_diagnostics import diagnose_shift


def test_shift_diagnostic_is_warning_only() -> None:
    rng = np.random.default_rng(7)
    calibration = rng.normal(size=(80, 3))
    target = rng.normal(loc=4.0, size=(80, 3))
    result = diagnose_shift(calibration, target, permutations=19, alpha=0.10)
    assert result["warning"]
    assert result["role"] == "warning_only"
    assert result["may_authorize_or_restore_certificate"] is False


def test_equal_population_does_not_force_warning() -> None:
    rng = np.random.default_rng(11)
    calibration = rng.normal(size=(120, 2))
    result = diagnose_shift(calibration, calibration.copy(), permutations=19, alpha=0.01)
    assert not result["warning"]

