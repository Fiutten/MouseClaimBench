import numpy as np

from mousebrainbench.benchmarks.ibl_behavior_v5_confirmation import (
    candidate_diagnostic,
    deterministic_trial_folds,
)


def test_trial_folds_are_deterministic_complete_and_eid_specific() -> None:
    indices = np.arange(300)
    first = deterministic_trial_folds("eid-a", indices)
    second = deterministic_trial_folds("eid-a", indices)
    other = deterministic_trial_folds("eid-b", indices)
    assert np.array_equal(first, second)
    assert set(first) == {0, 1, 2}
    assert not np.array_equal(first, other)


def test_actual_alignment_outperforms_circular_control() -> None:
    rng = np.random.default_rng(42)
    n = 900
    contrast = rng.choice([-1.0, -0.5, 0.0, 0.5, 1.0], size=n)
    probability_left = rng.choice([0.2, 0.5, 0.8], size=n)
    logits = 3.0 * contrast + 0.8 * (0.5 - probability_left)
    choice = (rng.random(n) < (1.0 / (1.0 + np.exp(-logits)))).astype(np.uint8)
    folds = deterministic_trial_folds("synthetic-eid", np.arange(n))
    actual = candidate_diagnostic(
        choice, contrast, probability_left, folds, offset=0
    )
    shifted = candidate_diagnostic(
        choice, np.roll(contrast, 43), probability_left, folds, offset=43
    )
    assert actual.tjur_r_squared > shifted.tjur_r_squared + 0.20
    assert actual.mean_log_loss_improvement > shifted.mean_log_loss_improvement
