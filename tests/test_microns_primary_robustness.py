import numpy as np

from mousebrainbench.benchmarks.microns_primary_robustness import _shuffle_within_distance_bins


def test_shuffle_within_distance_bins_preserves_positive_signal_shape() -> None:
    rng = np.random.default_rng(1)
    values = np.array([3.0, 2.8, 0.0, 0.1, 1.0, 1.1])
    connected = np.array([True, True, False, False, False, False])
    bins = np.array(["a", "a", "a", "a", "b", "b"])
    observed = float(values[connected].mean())

    result = _shuffle_within_distance_bins(
        values=values,
        connected_mask=connected,
        distance_bins=bins,
        observed=observed,
        n_permutations=50,
        rng=rng,
    )

    assert result["delta"] > 0
    assert 0 <= result["p_one_sided"] <= 1
