import numpy as np

from mousebrainbench.benchmarks.direction_router_v4_2_confirmation import (
    _generate,
)
from mousebrainbench.validation.direction_router import association_precondition


def test_strict_association_rule_rejects_independent_sample() -> None:
    x, y = _generate(
        "independent_beta_exponential", 2000, 1.0, np.random.default_rng(21)
    )
    result = association_precondition(
        x, y, familywise_alpha=0.001, minimum_passing_tests=2
    )
    assert not result["established"]


def test_strict_association_rule_retains_direct_sample() -> None:
    x, y = _generate(
        "direct_arctan_additive", 800, 1.0, np.random.default_rng(23)
    )
    result = association_precondition(
        x, y, familywise_alpha=0.001, minimum_passing_tests=2
    )
    assert result["established"]
    assert result["minimum_passing_tests"] == 2

