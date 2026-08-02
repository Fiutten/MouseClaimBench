import numpy as np

from mousebrainbench.benchmarks.direction_router_precondition_audit import (
    direction_metrics,
)


def test_direction_metrics_counts_spurious_attempts() -> None:
    regimes = np.asarray(
        ["independent_heavy_tailed", "direct_sigmoid_additive"]
    )
    attempted = np.asarray([True, True])
    predicted = np.asarray(["forward", "forward"])

    metrics = direction_metrics(regimes, attempted, predicted)

    assert metrics["attempts"] == 2
    assert metrics["spurious_attempts_without_reference_direction"] == 1
    assert metrics["attempted_accuracy"] == 0.5
