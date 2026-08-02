from mousebrainbench.benchmarks.causal_chambers_transport import (
    direct_and_control_pairs,
    experiment_partition,
)


def test_experiment_partition_is_stable_and_disjoint() -> None:
    assert experiment_partition("lt", "analog_calibration") == "locked_test"
    assert experiment_partition("lt", "current_sensor") == "calibration_context"
    assert experiment_partition("wt", "mic_effects") == "locked_test"


def test_controls_exclude_edges_and_reverse_edges() -> None:
    columns = ("a", "b", "c", "d")
    edges = {("a", "b"), ("b", "c")}
    pairs = direct_and_control_pairs(columns, edges, namespace="test")
    direct = {(source, target) for source, target, is_edge in pairs if is_edge}
    controls = {(source, target) for source, target, is_edge in pairs if not is_edge}

    assert direct == edges
    assert len(controls) == len(direct)
    assert not controls & edges
    assert not controls & {(target, source) for source, target in edges}
