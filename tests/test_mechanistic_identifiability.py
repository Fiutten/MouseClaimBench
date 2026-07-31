from mousebrainbench.validation.mechanistic_identifiability import (
    Criterion,
    build_mis_from_blocks,
)


def test_criterion_preserves_binary_pass_fail() -> None:
    strong = Criterion("strong_positive", 0.8, 0.5)
    weak = Criterion("weak_positive", 0.2, 0.5)

    assert strong.passed
    assert strong.normalized_score == 1.0
    assert not weak.passed
    assert weak.normalized_score == 0.4


def test_mis_requires_all_non_interchangeable_blocks() -> None:
    mis = build_mis_from_blocks(
        reproducibility=(Criterion("reproducible", 0.9, 0.5),),
        topology_specificity=(Criterion("not_topology_specific", 0.0, 0.1),),
        directed_identifiability=(Criterion("directed", 0.9, 0.5),),
    )

    assert not mis.passed
    assert mis.as_dict()["interpretation"] == "not_mechanistically_identifiable"

