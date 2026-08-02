from pathlib import Path

import yaml

from mousebrainbench.benchmarks.semantic_risk_orthogonal_shifts_v5_4 import (
    _limits,
)


def test_protocol_declares_four_orthogonal_families_and_fourteen_levels() -> None:
    protocol = yaml.safe_load(
        Path("configs/benchmarks/semantic_risk_orthogonal_shifts_v5_4.yaml").read_text()
    )
    assert set(protocol["families"]) == {
        "tail_heaviness",
        "dimensionality",
        "temporal_depth",
        "latent_confounding",
    }
    levels = [level for family in protocol["families"].values() for level in family["levels"]]
    assert len(levels) == 14
    assert len(set(levels)) == 14


def test_risk_confidence_is_bonferroni_adjusted_across_all_levels() -> None:
    protocol = yaml.safe_load(
        Path("configs/benchmarks/semantic_risk_orthogonal_shifts_v5_4.yaml").read_text()
    )
    assert _limits(protocol)["confidence"] == 1.0 - 0.05 / 14
