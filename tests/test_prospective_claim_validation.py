import json

from mousebrainbench.benchmarks.prospective_claim_validation import (
    REGIME_TRUTHS,
    ProspectiveCell,
    _build_blocks,
    run,
)
from mousebrainbench.validation.evidence_contract import blocks_by_name


def test_reference_truth_is_declared_independently_of_diagnostic_statuses() -> None:
    cell = ProspectiveCell("weak_direct", 150, 1.8)
    blocks = blocks_by_name(_build_blocks(cell, case_seed=99))

    assert "mechanistic" in REGIME_TRUTHS[cell.regime]
    assert set(REGIME_TRUTHS["independent_student"]) == {
        "computationally_reproducible"
    }
    assert {block.status.value for block in blocks.values()} <= {
        "passed",
        "failed",
        "not_applicable",
    }


def test_scaled_test_run_is_deterministic_and_not_protocol_eligible(tmp_path) -> None:
    first = run(
        output=tmp_path / "first.json",
        markdown=tmp_path / "first.md",
        test_mode=True,
        test_seeds_per_cell=2,
        test_bootstrap_replicates=50,
    )
    second = run(
        output=tmp_path / "second.json",
        markdown=tmp_path / "second.md",
        test_mode=True,
        test_seeds_per_cell=2,
        test_bootstrap_replicates=50,
    )
    left = json.loads(first.read_text())
    right = json.loads(second.read_text())

    assert left["num_cases"] == 18
    assert left["scale_matches_frozen_protocol"] is False
    assert left["primary_endpoint"]["passed"] is False
    assert left["aggregate_by_policy"] == right["aggregate_by_policy"]
    assert left["case_level_policy_comparisons"] == right[
        "case_level_policy_comparisons"
    ]
    assert left["prospective_model_refitting_performed"] is False

