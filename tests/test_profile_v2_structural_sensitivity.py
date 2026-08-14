import json
from pathlib import Path

import yaml

from mousebrainbench.benchmarks.profile_v2_structural_sensitivity import (
    build_variants,
    evaluate,
)


def test_structural_sensitivity_exhausts_single_relation_changes() -> None:
    protocol = yaml.safe_load(
        Path("configs/benchmarks/profile_v2_structural_sensitivity.yaml").read_text()
    )
    result = evaluate(protocol)

    assert len(build_variants()) == 221
    assert result["fixed_cases"] == 14
    assert result["relation_removal_variants"] == 60
    assert result["relation_addition_variants"] == 160
    assert result["profile_case_evaluations"] == 3094
    assert result["completed"] is True


def test_frozen_structural_sensitivity_is_complete_and_clean() -> None:
    payload = json.loads(Path("results/profile_v2_structural_sensitivity/summary.json").read_text())

    assert payload["decision"] == "structural_policy_sensitivity_completed"
    assert payload["relation_removal_variants"] == 60
    assert payload["relation_addition_variants"] == 160
    assert payload["profile_case_evaluations"] == 3094
    assert not payload["git_revision"].endswith("-dirty")
