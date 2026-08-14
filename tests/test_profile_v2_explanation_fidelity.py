import json
from pathlib import Path

import yaml

from mousebrainbench.benchmarks.profile_v2_explanation_fidelity import evaluate


def test_explanation_repairs_are_sufficient_minimal_and_traceable() -> None:
    protocol = yaml.safe_load(
        Path("configs/benchmarks/profile_v2_explanation_fidelity.yaml").read_text()
    )
    result = evaluate(protocol)

    assert result["random_packages"] == 10000
    assert result["repair_sufficiency_checks"] > 0
    assert result["repair_minimality_checks"] > result["repair_sufficiency_checks"]
    assert result["witness_checks"] > 0
    assert result["pristine_degradation_checks"] == 60
    assert result["multi_deficit_packages"] > 0
    assert result["deficits_hidden_by_single_reason_trace"] > 0
    assert result["all_explanation_properties_hold"] is True


def test_frozen_explanation_fidelity_is_complete_and_clean() -> None:
    payload = json.loads(Path("results/profile_v2_explanation_fidelity/summary.json").read_text())

    assert payload["decision"] == "counterfactual_explanation_fidelity_confirmed"
    assert payload["all_explanation_properties_hold"] is True
    assert not payload["git_revision"].endswith("-dirty")
