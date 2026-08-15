import json
from pathlib import Path

import yaml

from mousebrainbench.benchmarks.profile_v2_compositional_integrity_stress import evaluate
from mousebrainbench.benchmarks.profile_v2_provenance_attacks import (
    ORIGINAL_ATTACK_FAMILIES,
)


def test_all_declared_attack_compositions_and_trust_boundary_controls() -> None:
    protocol = yaml.safe_load(
        Path("configs/benchmarks/profile_v2_compositional_integrity_stress.yaml").read_text()
    )
    result = evaluate(protocol)

    assert len(ORIGINAL_ATTACK_FAMILIES) == 8
    assert result["in_model_packages"] == 2560
    assert result["in_model_attacked_packages"] == 2550
    assert result["in_model_false_authorizations"] == 0
    assert result["in_model_exact_traces"] == 2560
    assert result["trust_boundary_negative_controls"] == 20
    assert result["trust_boundary_authorizations"] == 20
    assert result["all_endpoints_passed"] is True


def test_frozen_compositional_integrity_stress_is_complete_and_clean() -> None:
    payload = json.loads(
        Path("results/profile_v2_compositional_integrity_stress/summary.json").read_text()
    )

    assert payload["decision"] == ("declared_compositions_confirmed_with_explicit_trust_boundary")
    assert payload["all_endpoints_passed"] is True
    assert not payload["git_revision"].endswith("-dirty")
