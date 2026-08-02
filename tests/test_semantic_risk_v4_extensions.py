from pathlib import Path

import numpy as np
import yaml

from mousebrainbench.benchmarks.knowledge_profile_validity_v4 import evaluate
from mousebrainbench.benchmarks.semantic_risk_v4_power import (
    maximum_compatible_failures,
    minimum_successes_for_lower_bound,
    minimum_units_for_zero_failures,
)
from mousebrainbench.validation.direction_router import association_precondition


def test_prospective_power_constraints_are_exact() -> None:
    assert minimum_units_for_zero_failures(0.10, 0.95) == 29
    assert maximum_compatible_failures(29, 0.10, 0.95) == 0
    assert maximum_compatible_failures(100, 0.10, 0.95) >= 4
    assert minimum_successes_for_lower_bound(100, 0.10, 0.95) is not None


def test_association_precondition_separates_signal_and_independence() -> None:
    rng = np.random.default_rng(17)
    x = rng.normal(size=500)
    independent = rng.normal(size=500)
    associated = np.tanh(x) + rng.normal(0.0, 0.2, size=500)
    assert not association_precondition(x, independent)["established"]
    result = association_precondition(x, associated)
    assert result["established"]
    assert result["causal_or_directional_evidence"] is False


def test_profile_validity_audit_does_not_claim_content_validation() -> None:
    basis = yaml.safe_load(
        Path("mousebrainbench/knowledge/profiles/mouse_brain_claims_v1_basis.yaml").read_text()
    )
    protocol = yaml.safe_load(
        Path("configs/benchmarks/semantic_risk_control_v4.yaml").read_text()
    )
    result = evaluate(basis, protocol)
    assert result["structural_documentation_complete"]
    assert result["profile_content_validated"] is False
    assert result["decision"].endswith("not_content_validated")
