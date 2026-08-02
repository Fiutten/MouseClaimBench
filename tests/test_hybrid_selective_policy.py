import json

import numpy as np

from mousebrainbench.benchmarks.hybrid_development_features import run as build_features
from mousebrainbench.benchmarks.hybrid_selective_policy import (
    clopper_pearson_upper,
    selective_decisions,
    train,
)


def _fake_direction(x, y, *, seed):
    del seed
    correlation = float(np.corrcoef(x, y)[0, 1])
    margin = 0.3 if correlation >= 0 else -0.3
    return {
        "status": "passed" if margin > 0 else "failed",
        "predicted_direction": "x_to_y" if margin > 0 else "y_to_x",
        "p_forward": 0.7 if margin > 0 else 0.1,
        "p_backward": 0.1 if margin > 0 else 0.7,
        "signed_margin": margin,
        "absolute_margin": abs(margin),
        "execution_error": None,
    }


def test_clopper_pearson_upper_is_conservative():
    assert 0.0 < clopper_pearson_upper(0, 100) < 0.04
    assert clopper_pearson_upper(10, 100) > 0.10
    assert clopper_pearson_upper(0, 0) == 1.0


def test_semantic_veto_cannot_be_overridden():
    names = (
        "status:causal_intervention:passed",
        "status:causal_intervention:failed",
        "status:causal_intervention:unknown",
        "status:causal_intervention:not_applicable",
        "status:causal_intervention:requires_review",
    )
    features = np.asarray([[0, 1, 0, 0, 0], [0, 0, 0, 0, 1]], dtype=float)
    probabilities = np.asarray([[0.99], [0.99]])
    decisions, violations = selective_decisions(
        probabilities,
        features,
        threshold=0.9,
        claim_names=("causal",),
        feature_names=names,
        support_vetoes={"causal": ("causal_intervention",)},
        constrained=True,
    )
    assert decisions[:, 0].tolist() == [-1, 0]
    assert violations == 0


def test_training_respects_disjoint_development_roles(tmp_path):
    matrix = tmp_path / "cases.npz"
    manifest = tmp_path / "summary.json"
    build_features(
        output=manifest,
        matrix=matrix,
        test_mode=True,
        direction_function=_fake_direction,
    )
    output = tmp_path / "model.json"
    train(output=output, matrix_path=matrix, manifest_path=manifest, test_mode=True)
    payload = json.loads(output.read_text())
    assert sum(payload["split_counts"].values()) == 70
    assert all(value > 0 for value in payload["split_counts"].values())
    assert payload["confirmatory_v2_cases_used"] == 0
    assert payload["confirmatory_refitting_permitted"] is False
    assert payload["runtime"]["packages"]["causal-learn"] == "0.1.4.8"
    assert payload["runtime"]["packages"]["scikit-learn"] == "1.9.0"
    assert payload["locked_development_audit"]["constrained_full"][
        "semantic_support_veto_violations"
    ] == 0
