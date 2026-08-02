import numpy as np
import pandas as pd

from mousebrainbench.benchmarks.ibl_mouse_transport import (
    ELECTROPHYSIOLOGY_FEATURES,
    PROHIBITED_PREDICTORS,
    deterministic_unit_folds,
    electrophysiology_matrix,
    insertion_feature_and_label,
    prediction_diagnostic,
    subject_partition_value,
)
from mousebrainbench.knowledge import load_default_profile


def test_subject_partition_is_stable() -> None:
    subject = "00000000-0000-0000-0000-000000000001"
    assert subject_partition_value(subject) == subject_partition_value(subject)
    assert 0 <= subject_partition_value(subject) <= 4


def test_unit_folds_are_disjoint_and_balanced() -> None:
    first = deterministic_unit_folds([f"unit-{i}" for i in range(50)], namespace="pid")
    second = deterministic_unit_folds([f"unit-{i}" for i in range(50)], namespace="pid")

    assert all(np.array_equal(a, b) for a, b in zip(first, second, strict=True))
    assert max(map(len, first)) - min(map(len, first)) <= 1
    assert len(set(np.concatenate(first))) == 50


def test_predictors_exclude_every_anatomical_leakage_field() -> None:
    assert not set(ELECTROPHYSIOLOGY_FEATURES) & PROHIBITED_PREDICTORS
    frame = pd.DataFrame(
        {name: [1.0, 2.0, np.nan] for name in ELECTROPHYSIOLOGY_FEATURES}
    )
    frame["x"] = [100.0, 200.0, 300.0]
    matrix = electrophysiology_matrix(frame)

    assert matrix.shape == (3, len(ELECTROPHYSIOLOGY_FEATURES))
    assert 100.0 not in matrix


def test_prediction_diagnostic_detects_multiclass_alignment() -> None:
    observed = np.repeat(np.asarray(["A", "B", "C"]), 20)
    result = prediction_diagnostic(observed, observed.copy(), seed=42)

    assert result["passed"] is True
    assert result["matthews_correlation"] == 1.0
    assert result["balanced_accuracy"] == 1.0


def test_reference_result_changes_label_but_not_policy_feature() -> None:
    claims = tuple(item.claim for item in load_default_profile().requirements)
    passed = {
        "passed": True,
        "correlation": 0.8,
        "p_value": 0.005,
        "r_squared": 0.0,
    }
    failed = {
        "passed": False,
        "correlation": 0.0,
        "p_value": 1.0,
        "r_squared": 0.0,
    }
    insertion = {"pid": "p", "subject_uuid": "s"}
    positive_feature, positive_label = insertion_feature_and_label(
        insertion=insertion,
        first=passed,
        second=passed,
        reference=passed,
        sample_size=20,
        claim_names=claims,
    )
    negative_feature, negative_label = insertion_feature_and_label(
        insertion=insertion,
        first=passed,
        second=passed,
        reference=failed,
        sample_size=20,
        claim_names=claims,
    )

    assert np.array_equal(positive_feature, negative_feature)
    assert positive_label[claims.index("predictive")] == 1
    assert negative_label[claims.index("predictive")] == 0
