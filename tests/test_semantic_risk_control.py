import numpy as np

from mousebrainbench.validation.semantic_risk_control import (
    authorize_with_policy,
    calibrate_semantic_risk,
    semantic_false_authorization_metrics,
)


def test_ltt_calibrates_complete_semantic_policy_and_never_overrides_gate() -> None:
    rng = np.random.default_rng(20260820)
    labels = rng.random((2400, 2)) < np.asarray([0.55, 0.35])
    signal = np.clip(0.12 + 0.76 * labels + rng.normal(0.0, 0.08, labels.shape), 0.0, 1.0)
    admissible = np.ones_like(labels, dtype=bool)
    admissible[:1200, 1] = False

    policy = calibrate_semantic_risk(
        signal,
        labels,
        admissible,
        claim_names=("predictive", "causal"),
        target_sfar=0.05,
        familywise_confidence=0.95,
    )
    decisions = authorize_with_policy(policy, signal, admissible)

    assert all(item.certified for item in policy.certificates)
    assert not decisions[:1200, 1].any()
    assert (decisions[~admissible] == 0).all()
    assert semantic_false_authorization_metrics(decisions, labels)[
        "semantic_false_authorization_risk"
    ] <= 0.05


def test_uncertifiable_claim_abstains_instead_of_fabricating_a_threshold() -> None:
    scores = np.full((80, 1), 0.9)
    labels = np.zeros((80, 1), dtype=bool)
    admissible = np.ones((80, 1), dtype=bool)

    policy = calibrate_semantic_risk(
        scores,
        labels,
        admissible,
        claim_names=("causal",),
    )
    decisions = authorize_with_policy(policy, scores, admissible)

    assert not policy.certificates[0].certified
    assert policy.certificates[0].threshold is None
    assert decisions.sum() == 0


def test_metrics_use_authorized_supports_as_sfar_denominator() -> None:
    decisions = np.asarray([[1, 0], [1, 1], [0, 1]], dtype=np.int8)
    labels = np.asarray([[1, 0], [0, 1], [1, 0]], dtype=bool)

    metrics = semantic_false_authorization_metrics(decisions, labels)

    assert metrics["authorizations"] == 4
    assert metrics["false_authorizations"] == 2
    assert metrics["semantic_false_authorization_risk"] == 0.5
    assert metrics["supported_coverage"] == 4 / 6
