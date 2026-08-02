from copy import deepcopy

from mousebrainbench.benchmarks.semantic_risk_v3_release import evaluate


def _payloads() -> dict:
    clean = "clean-revision"
    metric = {
        "authorizations": 10,
        "supported_coverage": 0.2,
        "semantic_false_authorization_risk": 0.01,
        "semantic_support_violations": 0,
    }
    return {
        "frozen_policy": {
            "git_revision": clean,
            "thresholds_selected_before_v3_outcome_access": True,
        },
        "synthetic": {
            "git_revision": clean,
            "computational_primary_passed": True,
            "aggregate_by_policy": [
                {"policy": "semantic_MAPIE_risk_control", **metric}
            ],
        },
        "asp_equivalence": {
            "git_revision": clean,
            "decision": "semantic_equivalence_observed",
            "mismatch_count": 0,
            "evaluated_case_count": 2847,
        },
        "causal_chambers": {
            "git_revision": clean,
            "by_partition": [
                {
                    "partition": "locked_test",
                    **metric,
                    "semantic_false_authorization_risk": 0.06,
                }
            ],
        },
        "causalbench": {
            "git_revision": clean,
            "domains": [
                {
                    "role": "locked_transport_test",
                    **metric,
                    "semantic_false_authorization_risk": 0.25,
                }
            ],
        },
        "ibl": {
            "git_revision": clean,
            "partitions": [
                {
                    "role": "locked_mouse_test",
                    **metric,
                    "authorizations": 0,
                    "supported_coverage": 0.0,
                }
            ],
        },
        "guarantee_scope": {
            "git_revision": clean,
            "decision": "out_of_scope_certificates_blocked",
            "cases": [
                {
                    "case": "fresh_synthetic_v3",
                    "assessment": {"valid": True},
                    "scope_enforced": {"authorizations": 10},
                },
                *[
                    {
                        "case": name,
                        "assessment": {"valid": False},
                        "scope_enforced": {"authorizations": 0},
                    }
                    for name in (
                        "causal_chambers_locked",
                        "causalbench_rpe1_locked",
                        "ibl_locked_mice",
                    )
                ],
            ],
        },
        "sensitivity": {
            "git_revision": clean,
            "analysis_role": "post_confirmation_exploratory_sensitivity",
            "confirmatory_reuse_prohibited": True,
            "semantic_support_violations": 0,
        },
        "router_repair": {
            "git_revision": clean,
            "analysis_role": "post_confirmation_exploratory_repair",
            "frozen_primary_router_unchanged": True,
            "decision": "precondition_removes_archived_spurious_attempts",
        },
    }


def test_release_distinguishes_core_validation_from_external_generalization() -> None:
    result = evaluate(_payloads())

    assert result["integrity_ready"] is True
    assert result["methodological_core_validated"] is True
    assert result["external_raw_risk_failure_detected"] is True
    assert result["external_generalization_established"] is False
    assert result["decision"] == (
        "methodological_core_validated_external_generalization_not_established"
    )


def test_dirty_artifact_blocks_methodological_release() -> None:
    payloads = deepcopy(_payloads())
    payloads["synthetic"]["git_revision"] = "revision-dirty"

    result = evaluate(payloads)

    assert result["integrity_ready"] is False
    assert result["decision"] == "semantic_risk_v3_release_requires_action"
    assert result["dirty_artifacts"] == ["synthetic"]


def test_failed_primary_condition_blocks_methodological_release() -> None:
    payloads = deepcopy(_payloads())
    payloads["synthetic"]["computational_primary_passed"] = False

    result = evaluate(payloads)

    assert result["conditions"]["synthetic_primary_passed"] is False
    assert result["integrity_ready"] is False
