import json
from pathlib import Path

import pandas as pd
import yaml

from mousebrainbench.benchmarks.knowledge_profile_internal_ai_audit import (
    evaluate,
    exhaustive_rule_safety,
)


def _inputs():
    protocol = yaml.safe_load(
        Path("configs/validation/knowledge_profile_internal_ai_audit_v1.yaml").read_text()
    )
    items = pd.read_csv(protocol["review_packet"])
    reviews = pd.read_csv(protocol["item_review"])
    return protocol, items, reviews


def test_internal_audit_cannot_masquerade_as_external_panel() -> None:
    protocol, items, reviews = _inputs()
    result = evaluate(protocol, items, reviews, Path("references.bib").read_text())
    assert result["independent_content_validation"] is False
    assert result["human_raters_created"] == 0
    assert result["cvi_computed"] is False
    assert result["external_panel_status"] == "still_required_and_pending"


def test_internal_audit_reviews_all_items_and_requires_revision() -> None:
    protocol, items, reviews = _inputs()
    result = evaluate(protocol, items, reviews, Path("references.bib").read_text())
    assert len(items) == len(reviews) == 29
    assert result["source_traceability"]["passed"] is True
    assert result["decision"] == "internal_ai_audit_revision_required"
    assert result["decision_counts"]["critical_veto"] >= 1


def test_exhaustive_rules_never_support_a_non_all_passed_state() -> None:
    result = exhaustive_rule_safety()
    assert result["analytical_status_space_size"] > 9_000_000
    assert result["executable_cases_evaluated"] > 10_000
    assert result["unsafe_non_all_passed_supports"] == 0
    assert result["structural_proof"][
        "every_nonpassed_status_has_a_higher_priority_any_rule"
    ] is True
    assert result["passed"] is True


def test_frozen_internal_artifact_preserves_external_validation_boundary() -> None:
    payload = json.loads(
        Path("results/knowledge_profile_internal_ai_audit_v1/summary.json").read_text()
    )
    assert payload["decision"] == "internal_ai_audit_revision_required"
    assert payload["independent_content_validation"] is False
    assert payload["human_raters_created"] == 0
    assert not payload["git_revision"].endswith("-dirty")
    assert payload["decision_counts"] == {
        "critical_veto": 1,
        "retain": 9,
        "revise_major": 11,
        "revise_minor": 8,
    }
