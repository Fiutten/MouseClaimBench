from pathlib import Path

import pandas as pd
import yaml

from mousebrainbench.benchmarks.knowledge_profile_external_validation_v5 import evaluate
from scripts.prepare_knowledge_profile_external_review import build_items


def test_blank_panel_remains_explicitly_pending() -> None:
    protocol = yaml.safe_load(
        Path("configs/validation/knowledge_profile_external_review_v1.yaml").read_text()
    )
    items = pd.DataFrame([{"item_id": "one"}])
    result = evaluate(protocol, items, pd.DataFrame(), pd.DataFrame())
    assert result["decision"] == "external_content_validation_pending"
    assert result["profile_content_validated"] is False


def test_packet_contains_every_relation_rule_and_coverage_item() -> None:
    profile = yaml.safe_load(
        Path("mousebrainbench/knowledge/profiles/mouse_brain_claims_v1.yaml").read_text()
    )
    basis = yaml.safe_load(
        Path("mousebrainbench/knowledge/profiles/mouse_brain_claims_v1_basis.yaml").read_text()
    )
    items = build_items(profile, basis)
    assert len(items) == len(basis["relations"]) + len(profile["inference_rules"]) + 2
    assert len({item["item_id"] for item in items}) == len(items)
