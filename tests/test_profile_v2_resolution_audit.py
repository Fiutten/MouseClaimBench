from pathlib import Path

import pandas as pd
import yaml

from mousebrainbench.benchmarks.profile_v2_resolution_audit import evaluate


def test_v2_maps_every_internal_audit_issue_without_claiming_humans() -> None:
    protocol = yaml.safe_load(
        Path("configs/validation/profile_v2_audit_resolution.yaml").read_text()
    )
    reviews = pd.read_csv(protocol["parent_review"])
    result = evaluate(protocol, reviews, Path("references.bib").read_text())

    assert result["parent_nonretained_items"] == 20
    assert result["resolution_items"] == 20
    assert result["unresolved_items"] == []
    assert result["unresolved_source_ids"] == []
    assert result["all_conditions_passed"] is True
    assert result["external_content_validity"] is False
    assert result["human_validation"] is False
