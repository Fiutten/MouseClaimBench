import csv
import json
from pathlib import Path

from mousebrainbench.benchmarks.human_evaluation_protocol import run


def test_human_protocol_builds_only_unlabeled_preparation_artifacts(tmp_path) -> None:
    candidates = tmp_path / "candidates.json"
    candidates.write_text(
        json.dumps(
            {
                "candidates": [
                    {
                        "claim_type": "causal" if index % 2 else "predictive",
                        "text": f"candidate {index}",
                        "source": "deterministic_extractor",
                    }
                    for index in range(20)
                ]
            }
        )
    )
    output = run(
        protocol_path=Path("configs/human_evaluation_protocol.yaml"),
        candidates_path=candidates,
        output=tmp_path / "study.json",
        template=tmp_path / "annotations.csv",
    )
    payload = json.loads(output.read_text())

    assert payload["protocol_valid"] is True
    assert payload["study_status"] == "not_executed"
    assert payload["results_available"] is False
    assert all(item["reference_label"] is None for item in payload["items"])
    assert all(item["human_annotations"] == [] for item in payload["items"])
    with (tmp_path / "annotations.csv").open() as handle:
        rows = list(csv.DictReader(handle))
    assert rows
    assert all(row["reference_label"] == "" for row in rows)
