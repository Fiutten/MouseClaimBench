import json

from mousebrainbench.benchmarks.scifact_claim_verification import run as run_scifact
from mousebrainbench.benchmarks.tuebingen_causal_direction import run as run_tuebingen


def test_scifact_adapter_detects_lexical_shortcut_overclaiming(tmp_path) -> None:
    root = tmp_path / "scifact"
    root.mkdir()
    (root / "corpus.jsonl").write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "doc_id": 1,
                        "title": "Alpha beta therapy",
                        "abstract": ["Alpha beta therapy improves marker response."],
                    }
                ),
                json.dumps(
                    {
                        "doc_id": 2,
                        "title": "Gamma delta marker",
                        "abstract": ["Gamma and delta are observed without support."],
                    }
                ),
            ]
        )
    )
    (root / "claims_dev.jsonl").write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "id": 1,
                        "claim": "Alpha beta therapy improves marker response.",
                        "evidence": {"1": [{"label": "SUPPORT"}]},
                        "cited_doc_ids": [1],
                    }
                ),
                json.dumps(
                    {
                        "id": 2,
                        "claim": "Gamma delta marker is supported.",
                        "evidence": {},
                        "cited_doc_ids": [2],
                    }
                ),
            ]
        )
    )

    output = run_scifact(
        root=root,
        output=tmp_path / "results/scifact_claim_verification/summary.json",
        markdown=tmp_path / "results/scifact_claim_verification/summary.md",
    )
    payload = json.loads(output.read_text())

    assert payload["shortcut_false_positives"] >= 1
    assert payload["shortcut_overclaiming_risk"] > 0
    assert payload["retrieval_recall_at_5"] > 0
    assert "retrieval_overclaiming_risk" in payload
    assert payload["rows"][0]["bm25_topk_doc_ids"]
    assert payload["rows"][0]["retrieval_label"] in {"SUPPORT", "NOT_ENOUGH_INFO"}


def test_tuebingen_adapter_runs_on_local_pair_subset(tmp_path) -> None:
    root = tmp_path / "tuebingen"
    root.mkdir()
    (root / "pairmeta.txt").write_text("0001 1 1 2 2 1.0\n")
    (root / "pair0001.txt").write_text("\n".join(f"{x} {x * x}" for x in range(1, 50)))

    output = run_tuebingen(
        root=root,
        output=tmp_path / "results/tuebingen_causal_direction/summary.json",
        markdown=tmp_path / "results/tuebingen_causal_direction/summary.md",
        max_pairs=1,
    )
    payload = json.loads(output.read_text())

    assert payload["num_pairs_loaded"] == 1
    assert payload["decision"] == "tuebingen_external_direction_benchmark_insufficient"
    assert set(payload["method_summary"]) == {"anm", "igci", "lingam_proxy"}
    assert payload["consensus_curve"]
    assert payload["causal_performance_claim_allowed"] is False
