import json

from mousebrainbench.benchmarks.claim_adversarial import run


def test_claim_adversarial_benchmark_exposes_overclaiming(tmp_path) -> None:
    output = run(output=tmp_path / "summary.json", markdown=tmp_path / "summary.md")
    payload = json.loads(output.read_text())

    aggregate = {row["evaluator"]: row for row in payload["aggregate_by_evaluator"]}
    assert payload["decision"] == "claim_gate_blocks_broad_adversarial_overclaims"
    assert payload["num_cases"] >= 40
    assert aggregate["claim_gate"]["fp"] == 0
    assert aggregate["correlation_only"]["fp"] > 0
    assert aggregate["compensatory_score"]["fp"] > 0
    assert aggregate["ablated_claim_gate_no_directed"]["fp"] > aggregate["claim_gate"]["fp"]
