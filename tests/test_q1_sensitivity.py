from mousebrainbench.benchmarks.q1_sensitivity import run_sensitivity


def test_q1_sensitivity_preserves_current_scientific_decision() -> None:
    payload = run_sensitivity()

    assert payload["allen_vbn"]["stable_negative"]
    assert payload["sensorium_static"]["stable_partial_positive"] > 0
    assert payload["dynamic_sensorium"]["cohorts"]
    assert payload["decision"] == (
        "sensitivity_supports_methodological_claim_not_q1_mechanistic_claim"
    )
