from mousebrainbench.benchmarks.digital_twin_claim_audit import audit_claims


def test_prediction_and_matched_nulls_do_not_imply_causal_mechanism() -> None:
    payload = audit_claims(
        {
            "predictive_or_reproducible": True,
            "matched_nulls_passed": True,
            "causal_or_interventional_evidence": False,
        }
    )

    mechanistic = next(
        audit for audit in payload["audits"] if audit["claim"] == "mechanistic_identifiability"
    )

    assert not mechanistic["passed"]
    assert "not causal mechanism" in mechanistic["blocked_wording"]


def test_measured_edges_do_not_make_whole_brain_twin() -> None:
    payload = audit_claims(
        {
            "single_neuron_units": True,
            "measured_synaptic_connectivity": True,
            "whole_brain_coverage": False,
            "independent_whole_brain_validation": True,
            "reproducible_compute_budget": True,
        }
    )

    single_neuron = next(
        audit
        for audit in payload["audits"]
        if audit["claim"] == "single_neuron_resolution_connectivity"
    )
    whole_brain = next(
        audit for audit in payload["audits"] if audit["claim"] == "whole_brain_digital_twin"
    )

    assert single_neuron["passed"]
    assert not whole_brain["passed"]
