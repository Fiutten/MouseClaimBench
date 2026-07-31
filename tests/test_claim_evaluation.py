from mousebrainbench.validation.claim_evaluation import (
    AblatedClaimGateEvaluator,
    ClaimEvidence,
    ClaimGateEvaluator,
    CompensatoryScoreEvaluator,
    CorrelationOnlyEvaluator,
    evidence_to_claim_contract,
    overclaiming_risk_index,
)


def test_claim_gate_blocks_prediction_only_mechanistic_claim() -> None:
    evidence = ClaimEvidence(predictive_score=0.95, reproducibility_score=0.95)

    naive = CorrelationOnlyEvaluator().evaluate(evidence)
    gate = ClaimGateEvaluator().evaluate(evidence)

    assert "mechanistic" in naive.allowed_claims
    assert "mechanistic" not in gate.allowed_claims
    assert set(gate.allowed_claims) == {"predictive", "reproducible"}


def test_compensatory_score_can_overauthorize_when_blocks_are_missing() -> None:
    evidence = ClaimEvidence(
        predictive_score=0.95,
        reproducibility_score=0.95,
        topology_effect=0.20,
        topology_specific=True,
        directed_fraction=0.0,
        matched_structure_function_effect=0.03,
        structure_function_effect=0.04,
        structure_function_fdr_passed=True,
    )

    compensatory = CompensatoryScoreEvaluator(threshold=0.60).evaluate(evidence)
    gate = ClaimGateEvaluator().evaluate(evidence)

    assert "mechanistic" in compensatory.allowed_claims
    assert "mechanistic" not in gate.allowed_claims
    assert "directed" not in gate.allowed_claims


def test_evidence_contract_exposes_authorized_claims() -> None:
    evidence = ClaimEvidence(
        predictive_score=0.90,
        reproducibility_score=0.92,
        topology_effect=0.10,
        topology_specific=True,
        directed_fraction=0.85,
    )

    contract = evidence_to_claim_contract(evidence)

    assert contract["mechanistic"]["authorized"] is True
    assert contract["causal"]["authorized"] is False
    assert contract["digital_twin"]["authorized"] is False


def test_ablation_reveals_direction_block_value() -> None:
    evidence = ClaimEvidence(
        predictive_score=0.90,
        reproducibility_score=0.92,
        topology_effect=0.10,
        topology_specific=True,
        directed_fraction=0.0,
    )

    gate = ClaimGateEvaluator().evaluate(evidence)
    ablated = AblatedClaimGateEvaluator("directed").evaluate(evidence)

    assert "mechanistic" not in gate.allowed_claims
    assert "mechanistic" in ablated.allowed_claims


def test_overclaiming_risk_index_is_false_positive_rate_over_unsupported_claims() -> None:
    assert overclaiming_risk_index(fp=2, tn=8) == 0.2
