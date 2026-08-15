"""Apply the hardened v2 profile retrospectively to four real artifact cases."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from mousebrainbench import __version__
from mousebrainbench.artifacts import code_revision
from mousebrainbench.knowledge import (
    ClaimAuthorizationSystem,
    authorize_with_clingo_v2,
    load_authorization_profile_v2,
)
from mousebrainbench.validation.evidence_contract import EvidenceBlock, EvidenceStatus

DEFAULT_PROTOCOL = Path("configs/benchmarks/profile_v2_artifact_application.yaml")
DEFAULT_OUTPUT = Path("results/profile_v2_artifact_application/summary.json")
DEFAULT_MARKDOWN = Path("results/profile_v2_artifact_application/summary.md")
STRICT_TWIN = "complete_entity_specific_mouse_brain_digital_twin"


@dataclass(frozen=True)
class ArtifactCase:
    name: str
    target_claim: str
    blocks: dict[str, EvidenceBlock]
    sources: tuple[str, ...]
    interpretation: str


def _load(path: str) -> dict[str, Any]:
    source = Path(path)
    if not source.exists():
        raise FileNotFoundError(f"required v2 artifact source is missing: {path}")
    return json.loads(source.read_text())


def _fact(
    name: str,
    status: EvidenceStatus,
    source: str,
    rationale: str,
    observations: dict[str, Any] | None = None,
) -> EvidenceBlock:
    return EvidenceBlock.from_mapping(
        name=name,
        status=status,
        source=source,
        rule="retrospective v2 mapping of the unchanged source predicate",
        rationale=rationale,
        observations=observations or {},
    )


def _allen_case(config: dict[str, Any]) -> ArtifactCase:
    source = str(config["source"])
    payload = _load(source)
    source_blocks = {
        row.get("name"): row for row in payload.get("mis", {}).get("blocks", [])
    }
    topology = source_blocks.get("topology_specificity", {})
    direction = source_blocks.get("directed_identifiability", {})
    blocks = {
        "prediction": _fact(
            "prediction",
            EvidenceStatus.UNKNOWN,
            source,
            "the artifact evaluates target stability rather than a held-out prediction contract",
        ),
        "internal_reproduction": _fact(
            "internal_reproduction",
            EvidenceStatus.REQUIRES_REVIEW,
            source,
            "cross-mouse and split-half stability pass, but overlap and unit identity are not fully encoded",
        ),
        "topology_specificity": _fact(
            "topology_specificity",
            EvidenceStatus.PASSED
            if topology.get("passed") is True
            else EvidenceStatus.FAILED,
            source,
            "the stored topology block does not outperform its controls",
            {"criteria": topology.get("criteria"), "score": topology.get("score")},
        ),
        "directed_identifiability": _fact(
            "directed_identifiability",
            EvidenceStatus.PASSED
            if direction.get("passed") is True
            else EvidenceStatus.FAILED,
            source,
            "the stored latency and lead-lag tests do not identify direction",
            {"criteria": direction.get("criteria"), "score": direction.get("score")},
        ),
        "uncertainty_quantification": _fact(
            "uncertainty_quantification",
            EvidenceStatus.UNKNOWN,
            source,
            "no complete claim-level uncertainty contract is stored",
        ),
        "distribution_shift": _fact(
            "distribution_shift",
            EvidenceStatus.NOT_APPLICABLE,
            source,
            "the Allen analysis did not target a transport population",
        ),
        "competing_mechanisms": _fact(
            "competing_mechanisms",
            EvidenceStatus.UNKNOWN,
            source,
            "the artifact does not evaluate a declared competing-mechanism set",
        ),
        "robustness": _fact(
            "robustness",
            EvidenceStatus.UNKNOWN,
            source,
            "no locked perturbation-family endpoint is stored",
        ),
        "context_of_use": _fact(
            "context_of_use",
            EvidenceStatus.PASSED,
            source,
            "the analysis is explicitly bounded to a visual-system identifiability audit",
            {
                "use": "audit a visual-system target",
                "population": "Allen Visual Behavior Neuropixels sessions",
                "output": "stability, topology-control, and direction diagnostics",
                "decision_consequence": "permit only bounded artifact claims",
                "prohibited_uses": ["causal mechanism", "whole-brain digital twin"],
            },
        ),
        "data_quality": _fact(
            "data_quality",
            EvidenceStatus.UNKNOWN,
            source,
            "the legacy summary does not encode the complete v2 data-quality contract",
        ),
    }
    return ArtifactCase(
        name="allen_vbn_negative",
        target_claim=str(config["target_claim"]),
        blocks=blocks,
        sources=(source,),
        interpretation=(
            "The narrowed directed-topology claim remains unauthorized because prediction, "
            "topology, direction, uncertainty, shift, alternatives, robustness, and data quality "
            "are failed, absent, untargeted, or require review."
        ),
    )


def _sensorium_static_case(config: dict[str, Any]) -> ArtifactCase:
    source = str(config["source"])
    payload = _load(source)
    repeated = payload.get("pretraining_test_repeated", {})
    rows = [row for row in payload.get("rows", []) if row.get("eval_tier") == "test"]
    predictive = (
        int(repeated.get("n", 0)) == 5
        and float(repeated.get("median_reliability", 0.0)) > 0.0
        and float(repeated.get("median_best_minus_mean", 0.0)) > 0.0
        and float(repeated.get("median_best_minus_scrambled", 0.0)) > 0.0
    )
    quality = (
        len(rows) == 5
        and all(int(row.get("n_trials", 0)) > 0 for row in rows)
        and all(int(row.get("n_neurons", 0)) > 0 for row in rows)
        and all(float(row.get("reliability", 0.0)) > 0.0 for row in rows)
    )
    blocks = {
        "prediction": _fact(
            "prediction",
            EvidenceStatus.PASSED if predictive else EvidenceStatus.FAILED,
            source,
            "the repeated-test cohort beats mean-response and scrambled baselines",
            {
                "target": "single-neuron visual response",
                "target_population": "five Sensorium 2022 repeated-test mouse datasets",
                "split": "provider test tier",
                "split_integrity": "test-tier artifacts are distinct from fitting summaries",
                "metric": "median predictive Pearson correlation",
                "threshold": "positive median gain over mean and scrambled baselines",
                "comparator": ["mean response", "scrambled stimulus"],
                "value": repeated.get("median_best_predictive_correlation"),
            },
        ),
        "context_of_use": _fact(
            "context_of_use",
            EvidenceStatus.PASSED,
            source,
            "the claim is restricted to static visual-response prediction",
            {
                "use": "compare stored predictors on static visual responses",
                "population": "five Sensorium 2022 repeated-test mice",
                "output": "median per-neuron predictive correlation",
                "decision_consequence": "authorize bounded predictive performance only",
                "prohibited_uses": ["SOTA", "mechanism", "causality", "digital twin"],
            },
        ),
        "data_quality": _fact(
            "data_quality",
            EvidenceStatus.PASSED if quality else EvidenceStatus.REQUIRES_REVIEW,
            source,
            "all five declared test rows contain trials, neurons, and estimable reliability",
            {
                "source": "Sensorium 2022 provider datasets",
                "lineage": [payload.get("input_summary"), source],
                "exclusions": "validation-only datasets are excluded from this bounded claim",
                "missingness": "five of five declared repeated-test rows are present",
                "quality_checks": "positive trial, neuron, and reliability counts in every row",
                "result": quality,
            },
        ),
    }
    return ArtifactCase(
        name="sensorium_static_bounded_prediction",
        target_claim=str(config["target_claim"]),
        blocks=blocks,
        sources=(source,),
        interpretation=(
            "The bounded predictive claim may be profile-authorized. It does not establish "
            "SOTA performance, topology, direction, mechanism, or external replication."
        ),
    )


def _dynamic_sensorium_case(config: dict[str, Any]) -> ArtifactCase:
    source = str(config["source"])
    payload = _load(source)
    cohort_rows = []
    for cohort in payload.get("cohorts", []):
        comparison = cohort.get("pairwise", {}).get(
            "mean_response_vs_temporal_svd", {}
        )
        cohort_rows.append(
            {
                "cohort": cohort.get("cohort"),
                "n_mice": cohort.get("n_mice"),
                "wins": comparison.get("right_wins"),
                "median_delta": comparison.get("median_delta"),
                "reliability_estimable_count": cohort.get(
                    "reliability_estimable_count", 0
                ),
            }
        )
    predictive = (
        len(cohort_rows) == 2
        and all(int(row.get("n_mice") or 0) == 5 for row in cohort_rows)
        and all(int(row.get("wins") or 0) >= 4 for row in cohort_rows)
        and all(float(row.get("median_delta") or 0.0) > 0.0 for row in cohort_rows)
    )
    blocks = {
        "prediction": _fact(
            "prediction",
            EvidenceStatus.PASSED if predictive else EvidenceStatus.FAILED,
            source,
            "temporal SVD beats mean response in the declared paired comparison",
            {
                "target": "dynamic visual response",
                "target_population": "two stored five-mouse Dynamic Sensorium cohorts",
                "split": "two non-overlapping provider cohorts",
                "split_integrity": "cohort identifiers are distinct",
                "metric": "paired per-mouse predictive-correlation difference",
                "threshold": "at least four wins and positive median difference per cohort",
                "comparator": "mean response",
                "value": cohort_rows,
            },
        ),
        "context_of_use": _fact(
            "context_of_use",
            EvidenceStatus.PASSED,
            source,
            "the claim is restricted to one dynamic predictive comparison",
            {
                "use": "compare temporal SVD with a mean-response baseline",
                "population": "two stored Dynamic Sensorium cohorts",
                "output": "paired predictive-correlation difference",
                "decision_consequence": "audit bounded temporal prediction",
                "prohibited_uses": ["reliability claim", "mechanism", "causality"],
            },
        ),
        "data_quality": _fact(
            "data_quality",
            EvidenceStatus.REQUIRES_REVIEW,
            source,
            "response reliability is not estimable in the stored cohorts",
            {"cohorts": cohort_rows},
        ),
    }
    return ArtifactCase(
        name="dynamic_sensorium_prediction_with_quality_deficit",
        target_claim=str(config["target_claim"]),
        blocks=blocks,
        sources=(source,),
        interpretation=(
            "Prediction passes its stored comparison, but the v2 bounded claim is not authorized "
            "because response-reliability quality remains unresolved."
        ),
    )


def _microns_case(config: dict[str, Any]) -> ArtifactCase:
    robustness_source = str(config["robustness_source"])
    package_source = str(config["package_source"])
    network_source = str(config["network_source"])
    robustness = _load(robustness_source)
    package = _load(package_source)
    network = _load(network_source)
    cohorts = package.get("cohorts", [])
    robust = (
        robustness.get("all_cohorts_robust") is True
        and package.get("q1_package_ready") is True
        and len(cohorts) == 3
    )
    intervals = [
        {
            "cohort": row.get("cohort"),
            "distance": row.get("unit_bootstrap", {}).get("distance_matched_delta"),
            "degree": row.get("unit_bootstrap", {}).get("degree_matched_delta"),
        }
        for row in cohorts
    ]
    network_cohorts = network.get("cohorts", [])
    confirmation_status = network.get("confirmation_passed")
    if confirmation_status is None:
        # Historical v0.12.1 artifacts predate the explicit discovery/hold-out
        # split. Retain read compatibility without reusing this legacy field in
        # newly generated results.
        confirmation_status = network.get("all_cohorts_passed")
    network_passed = bool(confirmation_status) and len(network_cohorts) == 3
    network_results = [
        {
            "cohort": row.get("cohort"),
            "coefficient": row.get("connected_coefficient"),
            "dyadic_standard_error": row.get("dyadic_cluster_standard_error"),
            "dyadic_p_value": row.get("dyadic_cluster_two_sided_p_value"),
            "node_permutation_p_value": row.get(
                "freedman_lane_node_permutation", {}
            ).get("one_sided_p_value"),
        }
        for row in network_cohorts
    ]
    blocks = {
        "structure_function_association": _fact(
            "structure_function_association",
            EvidenceStatus.PASSED if robust else EvidenceStatus.FAILED,
            package_source,
            "the fixed endpoint is positive under distance and degree matching in all windows",
            {
                "tissue": "one MICRONS visual-cortex volume",
                "functional_descriptor": "readout-location similarity",
                "connected_pairs": [row.get("n_connected_edge_pairs") for row in cohorts],
                "matched_controls": ["distance", "degree"],
                "matching_variables": ["soma distance", "pre/post degree"],
                "estimate": [row.get("primary_test") for row in cohorts],
            },
        ),
        "network_dependence_control": _fact(
            "network_dependence_control",
            EvidenceStatus.PASSED if network_passed else EvidenceStatus.FAILED,
            network_source,
            "directed dyadic covariance and simultaneous node-label permutation pass in all windows",
            {
                "inferential_unit": "directed neuron pair",
                "dependence_structure": "directed pairs share pre- and postsynaptic units",
                "method": [
                    "directed dyadic cluster-robust covariance",
                    "Freedman-Lane simultaneous node-label permutation",
                ],
                "estimand": "partial connected-pair coefficient for readout-location similarity",
                "result": network_results,
            },
        ),
        "uncertainty_quantification": _fact(
            "uncertainty_quantification",
            EvidenceStatus.PASSED if robust else EvidenceStatus.FAILED,
            package_source,
            "all unit-cluster stability intervals have positive lower bounds",
            {
                "inferential_unit": "unit-cluster weighted directed pair frame",
                "uncertainty_sources": ["unit reuse", "matching resampling"],
                "method": "300-sample unit-cluster weighted bootstrap",
                "confidence_level": 0.95,
                "interval": intervals,
                "acceptance_rule": "positive distance- and degree-matched lower bounds",
            },
        ),
        "robustness": _fact(
            "robustness",
            EvidenceStatus.PASSED if robust else EvidenceStatus.FAILED,
            robustness_source,
            "the endpoint survives combined distance-degree controls and within-distance shuffling",
            {
                "perturbation_family": [
                    "combined distance-degree matching",
                    "within-distance readout shuffle",
                ],
                "locked_endpoint": package.get("primary_endpoint"),
                "acceptance_rule": "positive in discovery and both non-overlapping hold-outs",
                "worst_case_result": min(
                    row.get("combined_distance_degree_control", {}).get("delta", 0.0)
                    for row in robustness.get("cohorts", [])
                ),
            },
        ),
        "context_of_use": _fact(
            "context_of_use",
            EvidenceStatus.PASSED,
            package_source,
            "the endpoint is explicitly local and observational",
            {
                "use": "audit one local structure-function association",
                "population": "one MICRONS visual-cortex volume",
                "output": "matched readout-location similarity difference",
                "decision_consequence": "permit only local observational wording",
                "prohibited_uses": ["causality", "mechanism", "external replication", "whole brain"],
            },
        ),
        "data_quality": _fact(
            "data_quality",
            EvidenceStatus.PASSED if robust else EvidenceStatus.REQUIRES_REVIEW,
            package_source,
            "all declared unit and edge windows are present and contain eligible pairs",
            {
                "source": "MICRONS CAVE-derived local exports",
                "lineage": [robustness_source, package_source, network_source],
                "exclusions": "eligibility and window rules recorded by source scripts",
                "missingness": "three of three declared windows are present",
                "quality_checks": "positive unit, synapse, and connected-pair counts",
                "result": robust,
            },
        ),
    }
    return ArtifactCase(
        name="microns_local_association_with_dyadic_control",
        target_claim=str(config["target_claim"]),
        blocks=blocks,
        sources=(robustness_source, package_source, network_source),
        interpretation=(
            "The local endpoint passes matched controls and unit-cluster stability. Discovery "
            "fixes the positive direction, and both hold-outs pass directed dyadic uncertainty "
            "and simultaneous node-label permutation. This authorizes only a local "
            "observational association in one tissue volume."
        ),
    )


def _ibl_case(config: dict[str, Any]) -> ArtifactCase:
    source = str(config["source"])
    source_protocol = str(config["source_protocol"])
    payload = _load(source)
    protocol = yaml.safe_load(Path(source_protocol).read_text())
    risk = payload.get("risk_lock", {})
    final = payload.get("final_evaluation", {})
    policy_name = "frozen_v5_1_complete_authorizer"
    risk_certificate = risk.get("comparators", {}).get(policy_name, {})
    final_certificate = final.get("comparators", {}).get(policy_name, {})
    prediction_rule = protocol["behavioral_task"]["prediction_pass_requires"]
    topology_rule = protocol["behavioral_task"]["topology_specificity_requires"]
    risk_alignment = risk.get("actual_alignment", {})
    final_alignment = final.get("actual_alignment", {})
    prediction_passed = all(
        float(row.get("median_tjur_r_squared", 0.0))
        >= float(prediction_rule["held_out_tjur_r2_minimum"])
        for row in (risk_alignment, final_alignment)
    )
    topology_passed = all(
        float(row.get("median_topology_margin", 0.0))
        >= float(topology_rule["held_out_tjur_r2_margin_over_best_other_candidate"])
        for row in (risk_alignment, final_alignment)
    )
    certificates_passed = (
        risk_certificate.get("certified") is True
        and final_certificate.get("certified") is True
        and int(risk_certificate.get("experiments", 0)) == 35
        and int(final_certificate.get("experiments", 0)) == 35
    )
    blocks = {
        "prediction": _fact(
            "prediction",
            EvidenceStatus.PASSED if prediction_passed else EvidenceStatus.FAILED,
            source,
            "the true trial alignment passes the frozen held-out prediction rule in both mouse splits",
            {
                "target": protocol["behavioral_task"]["endpoint"],
                "target_population": "35 risk-lock and 35 final IBL mice",
                "split": "mouse-disjoint risk-lock and final roles",
                "split_integrity": "mouse identifiers are assigned once by the frozen selection",
                "metric": "held-out Tjur R-squared with correlation and p-value guards",
                "threshold": prediction_rule,
                "comparator": "block-only behavioral baseline",
                "value": {
                    "risk_lock": risk_alignment,
                    "final": final_alignment,
                },
            },
        ),
        "topology_specificity": _fact(
            "topology_specificity",
            EvidenceStatus.PASSED if topology_passed else EvidenceStatus.FAILED,
            source,
            "the true trial alignment exceeds every frozen circular-offset control",
            {
                "topology_scale": "trial-alignment relation within each mouse",
                "candidate": "offset 0",
                "control_family": protocol["behavioral_task"]["candidate_offsets"][1:],
                "complexity_matching": "identical estimator and folds for every offset",
                "metric": "held-out Tjur R-squared margin",
                "margin": {
                    "risk_lock": risk_alignment.get("median_topology_margin"),
                    "final": final_alignment.get("median_topology_margin"),
                },
            },
        ),
        "uncertainty_quantification": _fact(
            "uncertainty_quantification",
            EvidenceStatus.PASSED if certificates_passed else EvidenceStatus.FAILED,
            source,
            "mouse-level Clopper-Pearson risk and utility bounds pass in both locked splits",
            {
                "inferential_unit": "mouse",
                "uncertainty_sources": ["finite mouse population", "nested trial decisions"],
                "method": "one-sided exact Clopper-Pearson bounds",
                "confidence_level": risk_certificate.get("confidence"),
                "interval": {
                    "risk_lock_risk_ucb": risk_certificate.get("risk_upper_bound"),
                    "final_risk_ucb": final_certificate.get("risk_upper_bound"),
                    "risk_lock_coverage_lcb": risk_certificate.get("coverage_lower_bound"),
                    "final_coverage_lcb": final_certificate.get("coverage_lower_bound"),
                },
                "acceptance_rule": protocol["inferential_contract"]["final_claim_requires"],
            },
        ),
        "robustness": _fact(
            "robustness",
            EvidenceStatus.PASSED if certificates_passed else EvidenceStatus.FAILED,
            source,
            "the unchanged contract passes in two non-overlapping 35-mouse roles",
            {
                "perturbation_family": "risk-lock versus untouched final mouse split",
                "locked_endpoint": protocol["behavioral_task"]["endpoint"],
                "acceptance_rule": "the complete mouse-level contract passes in both roles",
                "worst_case_result": max(
                    float(risk_certificate.get("risk_upper_bound", 1.0)),
                    float(final_certificate.get("risk_upper_bound", 1.0)),
                ),
            },
        ),
        "context_of_use": _fact(
            "context_of_use",
            EvidenceStatus.PASSED,
            source,
            "the source claim is explicitly bounded to one IBL behavioral alignment task",
            {
                "use": "audit visual-evidence trial-alignment specificity",
                "population": "IBL Brain-Wide Map mice under one standardized task",
                "output": "mouse-level topology-specific authorization",
                "decision_consequence": "permit one bounded behavioral alignment claim",
                "prohibited_uses": [
                    "neural mechanism",
                    "independent laboratories",
                    "whole-brain validity",
                    "digital twin",
                ],
            },
        ),
    }
    return ArtifactCase(
        name="ibl_behavior_topology_specific_prediction",
        target_claim=str(config["target_claim"]),
        blocks=blocks,
        sources=(source, source_protocol),
        interpretation=(
            "The bounded topology-specific behavioral prediction is profile-authorized across "
            "two locked mouse splits. Simple source comparators also passed, and all mice share "
            "the IBL task ecosystem, so no exclusive superiority or laboratory replication follows."
        ),
    )


def build_cases(protocol: dict[str, Any]) -> tuple[ArtifactCase, ...]:
    cases = protocol["cases"]
    return (
        _allen_case(cases["allen"]),
        _sensorium_static_case(cases["sensorium_static"]),
        _dynamic_sensorium_case(cases["dynamic_sensorium"]),
        _microns_case(cases["microns"]),
        _ibl_case(cases["ibl"]),
    )


def evaluate(
    protocol: dict[str, Any],
) -> dict[str, Any]:
    """Evaluate target and strict-twin decisions with Python-ASP equivalence."""

    profile = load_authorization_profile_v2()
    rows = []
    asp_matches = 0
    for case in build_cases(protocol):
        for claim in (case.target_claim, STRICT_TWIN):
            python = ClaimAuthorizationSystem(profile, case.blocks).infer(claim)
            asp = authorize_with_clingo_v2(profile, claim, case.blocks)
            match = (
                asp.status is python.status
                and asp.deficits
                == tuple(
                    sorted(
                        (
                            (fact.name, fact.effective_status)
                            for fact in python.deficits
                        ),
                        key=lambda item: item[0],
                    )
                )
            )
            asp_matches += int(match)
            rows.append(
                {
                    "case": case.name,
                    "claim": claim,
                    "status": python.status.value,
                    "authorized": python.authorized,
                    "deficits": [fact.as_dict() for fact in python.deficits],
                    "passed_blocks": [
                        fact.name
                        for fact in python.facts
                        if fact.effective_status is EvidenceStatus.PASSED
                    ],
                    "sources": list(case.sources),
                    "interpretation": case.interpretation,
                    "python_asp_equivalent": match,
                }
            )
    target_rows = [row for row in rows if row["claim"] != STRICT_TWIN]
    twin_rows = [row for row in rows if row["claim"] == STRICT_TWIN]
    conditions = {
        "no_complete_twin_authorized": not any(row["authorized"] for row in twin_rows),
        "all_decisions_have_complete_deficit_traces": all(
            row["authorized"] or row["deficits"] for row in rows
        ),
        "python_asp_equivalence_for_all_case_decisions": asp_matches == len(rows),
    }
    return {
        "cases": len(target_rows),
        "decisions": len(rows),
        "target_authorizations": sum(row["authorized"] for row in target_rows),
        "strict_twin_authorizations": sum(row["authorized"] for row in twin_rows),
        "decision_rows": rows,
        "release_conditions": conditions,
        "all_release_conditions_passed": all(conditions.values()),
        "interpretation": protocol["interpretation"],
    }


def _write_markdown(payload: dict[str, Any], path: Path) -> None:
    lines = [
        "# Profile v2 artifact application",
        "",
        f"- Decision: `{payload['decision']}`",
        f"- Cases: `{payload['cases']}`",
        f"- Target authorizations: `{payload['target_authorizations']}`",
        f"- Strict-twin authorizations: `{payload['strict_twin_authorizations']}`",
        "",
        "| Case | Claim | Status | Deficits |",
        "|---|---|---|---:|",
    ]
    for row in payload["decision_rows"]:
        lines.append(
            f"| `{row['case']}` | `{row['claim']}` | `{row['status']}` | "
            f"{len(row['deficits'])} |"
        )
    lines.extend(("", payload["interpretation"], ""))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines))


def run(
    *,
    protocol_path: Path = DEFAULT_PROTOCOL,
    output: Path = DEFAULT_OUTPUT,
    markdown: Path = DEFAULT_MARKDOWN,
) -> Path:
    """Run and persist the retrospective v2 artifact application."""

    protocol = yaml.safe_load(protocol_path.read_text())
    assessment = evaluate(protocol)
    payload = {
        "version": __version__,
        "git_revision": code_revision(),
        "analysis": "profile_v2_artifact_application",
        "protocol": str(protocol_path),
        "protocol_sha256": hashlib.sha256(protocol_path.read_bytes()).hexdigest(),
        **assessment,
        "decision": (
            "profile_v2_artifact_application_complete"
            if assessment["all_release_conditions_passed"]
            else "profile_v2_artifact_application_failed"
        ),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n")
    _write_markdown(payload, markdown)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--markdown", type=Path, default=DEFAULT_MARKDOWN)
    args = parser.parse_args()
    print(
        run(
            protocol_path=args.protocol,
            output=args.output,
            markdown=args.markdown,
        ).resolve()
    )


if __name__ == "__main__":
    main()
