"""Build an artifact-grounded claim matrix for mouse-brain evidence cases.

Unlike the legacy implementation, this module never invents normalized values
for heterogeneous measurements.  Each case preserves the source observations
and the domain-specific rule used to assign an evidence-block status.  The
result is a bounded case study of the executable claim contract, not an
independent validation of scientific truth.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from mousebrainbench import __version__
from mousebrainbench.artifacts import code_revision
from mousebrainbench.validation.evidence_contract import (
    DecisionStatus,
    EvidenceBlock,
    EvidenceContractEvaluator,
    EvidenceStatus,
    blocks_by_name,
)


DEFAULT_OUTPUT = Path("results/real_case_claim_matrix/summary.json")
DEFAULT_MARKDOWN = Path("results/real_case_claim_matrix/summary.md")

ALL_BLOCKS = (
    "prediction",
    "reproducible_compute",
    "internal_reproduction",
    "external_replication",
    "topology_specificity",
    "directed_identifiability",
    "structure_function_association",
    "causal_intervention",
    "whole_brain_coverage",
    "independent_validation",
)


@dataclass(frozen=True)
class RealEvidenceCase:
    """One bounded real-data case and its source-grounded evidence blocks."""

    name: str
    domain: str
    sources: tuple[str, ...]
    blocks: tuple[EvidenceBlock, ...]
    interpretation: str


def _load_required_json(root: Path, relative: str) -> dict[str, Any]:
    path = root / relative
    if not path.exists():
        raise FileNotFoundError(f"required real-case artifact is missing: {relative}")
    return json.loads(path.read_text())


def _block(
    name: str,
    status: EvidenceStatus,
    source: str,
    rule: str,
    rationale: str,
    **observations: Any,
) -> EvidenceBlock:
    return EvidenceBlock.from_mapping(
        name=name,
        status=status,
        source=source,
        rule=rule,
        rationale=rationale,
        observations=observations,
    )


def _not_applicable(name: str, source: str, rationale: str) -> EvidenceBlock:
    return _block(
        name,
        EvidenceStatus.NOT_APPLICABLE,
        source,
        "the source protocol did not target this evidence block",
        rationale,
    )


def _unknown(name: str, source: str, rationale: str) -> EvidenceBlock:
    return _block(
        name,
        EvidenceStatus.UNKNOWN,
        source,
        "no decision rule can be executed from the stored observations",
        rationale,
    )


def _compute_block(payload: Mapping[str, Any], source: str) -> EvidenceBlock:
    revision = payload.get("git_revision")
    if not isinstance(revision, str):
        return _unknown(
            "reproducible_compute",
            source,
            "the source summary does not record a Git revision",
        )
    clean = not revision.endswith("-dirty")
    return _block(
        "reproducible_compute",
        EvidenceStatus.PASSED if clean else EvidenceStatus.FAILED,
        source,
        "a recorded Git revision must exist and must not end in '-dirty'",
        "the artifact records a clean revision" if clean else "the artifact records a dirty revision",
        git_revision=revision,
    )


def _allen_case(payload: Mapping[str, Any], source: str) -> RealEvidenceCase:
    mis = payload.get("mis", {})
    source_blocks = {block.get("name"): block for block in mis.get("blocks", [])}
    reproducibility = source_blocks.get("reproducibility", {})
    topology = source_blocks.get("topology_specificity", {})
    direction = source_blocks.get("directed_identifiability", {})

    blocks = (
        _unknown(
            "prediction",
            source,
            "the Allen MIS artifact evaluates target stability, not a held-out predictive endpoint",
        ),
        _compute_block(payload, source),
        _block(
            "internal_reproduction",
            EvidenceStatus.PASSED if reproducibility.get("passed") is True else EvidenceStatus.FAILED,
            source,
            "the stored Allen reproducibility block must pass all declared criteria",
            "cross-mouse and split-half target stability passed"
            if reproducibility.get("passed") is True
            else "the reproducibility block did not pass",
            block_score=reproducibility.get("score"),
            criteria=reproducibility.get("criteria"),
        ),
        _unknown(
            "external_replication",
            source,
            "no independent laboratory or resource is represented in this artifact",
        ),
        _block(
            "topology_specificity",
            EvidenceStatus.PASSED if topology.get("passed") is True else EvidenceStatus.FAILED,
            source,
            "the stored topology-specificity block must pass all declared controls",
            "Allen topology passed its controls"
            if topology.get("passed") is True
            else "Allen topology did not outperform the declared controls",
            block_score=topology.get("score"),
            criteria=topology.get("criteria"),
        ),
        _block(
            "directed_identifiability",
            EvidenceStatus.PASSED if direction.get("passed") is True else EvidenceStatus.FAILED,
            source,
            "the stored directed-identifiability block must pass all declared criteria",
            "direction was identified"
            if direction.get("passed") is True
            else "latency and lead-lag criteria did not identify direction",
            block_score=direction.get("score"),
            criteria=direction.get("criteria"),
        ),
        _not_applicable(
            "structure_function_association",
            source,
            "the Allen analysis is not the local MICRONS structure-function protocol",
        ),
        _not_applicable(
            "causal_intervention",
            source,
            "the source contains no intervention that identifies a causal model mechanism",
        ),
        _block(
            "whole_brain_coverage",
            EvidenceStatus.FAILED,
            source,
            "whole-brain coverage requires a whole-brain model and validation target",
            "the evaluated Allen case covers a bounded visual-system target",
        ),
        _block(
            "independent_validation",
            EvidenceStatus.FAILED,
            source,
            "independent validation requires a separate validating resource or study",
            "the stored artifact is an internal analysis of one resource",
        ),
    )
    return RealEvidenceCase(
        name="allen_vbn_identifiability_negative",
        domain="Allen Visual Behavior Neuropixels",
        sources=(source,),
        blocks=blocks,
        interpretation=(
            "The target is internally reproduced, while topology specificity and directed "
            "identifiability fail. This is a real negative mechanistic-identifiability case."
        ),
    )


def _sensorium_static_case(payload: Mapping[str, Any], source: str) -> RealEvidenceCase:
    repeated = payload.get("pretraining_test_repeated", {})
    topology = payload.get("topographic_constraint", {})
    predictive_passed = (
        float(repeated.get("median_best_minus_mean", 0.0)) > 0.0
        and float(repeated.get("median_best_minus_scrambled", 0.0)) > 0.0
    )
    topology_passed = (
        topology.get("decision") == "structural_constraint_supported"
        and int(topology.get("passed_count", 0)) == int(topology.get("n_datasets", -1))
        and int(topology.get("n_datasets", 0)) > 1
    )

    blocks = (
        _block(
            "prediction",
            EvidenceStatus.PASSED if predictive_passed else EvidenceStatus.FAILED,
            source,
            "median best-minus-mean and best-minus-scrambled correlations must both be positive",
            "the repeated-test cohort beats both transparent baselines"
            if predictive_passed
            else "the repeated-test cohort does not beat both baselines",
            median_best_predictive_correlation=repeated.get("median_best_predictive_correlation"),
            median_best_minus_mean=repeated.get("median_best_minus_mean"),
            median_best_minus_scrambled=repeated.get("median_best_minus_scrambled"),
        ),
        _compute_block(payload, source),
        _block(
            "internal_reproduction",
            EvidenceStatus.PASSED if topology_passed else EvidenceStatus.FAILED,
            source,
            "the topographic constraint must pass in every stored dataset and in at least two datasets",
            "the constraint passes in all five datasets"
            if topology_passed
            else "the constraint is not reproduced across all stored datasets",
            passed_count=topology.get("passed_count"),
            n_datasets=topology.get("n_datasets"),
        ),
        _unknown(
            "external_replication",
            source,
            "the mice are internal datasets from the Sensorium resource rather than an external study",
        ),
        _block(
            "topology_specificity",
            EvidenceStatus.PASSED if topology_passed else EvidenceStatus.FAILED,
            source,
            "the stored topographic control must report structural_constraint_supported in all datasets",
            "the topographic effect exceeds its null in all five datasets"
            if topology_passed
            else "the stored topographic control did not pass",
            median_observed_spearman=topology.get("median_observed_spearman"),
            median_effect_over_null=topology.get("median_effect_over_null"),
            passed_count=topology.get("passed_count"),
            n_datasets=topology.get("n_datasets"),
        ),
        _not_applicable(
            "directed_identifiability",
            source,
            "the static predictive benchmark contains no directional-identification test",
        ),
        _not_applicable(
            "structure_function_association",
            source,
            "topographic readout regularity is not treated as synaptic structure-function evidence",
        ),
        _not_applicable(
            "causal_intervention",
            source,
            "the benchmark contains no causal intervention",
        ),
        _block(
            "whole_brain_coverage",
            EvidenceStatus.FAILED,
            source,
            "whole-brain coverage requires data and model validation beyond visual cortex",
            "Sensorium is a visual-cortex predictive benchmark",
        ),
        _block(
            "independent_validation",
            EvidenceStatus.FAILED,
            source,
            "independent validation requires a separate resource or study",
            "the stored analysis remains within Sensorium",
        ),
    )
    return RealEvidenceCase(
        name="sensorium_static_predictive_topographic",
        domain="Sensorium static visual cortex",
        sources=(source,),
        blocks=blocks,
        interpretation=(
            "Prediction and a topographic constraint pass within Sensorium. The case does not "
            "supply direction, intervention, whole-brain coverage, or external replication."
        ),
    )


def _dynamic_sensorium_case(payload: Mapping[str, Any], source: str) -> RealEvidenceCase:
    cohort_observations: dict[str, Any] = {}
    cohort_passes: list[bool] = []
    for cohort in payload.get("cohorts", []):
        paired = cohort.get("pairwise", {}).get("mean_response_vs_temporal_svd", {})
        n_paired = int(paired.get("n_paired", 0))
        right_wins = int(paired.get("right_wins", 0))
        median_delta = float(paired.get("median_delta", 0.0))
        passed = n_paired > 0 and right_wins >= 4 and median_delta > 0.0
        cohort_passes.append(passed)
        cohort_observations[str(cohort.get("cohort"))] = {
            "n_mice": cohort.get("n_mice"),
            "right_wins": right_wins,
            "median_delta": median_delta,
            "reliability_estimable_count": cohort.get("reliability_estimable_count"),
        }

    predictive_passed = len(cohort_passes) == 2 and all(cohort_passes)
    blocks = (
        _block(
            "prediction",
            EvidenceStatus.PASSED if predictive_passed else EvidenceStatus.FAILED,
            source,
            "temporal SVD must beat mean response in at least four of five mice with positive median delta in both cohorts",
            "the paired predictive comparison passes in both non-overlapping cohorts"
            if predictive_passed
            else "the paired predictive comparison does not pass in both cohorts",
            cohorts=cohort_observations,
        ),
        _compute_block(payload, source),
        _block(
            "internal_reproduction",
            EvidenceStatus.PASSED if predictive_passed else EvidenceStatus.FAILED,
            source,
            "the same signed predictive comparison must pass in both stored mouse cohorts",
            "the positive temporal-model comparison is reproduced across both cohorts"
            if predictive_passed
            else "the temporal-model comparison is not reproduced across both cohorts",
            cohorts=cohort_observations,
        ),
        _unknown(
            "external_replication",
            source,
            "the two cohorts remain internal to the Dynamic Sensorium resources used here",
        ),
        _not_applicable(
            "topology_specificity",
            source,
            "the dynamic comparator does not execute a topology-control experiment",
        ),
        _not_applicable(
            "directed_identifiability",
            source,
            "temporal prediction is not treated as identification of biological direction",
        ),
        _not_applicable(
            "structure_function_association",
            source,
            "the dynamic benchmark has no synaptic structure-function endpoint",
        ),
        _not_applicable(
            "causal_intervention",
            source,
            "the dynamic benchmark contains no intervention",
        ),
        _block(
            "whole_brain_coverage",
            EvidenceStatus.FAILED,
            source,
            "whole-brain coverage requires data and validation outside visual cortex",
            "Dynamic Sensorium is a visual-cortex prediction resource",
        ),
        _block(
            "independent_validation",
            EvidenceStatus.FAILED,
            source,
            "independent validation requires a separate external study",
            "both cohorts belong to the Sensorium ecosystem",
        ),
    )
    return RealEvidenceCase(
        name="dynamic_sensorium_temporal_prediction",
        domain="Dynamic Sensorium visual cortex",
        sources=(source,),
        blocks=blocks,
        interpretation=(
            "The temporal comparison is positive in both stored cohorts. Neural-response "
            "reliability is not estimable, and no topology, direction, or causal test is run."
        ),
    )


def _microns_case(
    robustness: Mapping[str, Any],
    q1_package: Mapping[str, Any],
    robustness_source: str,
    package_source: str,
) -> RealEvidenceCase:
    cohorts = q1_package.get("cohorts", [])
    primary_rows = [cohort.get("primary_test", {}) for cohort in cohorts]
    bootstrap_rows = [cohort.get("unit_bootstrap", {}) for cohort in cohorts]
    primary_passed = (
        q1_package.get("q1_package_ready") is True
        and robustness.get("all_cohorts_robust") is True
        and len(cohorts) == 3
        and all(row.get("confirmed_positive_after_fdr") is True for row in primary_rows)
        and all(
            float(row.get("distance_matched_delta", {}).get("ci95_low", 0.0)) > 0.0
            and float(row.get("degree_matched_delta", {}).get("ci95_low", 0.0)) > 0.0
            for row in bootstrap_rows
        )
    )
    observations = {
        str(cohort.get("cohort")): {
            "n_units": cohort.get("n_units"),
            "n_connected_pairs": cohort.get("n_connected_edge_pairs"),
            "distance_matched_delta": cohort.get("primary_test", {}).get("distance_matched_delta"),
            "distance_matched_q_one_sided": cohort.get("primary_test", {}).get(
                "distance_matched_q_one_sided"
            ),
            "distance_bootstrap_ci95": [
                cohort.get("unit_bootstrap", {})
                .get("distance_matched_delta", {})
                .get("ci95_low"),
                cohort.get("unit_bootstrap", {})
                .get("distance_matched_delta", {})
                .get("ci95_high"),
            ],
            "degree_matched_delta": cohort.get("primary_test", {}).get("degree_matched_delta"),
            "degree_matched_q_one_sided": cohort.get("primary_test", {}).get(
                "degree_matched_q_one_sided"
            ),
            "degree_bootstrap_ci95": [
                cohort.get("unit_bootstrap", {})
                .get("degree_matched_delta", {})
                .get("ci95_low"),
                cohort.get("unit_bootstrap", {})
                .get("degree_matched_delta", {})
                .get("ci95_high"),
            ],
        }
        for cohort in cohorts
    }

    blocks = (
        _not_applicable(
            "prediction",
            package_source,
            "the primary endpoint is an observational association rather than response prediction",
        ),
        _compute_block(q1_package, package_source),
        _block(
            "internal_reproduction",
            EvidenceStatus.PASSED if primary_passed else EvidenceStatus.FAILED,
            package_source,
            "the fixed endpoint must pass in discovery and two non-overlapping hold-outs",
            "the endpoint passes in all three within-resource windows"
            if primary_passed
            else "the endpoint does not pass in every window",
            cohorts=observations,
        ),
        _unknown(
            "external_replication",
            package_source,
            "the three windows come from one MICRONS volume and are not different animals or laboratories",
        ),
        _not_applicable(
            "topology_specificity",
            package_source,
            "distance- and degree-matched pair controls do not constitute a whole-network topology test",
        ),
        _not_applicable(
            "directed_identifiability",
            package_source,
            "directed synaptic edges are observed but functional direction is not identified",
        ),
        _block(
            "structure_function_association",
            EvidenceStatus.PASSED if primary_passed else EvidenceStatus.FAILED,
            package_source,
            "the fixed all_pairs/readout_location endpoint must pass FDR and have positive unit-cluster bootstrap lower bounds under distance and degree matching in all windows",
            "a bounded local observational association survives the declared controls"
            if primary_passed
            else "the fixed local association does not survive every declared control",
            primary_endpoint=q1_package.get("primary_endpoint"),
            cohorts=observations,
            robustness_decision=robustness.get("decision"),
        ),
        _not_applicable(
            "causal_intervention",
            package_source,
            "MICRONS supplies observational structure and function, not an intervention in this analysis",
        ),
        _block(
            "whole_brain_coverage",
            EvidenceStatus.FAILED,
            package_source,
            "whole-brain coverage requires brain-wide structural and functional evidence",
            "the sampled MICRONS volume is local visual cortex",
        ),
        _block(
            "independent_validation",
            EvidenceStatus.FAILED,
            package_source,
            "independent validation requires a different animal, resource, or laboratory",
            "the hold-outs are non-overlapping windows from the same resource",
        ),
    )
    return RealEvidenceCase(
        name="microns_local_structure_function",
        domain="MICRONS local visual cortex",
        sources=(robustness_source, package_source),
        blocks=blocks,
        interpretation=(
            "A fixed local structure-function endpoint is internally reproduced in two "
            "non-overlapping hold-outs. The result is observational and within one MICRONS volume."
        ),
    )


def build_cases(root: Path = Path(".")) -> tuple[RealEvidenceCase, ...]:
    """Load the four bounded real-data evidence cases."""

    allen_source = "results/allen_vbn_mechanistic_identifiability_score.json"
    sensorium_source = "results/sensorium_static_model_comparator/summary.json"
    dynamic_source = "results/dynamic_sensorium_model_comparator/summary.json"
    robustness_source = "results/microns_primary_robustness/summary.json"
    package_source = "results/microns_q1_package/summary.json"

    allen = _load_required_json(root, allen_source)
    sensorium = _load_required_json(root, sensorium_source)
    dynamic = _load_required_json(root, dynamic_source)
    robustness = _load_required_json(root, robustness_source)
    q1_package = _load_required_json(root, package_source)

    return (
        _allen_case(allen, allen_source),
        _sensorium_static_case(sensorium, sensorium_source),
        _dynamic_sensorium_case(dynamic, dynamic_source),
        _microns_case(robustness, q1_package, robustness_source, package_source),
    )


def run(
    output: Path = DEFAULT_OUTPUT,
    markdown: Path = DEFAULT_MARKDOWN,
    root: Path = Path("."),
) -> Path:
    """Evaluate real-data blocks and write a provenance-preserving matrix."""

    evaluator = EvidenceContractEvaluator()
    cases = build_cases(root)
    case_rows: list[dict[str, Any]] = []
    decision_rows: list[dict[str, Any]] = []
    for case in cases:
        indexed = blocks_by_name(case.blocks)
        if set(indexed) != set(ALL_BLOCKS):
            missing = sorted(set(ALL_BLOCKS) - set(indexed))
            extra = sorted(set(indexed) - set(ALL_BLOCKS))
            raise ValueError(f"invalid block inventory for {case.name}: missing={missing}, extra={extra}")
        decisions = evaluator.evaluate_all(indexed)
        case_rows.append(
            {
                "case": case.name,
                "domain": case.domain,
                "sources": list(case.sources),
                "interpretation": case.interpretation,
                "evidence_blocks": [block.as_dict() for block in case.blocks],
                "supported_claims": [
                    decision.claim
                    for decision in decisions
                    if decision.status is DecisionStatus.SUPPORTED
                ],
            }
        )
        decision_rows.extend(
            {"case": case.name, **decision.as_dict()} for decision in decisions
        )

    status_counts = {
        status.value: sum(row["status"] == status.value for row in decision_rows)
        for status in DecisionStatus
    }
    forbidden_supported = [
        row
        for row in decision_rows
        if row["status"] == DecisionStatus.SUPPORTED.value
        and row["claim"] in {"causal", "digital_twin", "externally_replicated"}
    ]
    payload = {
        "version": __version__,
        "git_revision": code_revision(),
        "analysis": "artifact_grounded_real_case_claim_matrix_v3",
        "contract": "domain-specific evidence blocks without cross-domain normalization",
        "cases": case_rows,
        "claim_decisions": decision_rows,
        "status_counts": status_counts,
        "forbidden_supported_claims": forbidden_supported,
        "limits": [
            "The case matrix is an artifact-grounded application of a declared policy, not independent ground truth.",
            "Internal reproduction does not mean replication across animals, laboratories, or resources.",
            "Unknown and not-applicable evidence are never converted into failed measurements.",
            "No real case supports a causal, externally replicated, whole-brain digital-twin claim.",
        ],
        "decision": (
            "artifact_grounded_case_matrix_complete_with_explicit_limits"
            if not forbidden_supported
            else "real_case_contract_requires_revision"
        ),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2))
    write_markdown(payload, markdown)
    return output


def write_markdown(payload: dict[str, Any], markdown: Path) -> None:
    """Write a concise human-readable audit report."""

    lines = [
        "# Artifact-Grounded Real-Case Claim Matrix v3",
        "",
        f"- Decision: `{payload['decision']}`",
        f"- Cases: `{len(payload['cases'])}`",
        f"- Status counts: `{payload['status_counts']}`",
        "",
        "| Case | Supported claims | Sources |",
        "|---|---|---|",
    ]
    for case in payload["cases"]:
        claims = ", ".join(f"`{claim}`" for claim in case["supported_claims"]) or "None"
        sources = "<br>".join(f"`{source}`" for source in case["sources"])
        lines.append(f"| `{case['case']}` | {claims} | {sources} |")
    lines.extend(["", "## Explicit Limits", ""])
    lines.extend(f"- {limit}" for limit in payload["limits"])
    lines.append("")
    markdown.parent.mkdir(parents=True, exist_ok=True)
    markdown.write_text("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--markdown", type=Path, default=DEFAULT_MARKDOWN)
    parser.add_argument("--root", type=Path, default=Path("."))
    args = parser.parse_args()
    print(json.dumps({"output": str(run(args.output, args.markdown, args.root).resolve())}))


if __name__ == "__main__":
    main()
