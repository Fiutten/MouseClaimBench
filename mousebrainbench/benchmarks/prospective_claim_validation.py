"""Prospective known-truth evaluation frozen by protocol version 1.0.0.

This module must not be used to tune the knowledge profile. Reference labels
come from previously unused data-generating processes, while policies receive
only noisy finite-sample evidence-block statuses. The development-trained
probabilistic comparator is loaded from its frozen artifact and cannot be
refitted here.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from scipy import stats

from mousebrainbench import __version__
from mousebrainbench.artifacts import code_revision
from mousebrainbench.benchmarks.oracle_sem_claim_benchmark import (
    GeneratedCohort,
    _classification_metrics,
    _confusion,
    _direction_diagnostic,
    _equal_weight_compensatory,
    _evidence_block,
    _prediction_diagnostic,
    _prediction_shortcut,
    _topology_diagnostic,
    _wilson_interval,
)
from mousebrainbench.benchmarks.prospective_probabilistic_baseline import (
    DEFAULT_PROTOCOL,
    encode_blocks,
    load_frozen_model,
    predict_claims,
)
from mousebrainbench.validation.evidence_contract import (
    CLAIM_REQUIREMENTS_V3,
    DecisionStatus,
    EvidenceBlock,
    EvidenceContractEvaluator,
    EvidenceStatus,
    blocks_by_name,
)


DEFAULT_MODEL = Path("results/prospective_probabilistic_baseline/model.json")
DEFAULT_OUTPUT = Path("results/prospective_claim_validation/summary.json")
DEFAULT_MARKDOWN = Path("results/prospective_claim_validation/summary.md")

REGIME_TRUTHS: dict[str, frozenset[str]] = {
    "independent_student": frozenset({"computationally_reproducible"}),
    "confounded_nonlinear": frozenset(
        {"predictive", "computationally_reproducible", "internally_reproduced"}
    ),
    "reverse_nonlinear": frozenset(
        {
            "predictive",
            "computationally_reproducible",
            "internally_reproduced",
            "topology_specific",
        }
    ),
    "direct_saturating": frozenset(
        {
            "predictive",
            "computationally_reproducible",
            "internally_reproduced",
            "topology_specific",
            "directed",
            "mechanistic",
        }
    ),
    "direct_heteroscedastic": frozenset(
        {
            "predictive",
            "computationally_reproducible",
            "internally_reproduced",
            "topology_specific",
            "directed",
            "mechanistic",
        }
    ),
    "weak_direct": frozenset(
        {
            "predictive",
            "computationally_reproducible",
            "internally_reproduced",
            "topology_specific",
            "directed",
            "mechanistic",
        }
    ),
    "shifted_direct": frozenset(
        {
            "predictive",
            "computationally_reproducible",
            "internally_reproduced",
            "topology_specific",
            "directed",
            "mechanistic",
        }
    ),
    "collider_selection": frozenset(
        {"predictive", "computationally_reproducible", "internally_reproduced"}
    ),
    "direct_interventional_saturating": frozenset(
        {
            "predictive",
            "computationally_reproducible",
            "internally_reproduced",
            "topology_specific",
            "directed",
            "mechanistic",
            "causal",
        }
    ),
}


@dataclass(frozen=True)
class ProspectiveCell:
    """One protocol cell before its deterministic seeds are expanded."""

    regime: str
    sample_size: int
    noise_scale: float


def _noise(rng: np.random.Generator, n: int, scale: float) -> np.ndarray:
    return rng.normal(scale=scale, size=n)


def _generate_cohort(
    regime: str,
    n: int,
    noise_scale: float,
    rng: np.random.Generator,
    cohort_index: int,
) -> GeneratedCohort:
    """Generate one cohort from a prospective-only structural equation."""

    u = rng.normal(size=(n, 3))
    if regime == "independent_student":
        x = rng.standard_t(df=3, size=n)
        y = rng.standard_t(df=3, size=n)
        controls = u
    elif regime == "confounded_nonlinear":
        z = rng.normal(size=n)
        x = np.tanh(1.1 * z) + _noise(rng, n, noise_scale)
        y = 0.9 * z + 0.35 * np.square(z) + _noise(rng, n, noise_scale)
        controls = np.column_stack((z, u[:, :2]))
    elif regime == "reverse_nonlinear":
        y = rng.normal(size=n)
        x = 1.2 * np.tanh(y) + 0.20 * np.square(y) + _noise(rng, n, noise_scale)
        controls = u
    elif regime == "direct_saturating":
        x = rng.standard_t(df=4, size=n)
        y = 1.2 * np.tanh(x) + _noise(rng, n, noise_scale)
        controls = u
    elif regime == "direct_heteroscedastic":
        x = rng.normal(size=n)
        local_scale = noise_scale * (0.35 + 0.55 * np.minimum(np.abs(x), 3.0))
        y = 0.9 * x + 0.15 * np.square(x) + rng.normal(scale=local_scale)
        controls = u
    elif regime == "weak_direct":
        x = rng.normal(size=n)
        y = 0.25 * x + _noise(rng, n, noise_scale)
        controls = u
    elif regime == "shifted_direct":
        location = (-1.5, 0.0, 1.5)[cohort_index]
        x = rng.normal(loc=location, size=n)
        y = 0.8 * x + 0.25 * np.square(x) + _noise(rng, n, noise_scale)
        controls = u
    elif regime == "collider_selection":
        candidate_n = n * 5
        candidate_x = rng.normal(size=candidate_n)
        candidate_y = rng.normal(size=candidate_n)
        collider = candidate_x + candidate_y + rng.normal(
            scale=noise_scale,
            size=candidate_n,
        )
        selected = np.flatnonzero(collider >= np.median(collider))[:n]
        x = candidate_x[selected]
        y = candidate_y[selected]
        selected_collider = collider[selected]
        controls = np.column_stack((selected_collider, rng.normal(size=(n, 2))))
    elif regime == "direct_interventional_saturating":
        x = rng.normal(size=n)
        y = 1.3 * np.tanh(x) + _noise(rng, n, noise_scale)
        controls = u
    else:
        raise ValueError(f"unknown prospective regime: {regime}")
    return GeneratedCohort(
        x=np.asarray(x, dtype=float),
        y=np.asarray(y, dtype=float),
        controls=np.asarray(controls, dtype=float),
    )


def _causal_diagnostic(
    regime: str,
    n: int,
    noise_scale: float,
    rng: np.random.Generator,
) -> dict[str, Any]:
    if regime != "direct_interventional_saturating":
        return {"available": False, "passed": False}
    y_plus = 1.3 * np.tanh(1.0) + _noise(rng, n, noise_scale)
    y_minus = 1.3 * np.tanh(-1.0) + _noise(rng, n, noise_scale)
    result = stats.ttest_ind(y_plus, y_minus, equal_var=False)
    effect = float(y_plus.mean() - y_minus.mean())
    return {
        "available": True,
        "passed": bool(effect > 0.0 and result.pvalue < 0.01),
        "mean_do_plus_minus_do_minus": effect,
        "p_value": float(result.pvalue),
    }


def _build_blocks(
    cell: ProspectiveCell,
    *,
    case_seed: int,
) -> tuple[EvidenceBlock, ...]:
    sequence = np.random.SeedSequence(case_seed)
    rng_train, rng_test, rng_rep, rng_intervention = [
        np.random.default_rng(child) for child in sequence.spawn(4)
    ]
    cohorts = (
        _generate_cohort(
            cell.regime,
            cell.sample_size,
            cell.noise_scale,
            rng_train,
            0,
        ),
        _generate_cohort(
            cell.regime,
            cell.sample_size,
            cell.noise_scale,
            rng_test,
            1,
        ),
        _generate_cohort(
            cell.regime,
            cell.sample_size,
            cell.noise_scale,
            rng_rep,
            2,
        ),
    )
    train, test, reproduction = cohorts
    prediction_a = _prediction_diagnostic(train, test)
    prediction_b = _prediction_diagnostic(test, reproduction)
    topology = _topology_diagnostic(train, test)
    direction = _direction_diagnostic(test)
    causal = _causal_diagnostic(
        cell.regime,
        cell.sample_size,
        cell.noise_scale,
        rng_intervention,
    )
    causal_status = (
        EvidenceStatus.PASSED
        if causal["available"] and causal["passed"]
        else EvidenceStatus.FAILED
        if causal["available"]
        else EvidenceStatus.NOT_APPLICABLE
    )
    return (
        _evidence_block(
            "prediction",
            EvidenceStatus.PASSED if prediction_a["passed"] else EvidenceStatus.FAILED,
            "held-out correlation > 0.10, p < 0.01, and R-squared > 0",
            prediction_a,
        ),
        _evidence_block(
            "reproducible_compute",
            EvidenceStatus.PASSED,
            "the prospective case is regenerated from a deterministic protocol seed",
            {
                "case_seed": case_seed,
                "sample_size": cell.sample_size,
                "noise_scale": cell.noise_scale,
            },
        ),
        _evidence_block(
            "internal_reproduction",
            EvidenceStatus.PASSED
            if prediction_a["passed"] and prediction_b["passed"]
            else EvidenceStatus.FAILED,
            "the predictive diagnostic passes in two independently generated cohorts",
            {"first": prediction_a, "second": prediction_b},
        ),
        _evidence_block(
            "external_replication",
            EvidenceStatus.NOT_APPLICABLE,
            "generated cases do not constitute external empirical replication",
            {},
        ),
        _evidence_block(
            "topology_specificity",
            EvidenceStatus.PASSED if topology["passed"] else EvidenceStatus.FAILED,
            "candidate held-out R-squared exceeds every control by more than 0.02",
            topology,
        ),
        _evidence_block(
            "directed_identifiability",
            EvidenceStatus.PASSED if direction["passed"] else EvidenceStatus.FAILED,
            "reverse-minus-forward residual-dependence margin exceeds 0.02",
            direction,
        ),
        _evidence_block(
            "structure_function_association",
            EvidenceStatus.NOT_APPLICABLE,
            "the generated variables are not a synaptic structure-function resource",
            {},
        ),
        _evidence_block(
            "causal_intervention",
            causal_status,
            "an available do(X) contrast is positive with Welch p < 0.01",
            causal,
        ),
        _evidence_block(
            "whole_brain_coverage",
            EvidenceStatus.FAILED,
            "a scalar SEM cannot provide whole-brain coverage",
            {},
        ),
        _evidence_block(
            "independent_validation",
            EvidenceStatus.FAILED,
            "generated cohorts are not an independent empirical validation study",
            {},
        ),
        _evidence_block(
            "entity_specificity",
            EvidenceStatus.FAILED,
            "a generated regime is not calibrated for one identified biological entity",
            {},
        ),
        _evidence_block(
            "operational_compute",
            EvidenceStatus.NOT_APPLICABLE,
            "the benchmark declares no deployment resource or latency budget",
            {},
        ),
    )


def _add_counts(target: dict[str, int], source: dict[str, int]) -> None:
    for key in ("tp", "fp", "tn", "fn"):
        target[key] += source[key]


def _stratified_case_bootstrap(
    rows: list[dict[str, Any]],
    policies: tuple[str, ...],
    *,
    replicates: int,
    seed: int,
) -> dict[str, dict[str, list[float]]]:
    """Resample complete cases within every frozen protocol cell."""

    rng = np.random.default_rng(seed)
    metric_order = ("tp", "fp", "tn", "fn")
    totals = {policy: np.zeros((replicates, 4), dtype=np.int64) for policy in policies}
    strata = sorted(
        {(row["regime"], row["sample_size"], row["noise_scale"]) for row in rows}
    )
    for stratum in strata:
        group = [
            row
            for row in rows
            if (row["regime"], row["sample_size"], row["noise_scale"]) == stratum
        ]
        indices = rng.integers(0, len(group), size=(replicates, len(group)))
        for policy in policies:
            values = np.asarray(
                [
                    [row["confusions"][policy][metric] for metric in metric_order]
                    for row in group
                ],
                dtype=np.int64,
            )
            totals[policy] += values[indices].sum(axis=1)

    intervals: dict[str, dict[str, list[float]]] = {}
    for policy, values in totals.items():
        tp, fp, tn, fn = values.T
        fpr = np.divide(fp, fp + tn, out=np.zeros(replicates), where=(fp + tn) > 0)
        fnr = np.divide(fn, fn + tp, out=np.zeros(replicates), where=(fn + tp) > 0)
        intervals[policy] = {
            "false_positive_rate_case_bootstrap_95": np.quantile(
                fpr, (0.025, 0.975)
            ).tolist(),
            "false_negative_rate_case_bootstrap_95": np.quantile(
                fnr, (0.025, 0.975)
            ).tolist(),
        }
    return intervals


def _paired_case_comparison(
    rows: list[dict[str, Any]],
    comparator: str,
) -> dict[str, Any]:
    contract_wins = comparator_wins = ties = 0
    for row in rows:
        contract = row["confusions"]["evidence_contract_v3"]
        other = row["confusions"][comparator]
        contract_errors = contract["fp"] + contract["fn"]
        comparator_errors = other["fp"] + other["fn"]
        if contract_errors < comparator_errors:
            contract_wins += 1
        elif comparator_errors < contract_errors:
            comparator_wins += 1
        else:
            ties += 1
    non_tied = contract_wins + comparator_wins
    p_value = (
        float(stats.binomtest(contract_wins, non_tied, p=0.5).pvalue)
        if non_tied
        else 1.0
    )
    return {
        "contract_fewer_errors": contract_wins,
        "comparator_fewer_errors": comparator_wins,
        "tied_errors": ties,
        "non_tied_cases": non_tied,
        "exact_two_sided_sign_test_p_value": p_value,
        "favors_contract": bool(contract_wins > comparator_wins and p_value < 0.05),
    }


def _summarize_groups(
    rows: list[dict[str, Any]],
    policies: tuple[str, ...],
    key: str,
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for value in sorted({row[key] for row in rows}):
        policy_rows = []
        for policy in policies:
            counts = {metric: 0 for metric in ("tp", "fp", "tn", "fn")}
            for row in rows:
                if row[key] == value:
                    _add_counts(counts, row["confusions"][policy])
            policy_rows.append({"policy": policy, **counts, **_classification_metrics(counts)})
        output.append({key: value, "policies": policy_rows})
    return output


def run(
    *,
    output: Path = DEFAULT_OUTPUT,
    markdown: Path = DEFAULT_MARKDOWN,
    protocol_path: Path = DEFAULT_PROTOCOL,
    model_path: Path = DEFAULT_MODEL,
    test_mode: bool = False,
    test_seeds_per_cell: int = 2,
    test_sample_sizes: tuple[int, ...] = (150,),
    test_noise_scales: tuple[float, ...] = (1.2,),
    test_bootstrap_replicates: int = 100,
) -> Path:
    """Execute the locked prospective benchmark or an explicitly ineligible test run."""

    protocol = yaml.safe_load(protocol_path.read_text())
    declared_regimes = tuple(protocol["prospective_partition"]["regimes"])
    if declared_regimes != tuple(REGIME_TRUTHS):
        raise ValueError("protocol regimes and immutable reference map differ")
    model = load_frozen_model(model_path, protocol_path=protocol_path)
    prospective = protocol["prospective_partition"]
    uncertainty = protocol["uncertainty"]
    sample_sizes = (
        test_sample_sizes if test_mode else tuple(int(v) for v in prospective["sample_sizes"])
    )
    noise_scales = (
        test_noise_scales
        if test_mode
        else tuple(float(v) for v in prospective["noise_scales"])
    )
    seeds_per_cell = test_seeds_per_cell if test_mode else int(prospective["seeds_per_cell"])
    bootstrap_replicates = (
        test_bootstrap_replicates if test_mode else int(uncertainty["replicates"])
    )
    seed_namespace = int(prospective["seed_namespace"])
    evaluator = EvidenceContractEvaluator()
    universe = tuple(requirement.claim for requirement in CLAIM_REQUIREMENTS_V3)
    policies = tuple(protocol["policies"])
    aggregate = {
        policy: {metric: 0 for metric in ("tp", "fp", "tn", "fn")}
        for policy in policies
    }
    per_claim = {
        policy: {
            claim: {metric: 0 for metric in ("tp", "fp", "tn", "fn")}
            for claim in universe
        }
        for policy in policies
    }
    case_rows: list[dict[str, Any]] = []
    cells = [
        ProspectiveCell(regime, sample_size, noise_scale)
        for regime in declared_regimes
        for sample_size in sample_sizes
        for noise_scale in noise_scales
    ]
    for cell_index, cell in enumerate(cells):
        reference = set(REGIME_TRUTHS[cell.regime])
        for seed in range(seeds_per_cell):
            case_seed = seed_namespace + cell_index * 10_000 + seed
            indexed = blocks_by_name(_build_blocks(cell, case_seed=case_seed))
            contract = {
                decision.claim
                for decision in evaluator.evaluate_all(indexed)
                if decision.status is DecisionStatus.SUPPORTED
            }
            predictions = {
                "evidence_contract_v3": contract,
                "equal_weight_compensatory_75": _equal_weight_compensatory(indexed),
                "prediction_shortcut": _prediction_shortcut(indexed),
                "development_trained_probabilistic": predict_claims(
                    model,
                    encode_blocks(indexed),
                ),
            }
            confusions: dict[str, dict[str, int]] = {}
            for policy, predicted in predictions.items():
                counts = _confusion(reference, predicted, universe)
                confusions[policy] = counts
                _add_counts(aggregate[policy], counts)
                for claim in universe:
                    expected = claim in reference
                    authorized = claim in predicted
                    metric = (
                        "tp"
                        if expected and authorized
                        else "fp"
                        if not expected and authorized
                        else "fn"
                        if expected
                        else "tn"
                    )
                    per_claim[policy][claim][metric] += 1
            case_rows.append(
                {
                    "regime": cell.regime,
                    "sample_size": cell.sample_size,
                    "noise_scale": cell.noise_scale,
                    "case_seed": case_seed,
                    "confusions": confusions,
                }
            )

    bootstrap = _stratified_case_bootstrap(
        case_rows,
        policies,
        replicates=bootstrap_replicates,
        seed=int(uncertainty["seed"]),
    )
    aggregate_rows = []
    for policy, counts in aggregate.items():
        aggregate_rows.append(
            {
                "policy": policy,
                **counts,
                **_classification_metrics(counts),
                **bootstrap[policy],
                "false_positive_rate_wilson_95": _wilson_interval(
                    counts["fp"], counts["fp"] + counts["tn"]
                ),
                "false_negative_rate_wilson_95": _wilson_interval(
                    counts["fn"], counts["fn"] + counts["tp"]
                ),
            }
        )
    per_claim_rows = [
        {"policy": policy, "claim": claim, **counts, **_classification_metrics(counts)}
        for policy, claims in per_claim.items()
        for claim, counts in claims.items()
    ]
    comparisons = {
        policy: _paired_case_comparison(case_rows, policy)
        for policy in policies
        if policy != "evidence_contract_v3"
    }
    metrics_by_policy = {row["policy"]: row for row in aggregate_rows}
    contract = metrics_by_policy["evidence_contract_v3"]
    comparators = [
        metrics_by_policy[policy]
        for policy in policies
        if policy != "evidence_contract_v3"
    ]
    lower_fpr_than_all = all(
        contract["false_positive_rate"] < row["false_positive_rate"]
        for row in comparators
    )
    fnr_within_margin = contract["false_negative_rate"] <= min(
        row["false_negative_rate"] for row in comparators
    ) + 0.10
    paired_favors_contract = all(row["favors_contract"] for row in comparisons.values())
    scale_matches_protocol = not test_mode
    primary_passed = bool(
        scale_matches_protocol
        and lower_fpr_than_all
        and fnr_within_margin
        and paired_favors_contract
    )
    payload = {
        "version": __version__,
        "git_revision": code_revision(),
        "analysis": "prospective_known_truth_claim_validation_v1",
        "protocol_id": protocol["protocol_id"],
        "protocol_version": str(protocol["version"]),
        "protocol_anchor_commit": "23bd4c7",
        "profile_hash": model["profile_hash"],
        "probabilistic_model_git_revision": model["git_revision"],
        "probabilistic_model_training_partition": model["training_partition"],
        "prospective_model_refitting_performed": False,
        "reference_label_source": "prespecified structural equations and intervention availability",
        "evidence_rule_source": "finite-sample diagnostics not used to define reference labels",
        "regimes": list(declared_regimes),
        "sample_sizes": list(sample_sizes),
        "noise_scales": list(noise_scales),
        "seeds_per_cell": seeds_per_cell,
        "num_protocol_cells": len(cells),
        "num_cases": len(case_rows),
        "num_claim_decisions_per_policy": len(case_rows) * len(universe),
        "scale_matches_frozen_protocol": scale_matches_protocol,
        "aggregate_by_policy": aggregate_rows,
        "aggregate_by_policy_and_claim": per_claim_rows,
        "by_regime": _summarize_groups(case_rows, policies, "regime"),
        "by_sample_size": _summarize_groups(case_rows, policies, "sample_size"),
        "by_noise_scale": _summarize_groups(case_rows, policies, "noise_scale"),
        "case_level_policy_comparisons": comparisons,
        "primary_endpoint": {
            "metric": "false_positive_rate",
            "lower_fpr_than_every_comparator": lower_fpr_than_all,
            "false_negative_rate_within_0.10_of_best_comparator": fnr_within_margin,
            "all_case_level_comparisons_favor_contract": paired_favors_contract,
            "passed": primary_passed,
        },
        "limits": list(protocol["prohibited_inferences"])
        + [
            "The known-truth systems remain low-dimensional abstractions rather than neural models.",
            "The selected regimes expand stress coverage but do not define a universal DGP distribution.",
            "Error rates validate the frozen operational policy, not the truth of a biological claim.",
        ],
        "decision": (
            "prospective_computational_primary_endpoint_passed"
            if primary_passed
            else "prospective_computational_primary_endpoint_not_passed"
        ),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2))
    write_markdown(payload, markdown)
    return output


def write_markdown(payload: dict[str, Any], markdown: Path) -> None:
    lines = [
        "# Prospective claim validation",
        "",
        f"- Decision: `{payload['decision']}`",
        f"- Cases: `{payload['num_cases']}`",
        f"- Decisions per policy: `{payload['num_claim_decisions_per_policy']}`",
        f"- Frozen scale executed: `{payload['scale_matches_frozen_protocol']}`",
        "",
        "| Policy | TP | FP | TN | FN | FPR | FNR |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in payload["aggregate_by_policy"]:
        lines.append(
            f"| `{row['policy']}` | {row['tp']} | {row['fp']} | {row['tn']} | "
            f"{row['fn']} | {row['false_positive_rate']:.4f} | "
            f"{row['false_negative_rate']:.4f} |"
        )
    lines.extend(["", "## Frozen primary endpoint", ""])
    for key, value in payload["primary_endpoint"].items():
        lines.append(f"- {key}: `{value}`")
    lines.extend(["", "## Limits", ""])
    lines.extend(f"- {limit}" for limit in payload["limits"])
    lines.append("")
    markdown.parent.mkdir(parents=True, exist_ok=True)
    markdown.write_text("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--markdown", type=Path, default=DEFAULT_MARKDOWN)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    args = parser.parse_args()
    print(
        json.dumps(
            {
                "output": str(
                    run(
                        output=args.output,
                        markdown=args.markdown,
                        protocol_path=args.protocol,
                        model_path=args.model,
                    ).resolve()
                )
            }
        )
    )


if __name__ == "__main__":
    main()

