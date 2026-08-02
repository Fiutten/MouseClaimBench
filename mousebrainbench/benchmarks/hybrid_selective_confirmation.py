"""Execute the frozen version-2 hybrid selective confirmation benchmark.

Reference labels are properties of prespecified structural equations. Policies
receive only finite-sample evidence. The frozen hybrid policy is loaded from a
development-only artifact and is never refitted by this module.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import time
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import yaml
from scipy import stats

from mousebrainbench import __version__
from mousebrainbench.artifacts import code_revision
from mousebrainbench.benchmarks.causal_direction_anm import (
    SUBSAMPLE_SEED_NAMESPACE,
    anm_direction_evidence,
)
from mousebrainbench.benchmarks.hybrid_development_features import (
    encode_hybrid_features,
    replace_direction_block,
)
from mousebrainbench.benchmarks.hybrid_selective_policy import (
    DEFAULT_OUTPUT as DEFAULT_POLICY,
    count_support_veto_violations,
    load_frozen_policy,
    predict_probabilities,
    selective_decisions,
)
from mousebrainbench.benchmarks.oracle_sem_claim_benchmark import (
    GeneratedCohort,
    _direction_diagnostic,
    _equal_weight_compensatory,
    _evidence_block,
    _prediction_diagnostic,
    _prediction_shortcut,
    _topology_diagnostic,
)
from mousebrainbench.validation.evidence_contract import (
    DecisionStatus,
    EvidenceBlock,
    EvidenceContractEvaluator,
    EvidenceStatus,
    blocks_by_name,
)


DEFAULT_PROTOCOL = Path("configs/benchmarks/hybrid_selective_claim_validation_v2.yaml")
DEFAULT_OUTPUT = Path("results/hybrid_selective_confirmation/summary.json")
DEFAULT_MARKDOWN = Path("results/hybrid_selective_confirmation/summary.md")
DEFAULT_CASES = Path("results/hybrid_selective_confirmation/cases.npz")

BASE_POLICIES = (
    "evidence_contract_v3",
    "equal_weight_compensatory_75",
    "prediction_shortcut",
    "unconstrained_selective_logistic",
    "constrained_selective_hybrid",
)
ABLATION_POLICIES = (
    "constrained_anm_predictor_ablation",
    "constrained_uncalibrated_ablation",
)

POSITIVE_STANDARD = frozenset(
    {
        "predictive",
        "computationally_reproducible",
        "internally_reproduced",
        "topology_specific",
        "directed",
        "mechanistic",
    }
)
CONFIRMATORY_TRUTHS: dict[str, frozenset[str]] = {
    "independent_mixture": frozenset({"computationally_reproducible"}),
    "confounded_threshold": frozenset(
        {"predictive", "computationally_reproducible", "internally_reproduced"}
    ),
    "reverse_piecewise": frozenset(
        {
            "predictive",
            "computationally_reproducible",
            "internally_reproduced",
            "topology_specific",
        }
    ),
    "direct_exponential": POSITIVE_STANDARD,
    "direct_piecewise": POSITIVE_STANDARD,
    "direct_post_nonlinear": POSITIVE_STANDARD,
    "direct_linear_nongaussian": POSITIVE_STANDARD,
    "measurement_error_direct": POSITIVE_STANDARD,
    "collider_truncation": frozenset(
        {"predictive", "computationally_reproducible", "internally_reproduced"}
    ),
    "direct_interventional_piecewise": POSITIVE_STANDARD | {"causal"},
}

DIRECTION_TRUTHS: dict[str, str | None] = {
    "independent_mixture": None,
    "confounded_threshold": None,
    "reverse_piecewise": "y_to_x",
    "direct_exponential": "x_to_y",
    "direct_piecewise": "x_to_y",
    "direct_post_nonlinear": "x_to_y",
    "direct_linear_nongaussian": "x_to_y",
    "measurement_error_direct": "x_to_y",
    "collider_truncation": None,
    "direct_interventional_piecewise": "x_to_y",
}


@dataclass(frozen=True)
class ConfirmatoryCell:
    """One frozen regime-by-scale cell before seed expansion."""

    regime: str
    sample_size: int
    noise_scale: float


def _sha256(path: Path) -> str:
    return f"sha256:{hashlib.sha256(path.read_bytes()).hexdigest()}"


def _noise(rng: np.random.Generator, n: int, scale: float) -> np.ndarray:
    return rng.normal(scale=scale, size=n)


def _mixture(rng: np.random.Generator, n: int) -> np.ndarray:
    component = rng.integers(0, 2, size=n)
    return rng.normal(loc=np.where(component == 0, -1.4, 1.4), scale=0.65)


def _generate_cohort(
    regime: str,
    n: int,
    noise_scale: float,
    rng: np.random.Generator,
    cohort_index: int,
) -> GeneratedCohort:
    """Generate one cohort from a v2-only structural equation."""

    del cohort_index
    nuisance = rng.normal(size=(n, 3))
    if regime == "independent_mixture":
        x = _mixture(rng, n)
        y = _mixture(rng, n)
        controls = nuisance
    elif regime == "confounded_threshold":
        z = rng.normal(size=n)
        x = z + _noise(rng, n, noise_scale)
        y = np.where(z >= 0.0, 1.0, -1.0) + _noise(rng, n, noise_scale)
        controls = np.column_stack((z, nuisance[:, :2]))
    elif regime == "reverse_piecewise":
        y = rng.normal(size=n)
        x = np.where(y < 0.0, 0.45 * y, 1.20 * y) + _noise(
            rng, n, noise_scale
        )
        controls = nuisance
    elif regime == "direct_exponential":
        x = rng.uniform(-2.0, 2.0, size=n)
        y = np.exp(0.55 * x) + _noise(rng, n, noise_scale)
        controls = nuisance
    elif regime in {"direct_piecewise", "direct_interventional_piecewise"}:
        x = rng.normal(size=n)
        y = np.where(x < 0.0, 0.40 * x, 1.25 * x) + _noise(
            rng, n, noise_scale
        )
        controls = nuisance
    elif regime == "direct_post_nonlinear":
        x = rng.normal(size=n)
        latent = 0.90 * x + _noise(rng, n, noise_scale)
        y = 2.0 * np.tanh(latent)
        controls = nuisance
    elif regime == "direct_linear_nongaussian":
        x = rng.laplace(size=n)
        y = 0.85 * x + noise_scale * rng.standard_t(df=5, size=n)
        controls = nuisance
    elif regime == "measurement_error_direct":
        latent_x = rng.normal(size=n)
        x = latent_x + _noise(rng, n, 0.45 * noise_scale)
        y = 0.95 * latent_x + _noise(rng, n, noise_scale)
        controls = nuisance
    elif regime == "collider_truncation":
        candidate_n = n * 5
        candidate_x = rng.normal(size=candidate_n)
        candidate_y = rng.normal(size=candidate_n)
        collider = candidate_x + candidate_y + rng.normal(
            scale=noise_scale, size=candidate_n
        )
        threshold = np.quantile(collider, 0.60)
        selected = np.flatnonzero(collider >= threshold)[:n]
        x = candidate_x[selected]
        y = candidate_y[selected]
        controls = np.column_stack(
            (collider[selected], rng.normal(size=(n, 2)))
        )
    else:
        raise ValueError(f"unknown v2 confirmatory regime: {regime}")
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
    if regime != "direct_interventional_piecewise":
        return {"available": False, "passed": False}
    y_plus = 1.25 + _noise(rng, n, noise_scale)
    y_minus = -0.40 + _noise(rng, n, noise_scale)
    result = stats.ttest_ind(y_plus, y_minus, equal_var=False)
    effect = float(y_plus.mean() - y_minus.mean())
    return {
        "available": True,
        "passed": bool(effect > 0.0 and result.pvalue < 0.01),
        "mean_do_plus_minus_do_minus": effect,
        "p_value": float(result.pvalue),
    }


def _build_blocks_and_test_cohort(
    cell: ConfirmatoryCell,
    case_seed: int,
) -> tuple[dict[str, EvidenceBlock], GeneratedCohort]:
    sequence = np.random.SeedSequence(case_seed)
    rng_train, rng_test, rng_rep, rng_intervention = [
        np.random.default_rng(child) for child in sequence.spawn(4)
    ]
    train = _generate_cohort(
        cell.regime, cell.sample_size, cell.noise_scale, rng_train, 0
    )
    test = _generate_cohort(
        cell.regime, cell.sample_size, cell.noise_scale, rng_test, 1
    )
    reproduction = _generate_cohort(
        cell.regime, cell.sample_size, cell.noise_scale, rng_rep, 2
    )
    prediction_a = _prediction_diagnostic(train, test)
    prediction_b = _prediction_diagnostic(test, reproduction)
    topology = _topology_diagnostic(train, test)
    direction = _direction_diagnostic(test)
    causal = _causal_diagnostic(
        cell.regime, cell.sample_size, cell.noise_scale, rng_intervention
    )
    causal_status = (
        EvidenceStatus.PASSED
        if causal["available"] and causal["passed"]
        else EvidenceStatus.FAILED
        if causal["available"]
        else EvidenceStatus.NOT_APPLICABLE
    )
    blocks = blocks_by_name(
        (
            _evidence_block(
                "prediction",
                EvidenceStatus.PASSED
                if prediction_a["passed"]
                else EvidenceStatus.FAILED,
                "held-out correlation > 0.10, p < 0.01, and R-squared > 0",
                prediction_a,
            ),
            _evidence_block(
                "reproducible_compute",
                EvidenceStatus.PASSED,
                "the case is regenerated from the frozen protocol seed",
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
                "the diagnostic passes in two independently generated cohorts",
                {"first": prediction_a, "second": prediction_b},
            ),
            _evidence_block(
                "external_replication",
                EvidenceStatus.NOT_APPLICABLE,
                "generated cases are not external empirical replication",
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
                "legacy residual-dependence direction diagnostic",
                direction,
            ),
            _evidence_block(
                "structure_function_association",
                EvidenceStatus.NOT_APPLICABLE,
                "generated scalar variables are not a structure-function resource",
                {},
            ),
            _evidence_block(
                "causal_intervention",
                causal_status,
                "available do(X) contrast is positive with Welch p < 0.01",
                causal,
            ),
            _evidence_block(
                "whole_brain_coverage",
                EvidenceStatus.FAILED,
                "a scalar structural equation has no whole-brain coverage",
                {},
            ),
            _evidence_block(
                "independent_validation",
                EvidenceStatus.FAILED,
                "generated cohorts are not independent empirical validation",
                {},
            ),
            _evidence_block(
                "entity_specificity",
                EvidenceStatus.FAILED,
                "a generated regime is not one identified biological entity",
                {},
            ),
            _evidence_block(
                "operational_compute",
                EvidenceStatus.NOT_APPLICABLE,
                "the benchmark declares no deployment resource budget",
                {},
            ),
        )
    )
    return blocks, test


def _contract_decisions(
    evaluator: EvidenceContractEvaluator,
    blocks: Mapping[str, EvidenceBlock],
    claim_names: Sequence[str],
) -> np.ndarray:
    by_claim = {decision.claim: decision for decision in evaluator.evaluate_all(blocks)}
    values = []
    for claim in claim_names:
        status = by_claim[claim].status
        values.append(
            1
            if status is DecisionStatus.SUPPORTED
            else -1
            if status is DecisionStatus.BLOCKED
            else 0
        )
    return np.asarray(values, dtype=np.int8)


def _set_decisions(predicted: set[str], claim_names: Sequence[str]) -> np.ndarray:
    return np.asarray([1 if claim in predicted else -1 for claim in claim_names], dtype=np.int8)


def _build_case(
    task: tuple[ConfirmatoryCell, int, tuple[str, ...], Callable[..., dict[str, Any]]],
) -> dict[str, Any]:
    cell, case_seed, claim_names, direction_function = task
    blocks, test = _build_blocks_and_test_cohort(cell, case_seed)
    legacy_direction = dict(blocks["directed_identifiability"].observations)
    anm = direction_function(
        test.x,
        test.y,
        seed=SUBSAMPLE_SEED_NAMESPACE + case_seed,
    )
    updated = replace_direction_block(blocks, anm)
    evaluator = EvidenceContractEvaluator()
    return {
        "regime": cell.regime,
        "sample_size": cell.sample_size,
        "noise_scale": cell.noise_scale,
        "case_seed": case_seed,
        "features": encode_hybrid_features(
            updated,
            legacy_direction=legacy_direction,
            anm=anm,
            sample_size=cell.sample_size,
            noise_scale=cell.noise_scale,
        ),
        "labels": np.asarray(
            [claim in CONFIRMATORY_TRUTHS[cell.regime] for claim in claim_names],
            dtype=np.uint8,
        ),
        "contract": _contract_decisions(evaluator, updated, claim_names),
        "compensatory": _set_decisions(
            _equal_weight_compensatory(updated), claim_names
        ),
        "prediction_shortcut": _set_decisions(_prediction_shortcut(updated), claim_names),
        "anm_status": str(anm["status"]),
        "anm_direction": str(anm["predicted_direction"]),
        "anm_error": anm["execution_error"],
    }


def _decision_metrics(decisions: np.ndarray, labels: np.ndarray) -> dict[str, Any]:
    decided = decisions != 0
    support = decisions == 1
    blocked = decisions == -1
    errors = (support & ~labels) | (blocked & labels)
    decision_count = int(decided.sum())
    error_count = int((errors & decided).sum())
    support_count = int(support.sum())
    return {
        "total": int(decisions.size),
        "decisions": decision_count,
        "abstentions": int((~decided).sum()),
        "supports": support_count,
        "blocks": int(blocked.sum()),
        "errors": error_count,
        "false_authorizations": int((support & ~labels).sum()),
        "coverage": float(decision_count / decisions.size),
        "selective_error": float(error_count / decision_count) if decision_count else 1.0,
        "selective_error_cp95_upper": _cp_upper(error_count, decision_count),
        "false_authorization_fraction": (
            float((support & ~labels).sum() / support_count) if support_count else 0.0
        ),
    }


def _cp_upper(errors: int, decisions: int) -> float:
    if decisions == 0:
        return 1.0
    if errors == decisions:
        return 1.0
    return float(stats.beta.ppf(0.95, errors + 1, decisions - errors))


def _summarize_groups(
    decisions: Mapping[str, np.ndarray],
    labels: np.ndarray,
    values: np.ndarray,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for value in sorted(set(values.tolist())):
        mask = values == value
        rows.append(
            {
                "value": value.item() if hasattr(value, "item") else value,
                "policies": [
                    {"policy": policy, **_decision_metrics(array[mask], labels[mask].astype(bool))}
                    for policy, array in decisions.items()
                ],
            }
        )
    return rows


def _risk_coverage_curve(
    probabilities: np.ndarray,
    features: np.ndarray,
    labels: np.ndarray,
    *,
    policy: Mapping[str, Any],
    constrained: bool,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for threshold in np.round(np.arange(0.50, 1.00, 0.01), 2):
        selected, violations = selective_decisions(
            probabilities,
            features,
            threshold=float(threshold),
            claim_names=policy["claim_names"],
            feature_names=policy["feature_names"],
            support_vetoes=policy["support_vetoes"],
            constrained=constrained,
        )
        rows.append(
            {
                "threshold": float(threshold),
                "semantic_support_veto_violations": violations,
                **_decision_metrics(selected, labels.astype(bool)),
            }
        )
    ordered = sorted(rows, key=lambda row: row["coverage"])
    area = float(
        np.trapezoid(
            [row["selective_error"] for row in ordered],
            [row["coverage"] for row in ordered],
        )
    )
    return {
        "area_under_observed_risk_coverage_curve": area,
        "area_definition": "trapezoid over the frozen threshold grid's observed coverage range",
        "curve": rows,
    }


def _paired_comparison(
    hybrid: np.ndarray,
    comparator: np.ndarray,
    labels: np.ndarray,
) -> dict[str, Any]:
    def loss(decisions: np.ndarray) -> np.ndarray:
        expected = labels.astype(bool)
        error = ((decisions == 1) & ~expected) | ((decisions == -1) & expected)
        return error.sum(axis=1) + 0.5 * (decisions == 0).sum(axis=1)

    hybrid_loss = loss(hybrid)
    comparator_loss = loss(comparator)
    hybrid_wins = int((hybrid_loss < comparator_loss).sum())
    comparator_wins = int((comparator_loss < hybrid_loss).sum())
    ties = int((hybrid_loss == comparator_loss).sum())
    non_tied = hybrid_wins + comparator_wins
    p_value = (
        float(stats.binomtest(hybrid_wins, non_tied, p=0.5).pvalue)
        if non_tied
        else 1.0
    )
    return {
        "loss": "claim errors plus 0.5 per abstention within a complete case",
        "hybrid_lower_loss": hybrid_wins,
        "comparator_lower_loss": comparator_wins,
        "ties": ties,
        "exact_two_sided_sign_test_p_value": p_value,
    }


def _bootstrap(
    decisions: Mapping[str, np.ndarray],
    labels: np.ndarray,
    regimes: np.ndarray,
    sample_sizes: np.ndarray,
    noise_scales: np.ndarray,
    *,
    replicates: int,
    seed: int,
) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    policies = tuple(decisions)
    totals = {
        policy: np.zeros((replicates, 6), dtype=np.int64) for policy in policies
    }
    strata = sorted(set(zip(regimes.tolist(), sample_sizes.tolist(), noise_scales.tolist())))
    for stratum in strata:
        mask = (
            (regimes == stratum[0])
            & (sample_sizes == stratum[1])
            & (noise_scales == stratum[2])
        )
        indices_in_stratum = np.flatnonzero(mask)
        sampled = rng.integers(
            0, len(indices_in_stratum), size=(replicates, len(indices_in_stratum))
        )
        expected = labels[mask].astype(bool)
        for policy in policies:
            observed = decisions[policy][mask]
            support = observed == 1
            blocked = observed == -1
            case_values = np.column_stack(
                (
                    (observed != 0).sum(axis=1),
                    (((support & ~expected) | (blocked & expected))).sum(axis=1),
                    support.sum(axis=1),
                    (support & ~expected).sum(axis=1),
                    np.full(len(observed), observed.shape[1]),
                    (observed == 0).sum(axis=1),
                )
            )
            totals[policy] += case_values[sampled].sum(axis=1)

    output: dict[str, Any] = {}
    for policy, values in totals.items():
        decided, errors, supports, false_supports, total, _ = values.T
        coverage = decided / total
        risk = np.divide(errors, decided, out=np.ones(replicates), where=decided > 0)
        false_fraction = np.divide(
            false_supports,
            supports,
            out=np.zeros(replicates),
            where=supports > 0,
        )
        output[policy] = {
            "coverage_case_bootstrap_95": np.quantile(coverage, (0.025, 0.975)).tolist(),
            "selective_error_case_bootstrap_95": np.quantile(
                risk, (0.025, 0.975)
            ).tolist(),
            "false_authorization_fraction_case_bootstrap_95": np.quantile(
                false_fraction, (0.025, 0.975)
            ).tolist(),
        }
    return output


def _direction_metrics(
    regimes: np.ndarray,
    statuses: np.ndarray,
    predicted: np.ndarray,
) -> dict[str, Any]:
    attempted = statuses != "requires_review"
    truth = np.asarray([DIRECTION_TRUTHS[regime] or "none" for regime in regimes])
    normalized = np.asarray(
        [
            "x_to_y"
            if value == "forward"
            else "y_to_x"
            if value == "reverse"
            else "none"
            for value in predicted
        ]
    )
    correct = attempted & (normalized == truth)
    directional = truth != "none"
    directional_attempted = attempted & directional
    return {
        "attempts": int(attempted.sum()),
        "cases": int(len(regimes)),
        "coverage": float(attempted.mean()),
        "attempted_accuracy_all_regimes": (
            float(correct.sum() / attempted.sum()) if attempted.any() else 0.0
        ),
        "attempted_accuracy_identifiable_direction_regimes": (
            float((correct & directional).sum() / directional_attempted.sum())
            if directional_attempted.any()
            else 0.0
        ),
        "spurious_attempts_in_no_direction_regimes": int((attempted & ~directional).sum()),
        "ambiguous_cases": int((~attempted).sum()),
    }


def run(
    *,
    output: Path = DEFAULT_OUTPUT,
    markdown: Path = DEFAULT_MARKDOWN,
    cases_path: Path = DEFAULT_CASES,
    protocol_path: Path = DEFAULT_PROTOCOL,
    policy_path: Path = DEFAULT_POLICY,
    workers: int = 1,
    test_mode: bool = False,
    test_seeds_per_cell: int = 1,
    test_sample_sizes: tuple[int, ...] = (200,),
    test_noise_scales: tuple[float, ...] = (1.0,),
    test_bootstrap_replicates: int = 20,
    direction_function: Callable[..., dict[str, Any]] = anm_direction_evidence,
) -> Path:
    """Execute the frozen confirmation or an explicitly ineligible smoke run."""

    started = time.perf_counter()
    protocol = yaml.safe_load(protocol_path.read_text())
    declared_regimes = tuple(protocol["confirmatory_partition_v2"]["regimes"])
    if declared_regimes != tuple(CONFIRMATORY_TRUTHS):
        raise ValueError("protocol regimes and immutable v2 truth map differ")
    policy = load_frozen_policy(policy_path, protocol_path=protocol_path)
    partition = protocol["confirmatory_partition_v2"]
    sample_sizes = (
        test_sample_sizes
        if test_mode
        else tuple(int(value) for value in partition["sample_sizes"])
    )
    noise_scales = (
        test_noise_scales
        if test_mode
        else tuple(float(value) for value in partition["noise_scales"])
    )
    seeds_per_cell = (
        test_seeds_per_cell if test_mode else int(partition["seeds_per_cell"])
    )
    bootstrap_replicates = (
        test_bootstrap_replicates
        if test_mode
        else int(protocol["uncertainty"]["replicates"])
    )
    cells = [
        ConfirmatoryCell(regime, sample_size, noise_scale)
        for regime in declared_regimes
        for sample_size in sample_sizes
        for noise_scale in noise_scales
    ]
    tasks = []
    namespace = int(partition["seed_namespace"])
    for cell_index, cell in enumerate(cells):
        for seed in range(seeds_per_cell):
            tasks.append(
                (
                    cell,
                    namespace + cell_index * 10_000 + seed,
                    tuple(policy["claim_names"]),
                    direction_function,
                )
            )
    if workers > 1:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            records = list(executor.map(_build_case, tasks, chunksize=1))
    else:
        records = [_build_case(task) for task in tasks]

    features = np.vstack([record["features"] for record in records])
    labels = np.vstack([record["labels"] for record in records]).astype(bool)
    regimes = np.asarray([record["regime"] for record in records])
    sample_size_array = np.asarray([record["sample_size"] for record in records])
    noise_scale_array = np.asarray([record["noise_scale"] for record in records])
    case_seeds = np.asarray([record["case_seed"] for record in records], dtype=np.int64)
    anm_status = np.asarray([record["anm_status"] for record in records])
    anm_direction = np.asarray([record["anm_direction"] for record in records])

    full_probabilities = predict_probabilities(
        policy["model_sets"]["full"], features, policy["claim_names"]
    )
    raw_probabilities = predict_probabilities(
        policy["model_sets"]["full"],
        features,
        policy["claim_names"],
        calibrated=False,
    )
    ablation_probabilities = predict_probabilities(
        policy["model_sets"]["anm_predictor_ablation"],
        features,
        policy["claim_names"],
    )
    constrained, constrained_violations = selective_decisions(
        full_probabilities,
        features,
        threshold=float(policy["selected_thresholds"]["full"]),
        claim_names=policy["claim_names"],
        feature_names=policy["feature_names"],
        support_vetoes=policy["support_vetoes"],
        constrained=True,
    )
    unconstrained, _ = selective_decisions(
        full_probabilities,
        features,
        threshold=float(policy["selected_thresholds"]["unconstrained_full"]),
        claim_names=policy["claim_names"],
        feature_names=policy["feature_names"],
        support_vetoes=policy["support_vetoes"],
        constrained=False,
    )
    anm_ablation, _ = selective_decisions(
        ablation_probabilities,
        features,
        threshold=float(policy["selected_thresholds"]["anm_predictor_ablation"]),
        claim_names=policy["claim_names"],
        feature_names=policy["feature_names"],
        support_vetoes=policy["support_vetoes"],
        constrained=True,
    )
    calibration_ablation, _ = selective_decisions(
        raw_probabilities,
        features,
        threshold=float(policy["selected_thresholds"]["full"]),
        claim_names=policy["claim_names"],
        feature_names=policy["feature_names"],
        support_vetoes=policy["support_vetoes"],
        constrained=True,
    )
    decisions = {
        "evidence_contract_v3": np.vstack([record["contract"] for record in records]),
        "equal_weight_compensatory_75": np.vstack(
            [record["compensatory"] for record in records]
        ),
        "prediction_shortcut": np.vstack(
            [record["prediction_shortcut"] for record in records]
        ),
        "unconstrained_selective_logistic": unconstrained,
        "constrained_selective_hybrid": constrained,
        "constrained_anm_predictor_ablation": anm_ablation,
        "constrained_uncalibrated_ablation": calibration_ablation,
    }
    semantic_violations = {
        name: count_support_veto_violations(
            values,
            features,
            claim_names=policy["claim_names"],
            feature_names=policy["feature_names"],
            support_vetoes=policy["support_vetoes"],
        )
        for name, values in decisions.items()
    }
    if constrained_violations != semantic_violations["constrained_selective_hybrid"]:
        raise RuntimeError("semantic veto postcondition disagrees with independent audit")

    aggregate = {
        name: {
            "policy": name,
            "semantic_support_veto_violations": semantic_violations[name],
            **_decision_metrics(values, labels),
        }
        for name, values in decisions.items()
    }
    bootstrap = _bootstrap(
        decisions,
        labels,
        regimes,
        sample_size_array,
        noise_scale_array,
        replicates=bootstrap_replicates,
        seed=int(protocol["uncertainty"]["seed"]),
    )
    for name in aggregate:
        aggregate[name].update(bootstrap[name])

    direction = _direction_metrics(regimes, anm_status, anm_direction)
    hybrid = aggregate["constrained_selective_hybrid"]
    unconstrained_metrics = aggregate["unconstrained_selective_logistic"]
    scale_matches = bool(
        not test_mode
        and len(records) == int(partition["expected_cases"])
        and len(records) * len(policy["claim_names"])
        == int(partition["expected_claim_decisions_per_policy"])
    )
    conditions = {
        "scale_matches_frozen_protocol": scale_matches,
        "semantic_support_veto_violations_equal_0": hybrid[
            "semantic_support_veto_violations"
        ]
        == 0,
        "constrained_hybrid_coverage_at_least_0.30": hybrid["coverage"] >= 0.30,
        "constrained_hybrid_selective_error_cp95_upper_at_most_0.12": hybrid[
            "selective_error_cp95_upper"
        ]
        <= 0.12,
        "constrained_hybrid_false_authorization_fraction_at_most_0.08": hybrid[
            "false_authorization_fraction"
        ]
        <= 0.08,
        "constrained_hybrid_selective_error_no_more_than_0.02_above_unconstrained": hybrid[
            "selective_error"
        ]
        <= unconstrained_metrics["selective_error"] + 0.02,
        "anm_direction_attempted_accuracy_at_least_0.75": direction[
            "attempted_accuracy_all_regimes"
        ]
        >= 0.75,
        "anm_direction_coverage_at_least_0.20": direction["coverage"] >= 0.20,
    }
    primary_passed = all(conditions.values())

    cases_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cases_path,
        features=features,
        labels=labels.astype(np.uint8),
        regimes=regimes,
        sample_sizes=sample_size_array,
        noise_scales=noise_scale_array,
        case_seeds=case_seeds,
        anm_status=anm_status,
        anm_direction=anm_direction,
        policy_names=np.asarray(tuple(decisions)),
        policy_decisions=np.stack([decisions[name] for name in decisions]),
        full_calibrated_probabilities=full_probabilities,
        full_raw_probabilities=raw_probabilities,
    )
    payload = {
        "version": __version__,
        "git_revision": code_revision(),
        "analysis": "hybrid_selective_known_truth_confirmation_v2",
        "protocol_id": protocol["protocol_id"],
        "protocol_version": str(protocol["version"]),
        "protocol_hash": _sha256(protocol_path),
        "frozen_policy_hash": _sha256(policy_path),
        "frozen_policy_git_revision": policy["git_revision"],
        "confirmatory_model_refitting_performed": False,
        "test_mode": test_mode,
        "scale_matches_frozen_protocol": scale_matches,
        "regimes": list(declared_regimes),
        "sample_sizes": list(sample_sizes),
        "noise_scales": list(noise_scales),
        "seeds_per_cell": seeds_per_cell,
        "workers": workers,
        "num_cases": len(records),
        "num_claim_decisions_per_policy": len(records) * len(policy["claim_names"]),
        "cases_file": str(cases_path),
        "cases_sha256": _sha256(cases_path),
        "wall_time_seconds": time.perf_counter() - started,
        "aggregate_by_policy": [aggregate[name] for name in decisions],
        "by_regime": _summarize_groups(decisions, labels, regimes),
        "by_sample_size": _summarize_groups(decisions, labels, sample_size_array),
        "by_noise_scale": _summarize_groups(decisions, labels, noise_scale_array),
        "per_claim": [
            {
                "claim": claim,
                "policies": [
                    {
                        "policy": name,
                        **_decision_metrics(
                            values[:, claim_index], labels[:, claim_index]
                        ),
                    }
                    for name, values in decisions.items()
                ],
            }
            for claim_index, claim in enumerate(policy["claim_names"])
        ],
        "directional_evidence": {
            **direction,
            "execution_errors": sum(record["anm_error"] is not None for record in records),
            "status_counts": {
                status: int((anm_status == status).sum())
                for status in ("passed", "failed", "requires_review")
            },
        },
        "risk_coverage": {
            "constrained_selective_hybrid": _risk_coverage_curve(
                full_probabilities,
                features,
                labels,
                policy=policy,
                constrained=True,
            ),
            "unconstrained_selective_logistic": _risk_coverage_curve(
                full_probabilities,
                features,
                labels,
                policy=policy,
                constrained=False,
            ),
        },
        "paired_case_comparisons": {
            name: _paired_comparison(constrained, values, labels)
            for name, values in decisions.items()
            if name != "constrained_selective_hybrid"
        },
        "primary_endpoint": {
            "all_conditions_required": True,
            "conditions": conditions,
            "passed": primary_passed,
        },
        "limits": list(protocol["prohibited_inferences"])
        + [
            "Known-truth regimes are low-dimensional stress tests, not neural models.",
            "ANM direction remains assumption-conditional and is not causal proof.",
            "The measurement-error label refers to the latent generating direction.",
            "No expert content-validity or human-utility study was performed.",
        ],
        "decision": (
            "hybrid_selective_v2_primary_endpoint_passed"
            if primary_passed
            else "hybrid_selective_v2_primary_endpoint_not_passed"
        ),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2))
    write_markdown(payload, markdown)
    return output


def write_markdown(payload: Mapping[str, Any], markdown: Path) -> None:
    lines = [
        "# Hybrid selective confirmation v2",
        "",
        f"- Decision: `{payload['decision']}`",
        f"- Cases: `{payload['num_cases']}`",
        f"- Frozen scale: `{payload['scale_matches_frozen_protocol']}`",
        f"- Confirmatory refitting: `{payload['confirmatory_model_refitting_performed']}`",
        "",
        "| Policy | Coverage | Selective error | CP95 upper | False auth. fraction | Veto violations |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in payload["aggregate_by_policy"]:
        lines.append(
            f"| `{row['policy']}` | {row['coverage']:.4f} | "
            f"{row['selective_error']:.4f} | {row['selective_error_cp95_upper']:.4f} | "
            f"{row['false_authorization_fraction']:.4f} | "
            f"{row['semantic_support_veto_violations']} |"
        )
    lines.extend(["", "## Primary endpoint", ""])
    for name, value in payload["primary_endpoint"]["conditions"].items():
        lines.append(f"- {name}: `{value}`")
    lines.extend(["", "## Directional evidence", ""])
    for name, value in payload["directional_evidence"].items():
        lines.append(f"- {name}: `{value}`")
    lines.extend(["", "## Limits", ""])
    lines.extend(f"- {limit}" for limit in payload["limits"])
    lines.append("")
    markdown.parent.mkdir(parents=True, exist_ok=True)
    markdown.write_text("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--markdown", type=Path, default=DEFAULT_MARKDOWN)
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--policy", type=Path, default=DEFAULT_POLICY)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--test-mode", action="store_true")
    args = parser.parse_args()
    print(
        json.dumps(
            {
                "output": str(
                    run(
                        output=args.output,
                        markdown=args.markdown,
                        cases_path=args.cases,
                        protocol_path=args.protocol,
                        policy_path=args.policy,
                        workers=args.workers,
                        test_mode=args.test_mode,
                    ).resolve()
                )
            }
        )
    )


if __name__ == "__main__":
    main()
