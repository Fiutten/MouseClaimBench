"""Run the frozen v3 synthetic confirmation without refitting or recalibration.

Reference labels are properties of sixteen new structural-equation regimes.
The frozen v2 score model and the LTT thresholds committed before this module is
executed are loaded read-only. The assumption router sees declared data-generating
facts but never the claim label matrix.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from mousebrainbench import __version__
from mousebrainbench.artifacts import code_revision
from mousebrainbench.benchmarks.hybrid_development_features import (
    encode_hybrid_features,
)
from mousebrainbench.benchmarks.hybrid_selective_policy import (
    predict_probabilities,
    semantic_admissibility_matrix,
)
from mousebrainbench.benchmarks.oracle_sem_claim_benchmark import (
    GeneratedCohort,
    _direction_diagnostic,
    _evidence_block,
    _prediction_diagnostic,
    _topology_diagnostic,
)
from mousebrainbench.knowledge import load_default_profile
from mousebrainbench.validation.direction_router import (
    DirectionAssumptions,
    route_direction,
)
from mousebrainbench.validation.evidence_contract import (
    EvidenceBlock,
    EvidenceStatus,
    blocks_by_name,
)
from mousebrainbench.validation.semantic_risk_control import (
    SemanticRiskPolicy,
    authorize_with_policy,
    semantic_false_authorization_metrics,
)

DEFAULT_PROTOCOL = Path("configs/benchmarks/semantic_risk_control_v3.yaml")
DEFAULT_SCORE_MODEL = Path("results/hybrid_selective_policy/model.json")
DEFAULT_RISK_POLICY = Path("results/semantic_risk_policy/model.json")
DEFAULT_OUTPUT = Path("results/semantic_risk_confirmation/summary.json")
DEFAULT_MARKDOWN = Path("results/semantic_risk_confirmation/summary.md")
DEFAULT_CASES = Path("results/semantic_risk_confirmation/cases.npz")

REGIMES = (
    "independent_heavy_tailed",
    "confounded_sine",
    "confounded_piecewise",
    "reverse_quadratic_additive",
    "direct_sigmoid_additive",
    "direct_cubic_additive",
    "direct_linear_laplace",
    "direct_linear_uniform",
    "direct_interventional_quadratic",
    "direct_interventional_sigmoid",
    "direct_post_nonlinear",
    "direct_heteroscedastic",
    "measurement_error_direct",
    "collider_selection",
    "confounded_direct",
    "direct_multiplicative_noise",
)

PREDICTIVE_ONLY = frozenset(
    {"predictive", "computationally_reproducible", "internally_reproduced"}
)
DIRECTED_STANDARD = PREDICTIVE_ONLY | {"topology_specific", "directed", "mechanistic"}
TRUTHS: dict[str, frozenset[str]] = {
    "independent_heavy_tailed": frozenset({"computationally_reproducible"}),
    "confounded_sine": PREDICTIVE_ONLY,
    "confounded_piecewise": PREDICTIVE_ONLY,
    "reverse_quadratic_additive": DIRECTED_STANDARD,
    "direct_sigmoid_additive": DIRECTED_STANDARD,
    "direct_cubic_additive": DIRECTED_STANDARD,
    "direct_linear_laplace": DIRECTED_STANDARD,
    "direct_linear_uniform": DIRECTED_STANDARD,
    "direct_interventional_quadratic": DIRECTED_STANDARD | {"causal"},
    "direct_interventional_sigmoid": DIRECTED_STANDARD | {"causal"},
    "direct_post_nonlinear": DIRECTED_STANDARD,
    "direct_heteroscedastic": DIRECTED_STANDARD,
    "measurement_error_direct": PREDICTIVE_ONLY | {"topology_specific"},
    "collider_selection": PREDICTIVE_ONLY,
    "confounded_direct": PREDICTIVE_ONLY | {"topology_specific"},
    "direct_multiplicative_noise": DIRECTED_STANDARD,
}

EXPECTED_DIRECTION = {
    regime: (
        None
        if regime
        in {
            "independent_heavy_tailed",
            "confounded_sine",
            "confounded_piecewise",
            "collider_selection",
            "confounded_direct",
        }
        else "reverse"
        if regime == "reverse_quadratic_additive"
        else "forward"
    )
    for regime in REGIMES
}


@dataclass(frozen=True)
class ConfirmationCell:
    regime: str
    sample_size: int
    noise_scale: float


def _sha256(path: Path) -> str:
    return f"sha256:{hashlib.sha256(path.read_bytes()).hexdigest()}"


def _noise(rng: np.random.Generator, n: int, scale: float) -> np.ndarray:
    return rng.normal(0.0, scale, size=n)


def _generate_cohort(
    regime: str,
    n: int,
    noise_scale: float,
    rng: np.random.Generator,
) -> GeneratedCohort:
    nuisance = rng.normal(size=(n, 3))
    if regime == "independent_heavy_tailed":
        x = rng.standard_t(df=4, size=n)
        y = rng.laplace(size=n)
        controls = nuisance
    elif regime == "confounded_sine":
        z = rng.uniform(-2.5, 2.5, size=n)
        x = z + _noise(rng, n, noise_scale)
        y = np.sin(z) + _noise(rng, n, noise_scale)
        controls = np.column_stack((z, nuisance[:, :2]))
    elif regime == "confounded_piecewise":
        z = rng.normal(size=n)
        x = z + _noise(rng, n, noise_scale)
        y = np.where(z < 0.0, 0.3 * z, 1.4 * z) + _noise(rng, n, noise_scale)
        controls = np.column_stack((z, nuisance[:, :2]))
    elif regime == "reverse_quadratic_additive":
        y = rng.laplace(size=n)
        x = 0.65 * y + 0.18 * np.square(y) + _noise(rng, n, noise_scale)
        controls = nuisance
    elif regime in {"direct_sigmoid_additive", "direct_interventional_sigmoid"}:
        x = rng.uniform(-2.5, 2.5, size=n)
        y = 1.5 * np.tanh(0.9 * x) + _noise(rng, n, noise_scale)
        controls = nuisance
    elif regime in {"direct_cubic_additive", "direct_interventional_quadratic"}:
        x = rng.uniform(-2.0, 2.0, size=n)
        y = 0.55 * x + 0.12 * np.power(x, 3) + _noise(rng, n, noise_scale)
        controls = nuisance
    elif regime == "direct_linear_laplace":
        x = rng.laplace(size=n)
        y = 0.9 * x + noise_scale * rng.laplace(size=n)
        controls = nuisance
    elif regime == "direct_linear_uniform":
        x = rng.uniform(-2.0, 2.0, size=n)
        y = 0.9 * x + noise_scale * rng.uniform(-1.5, 1.5, size=n)
        controls = nuisance
    elif regime == "direct_post_nonlinear":
        x = rng.normal(size=n)
        y = 2.0 * np.tanh(0.85 * x + _noise(rng, n, noise_scale))
        controls = nuisance
    elif regime == "direct_heteroscedastic":
        x = rng.normal(size=n)
        y = 0.8 * x + (0.25 + 0.5 * np.abs(x)) * _noise(rng, n, noise_scale)
        controls = nuisance
    elif regime == "measurement_error_direct":
        latent = rng.normal(size=n)
        x = latent + _noise(rng, n, 0.55 * noise_scale)
        y = 0.9 * latent + _noise(rng, n, noise_scale)
        controls = nuisance
    elif regime == "collider_selection":
        available = n * 5
        candidate_x = rng.normal(size=available)
        candidate_y = rng.normal(size=available)
        collider = candidate_x + candidate_y + _noise(rng, available, noise_scale)
        chosen = np.flatnonzero(collider >= np.quantile(collider, 0.60))[:n]
        x = candidate_x[chosen]
        y = candidate_y[chosen]
        controls = np.column_stack((collider[chosen], rng.normal(size=(n, 2))))
    elif regime == "confounded_direct":
        z = rng.normal(size=n)
        x = 0.8 * z + _noise(rng, n, noise_scale)
        y = 0.55 * x + 0.9 * z + _noise(rng, n, noise_scale)
        controls = np.column_stack((z, nuisance[:, :2]))
    elif regime == "direct_multiplicative_noise":
        x = rng.uniform(-1.5, 1.5, size=n)
        scale = 0.35 + 0.25 * np.square(x)
        y = 0.85 * x + scale * _noise(rng, n, noise_scale)
        controls = nuisance
    else:
        raise ValueError(f"unknown v3 SEM regime: {regime}")
    return GeneratedCohort(
        x=np.asarray(x, dtype=float),
        y=np.asarray(y, dtype=float),
        controls=np.asarray(controls, dtype=float),
    )


def _assumptions(regime: str) -> DirectionAssumptions:
    common = {
        "continuous": True,
        "acyclic": True,
        "selection_bias_excluded": regime != "collider_selection",
        "hidden_confounding_excluded": regime
        not in {"confounded_sine", "confounded_piecewise", "confounded_direct"},
        "material_measurement_error": regime == "measurement_error_direct",
        "provenance": f"frozen v3 structural equation: {regime}",
    }
    return DirectionAssumptions(
        randomized_intervention=regime.startswith("direct_interventional"),
        linear=regime in {"direct_linear_laplace", "direct_linear_uniform"},
        non_gaussian=regime in {"direct_linear_laplace", "direct_linear_uniform"},
        additive_noise=regime
        in {
            "independent_heavy_tailed",
            "reverse_quadratic_additive",
            "direct_sigmoid_additive",
            "direct_cubic_additive",
        },
        **common,
    )


def _intervention_arms(
    regime: str,
    n: int,
    noise_scale: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    if regime == "direct_interventional_quadratic":
        control = -0.55 - 0.12 + _noise(rng, n, noise_scale)
        treated = 0.55 + 0.12 + _noise(rng, n, noise_scale)
        return control, treated
    if regime == "direct_interventional_sigmoid":
        effect = 1.5 * np.tanh(0.9)
        return -effect + _noise(rng, n, noise_scale), effect + _noise(rng, n, noise_scale)
    return None, None


def _build_record(task: tuple[ConfirmationCell, int, tuple[str, ...]]) -> dict[str, Any]:
    cell, case_seed, claim_names = task
    children = np.random.SeedSequence(case_seed).spawn(5)
    train = _generate_cohort(
        cell.regime, cell.sample_size, cell.noise_scale, np.random.default_rng(children[0])
    )
    test = _generate_cohort(
        cell.regime, cell.sample_size, cell.noise_scale, np.random.default_rng(children[1])
    )
    reproduction = _generate_cohort(
        cell.regime, cell.sample_size, cell.noise_scale, np.random.default_rng(children[2])
    )
    control, treated = _intervention_arms(
        cell.regime,
        cell.sample_size,
        cell.noise_scale,
        np.random.default_rng(children[3]),
    )
    prediction_a = _prediction_diagnostic(train, test)
    prediction_b = _prediction_diagnostic(test, reproduction)
    topology = _topology_diagnostic(train, test)
    legacy_direction = _direction_diagnostic(test)
    routed = route_direction(
        test.x,
        test.y,
        _assumptions(cell.regime),
        seed=case_seed,
        intervention_control=control,
        intervention_treated=treated,
    )
    router_observations = {
        **routed,
        **{
            key: value
            for key, value in routed["evidence"].items()
            if key
            in {
                "p_forward",
                "p_backward",
                "signed_margin",
                "absolute_margin",
            }
        },
    }
    causal_observations = {
        "available": control is not None and treated is not None,
        "passed": routed["causal_support_allowed"],
        "mean_do_plus_minus_do_minus": routed["evidence"].get("mean_effect", 0.0),
        "p_value": routed["evidence"].get("p_value"),
    }
    blocks = blocks_by_name(
        (
            _evidence_block(
                "prediction",
                EvidenceStatus.PASSED if prediction_a["passed"] else EvidenceStatus.FAILED,
                "held-out correlation > 0.10, p < 0.01, and R-squared > 0",
                prediction_a,
            ),
            _evidence_block(
                "reproducible_compute",
                EvidenceStatus.PASSED,
                "case regenerated from frozen version-3 seed and equation",
                {"case_seed": case_seed, "regime": cell.regime},
            ),
            _evidence_block(
                "internal_reproduction",
                EvidenceStatus.PASSED
                if prediction_a["passed"] and prediction_b["passed"]
                else EvidenceStatus.FAILED,
                "predictive rule passes in two non-overlapping generated cohorts",
                {"first": prediction_a, "second": prediction_b},
            ),
            _evidence_block(
                "external_replication",
                EvidenceStatus.NOT_APPLICABLE,
                "synthetic cohorts are not external empirical replication",
                {},
            ),
            _evidence_block(
                "topology_specificity",
                EvidenceStatus.PASSED if topology["passed"] else EvidenceStatus.FAILED,
                "candidate held-out R-squared exceeds every control by 0.02",
                topology,
            ),
            EvidenceBlock.from_mapping(
                name="directed_identifiability",
                status=EvidenceStatus(routed["status"]),
                source=routed["method"],
                rule="assumption-aware frozen route; confident forward or reverse passes",
                rationale="invalid or unmatched assumptions force review",
                observations=router_observations,
            ),
            _evidence_block(
                "structure_function_association",
                EvidenceStatus.NOT_APPLICABLE,
                "scalar SEMs are not a structure-function resource",
                {},
            ),
            _evidence_block(
                "causal_intervention",
                EvidenceStatus.PASSED
                if routed["causal_support_allowed"]
                else EvidenceStatus.FAILED
                if control is not None
                else EvidenceStatus.NOT_APPLICABLE,
                "controlled intervention contrast at p < 0.01",
                causal_observations,
            ),
            _evidence_block(
                "whole_brain_coverage",
                EvidenceStatus.FAILED,
                "a scalar SEM has no whole-brain coverage",
                {},
            ),
            _evidence_block(
                "independent_validation",
                EvidenceStatus.FAILED,
                "generated cases are not independent biological validation",
                {},
            ),
            _evidence_block(
                "entity_specificity",
                EvidenceStatus.FAILED,
                "a generated regime is not one biological entity",
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
    return {
        "regime": cell.regime,
        "sample_size": cell.sample_size,
        "noise_scale": cell.noise_scale,
        "case_seed": case_seed,
        "features": encode_hybrid_features(
            blocks,
            legacy_direction=legacy_direction,
            anm=router_observations,
            sample_size=cell.sample_size,
            noise_scale=cell.noise_scale,
        ),
        "labels": np.asarray(
            [claim in TRUTHS[cell.regime] for claim in claim_names], dtype=np.uint8
        ),
        "route_method": routed["method"],
        "route_attempted": routed["attempted"],
        "route_direction": routed["predicted_direction"],
        "route_blockers": "|".join(routed["blockers"]),
    }


def _claim_rows(
    decisions: np.ndarray,
    labels: np.ndarray,
    claim_names: tuple[str, ...],
) -> list[dict[str, Any]]:
    rows = []
    for index, claim in enumerate(claim_names):
        metrics = semantic_false_authorization_metrics(
            decisions[:, [index]], labels[:, [index]]
        )
        rows.append({"claim": claim, **metrics})
    return rows


def _direction_metrics(
    regimes: np.ndarray,
    attempted: np.ndarray,
    predicted: np.ndarray,
) -> dict[str, Any]:
    expected = np.asarray([EXPECTED_DIRECTION[value] or "none" for value in regimes])
    correct = attempted & (predicted == expected)
    identifiable = expected != "none"
    return {
        "cases": len(regimes),
        "attempts": int(attempted.sum()),
        "coverage": float(attempted.mean()),
        "attempted_accuracy": float(correct.sum() / attempted.sum()) if attempted.any() else 0.0,
        "identifiable_regime_attempted_accuracy": (
            float((correct & identifiable).sum() / (attempted & identifiable).sum())
            if (attempted & identifiable).any()
            else 0.0
        ),
        "spurious_attempts_without_reference_direction": int(
            (attempted & ~identifiable).sum()
        ),
    }


def run(
    *,
    protocol_path: Path = DEFAULT_PROTOCOL,
    score_model_path: Path = DEFAULT_SCORE_MODEL,
    risk_policy_path: Path = DEFAULT_RISK_POLICY,
    output: Path = DEFAULT_OUTPUT,
    markdown: Path = DEFAULT_MARKDOWN,
    cases_path: Path = DEFAULT_CASES,
    workers: int = 1,
    test_mode: bool = False,
) -> Path:
    """Execute the frozen v3 cases and compare risk-control ablations."""

    started = time.perf_counter()
    protocol = yaml.safe_load(protocol_path.read_text())
    if tuple(REGIMES) != tuple(TRUTHS):
        raise RuntimeError("v3 regime ordering and truth map differ")
    score_model = json.loads(score_model_path.read_text())
    frozen = json.loads(risk_policy_path.read_text())
    if frozen["protocol_hash"] != _sha256(protocol_path):
        raise ValueError("risk policy was calibrated under another protocol")
    claim_names = tuple(str(value) for value in score_model["claim_names"])
    variable_claims = tuple(str(value) for value in frozen["variable_claims"])
    variable_indices = np.asarray([claim_names.index(claim) for claim in variable_claims])
    semantic_policy = SemanticRiskPolicy.from_dict(frozen["semantic_policy"])
    unconstrained_policy = SemanticRiskPolicy.from_dict(frozen["unconstrained_policy"])

    partition = protocol["fresh_confirmatory_blocks"]["synthetic_v3"]
    sample_sizes = (200,) if test_mode else tuple(partition["sample_sizes"])
    noise_scales = (1.0,) if test_mode else tuple(partition["noise_scales"])
    seeds_per_cell = 1 if test_mode else int(partition["seeds_per_cell"])
    cells = [
        ConfirmationCell(regime, int(sample_size), float(noise_scale))
        for regime in REGIMES
        for sample_size in sample_sizes
        for noise_scale in noise_scales
    ]
    namespace = int(partition["seed_namespace"])
    tasks = [
        (cell, namespace + cell_index * 10_000 + seed, claim_names)
        for cell_index, cell in enumerate(cells)
        for seed in range(seeds_per_cell)
    ]
    if workers > 1:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            records = list(executor.map(_build_record, tasks, chunksize=1))
    else:
        records = [_build_record(task) for task in tasks]

    features = np.vstack([row["features"] for row in records])
    labels = np.vstack([row["labels"] for row in records]).astype(bool)
    probabilities = predict_probabilities(
        score_model["model_sets"]["full"], features, claim_names
    )
    complete_requirements = {
        item.claim: item.required_blocks for item in load_default_profile().requirements
    }
    admissible = semantic_admissibility_matrix(
        features,
        claim_names=claim_names,
        feature_names=score_model["feature_names"],
        support_vetoes=complete_requirements,
    )
    scores = probabilities[:, variable_indices]
    truths = labels[:, variable_indices]
    gates = admissible[:, variable_indices]
    policies = {
        "naive_probability_threshold": (scores >= 0.5).astype(np.int8),
        "semantic_gate_without_risk_control": (gates & (scores >= 0.5)).astype(np.int8),
        "unconstrained_MAPIE_risk_control": authorize_with_policy(
            unconstrained_policy, scores, np.ones_like(gates, dtype=bool)
        ),
        "semantic_MAPIE_risk_control": authorize_with_policy(
            semantic_policy, scores, gates
        ),
        "evidence_contract_only": gates.astype(np.int8),
    }
    aggregate = {
        name: {
            "policy": name,
            **semantic_false_authorization_metrics(values, truths),
            "semantic_support_violations": int(((values == 1) & ~gates).sum()),
            "per_claim": _claim_rows(values, truths, variable_claims),
        }
        for name, values in policies.items()
    }
    primary = aggregate["semantic_MAPIE_risk_control"]
    all_certified = all(item.certified for item in semantic_policy.certificates)
    scale_matches = not test_mode and len(records) == int(partition["expected_cases"])
    conditions = {
        "scale_matches_frozen_protocol": scale_matches,
        "semantic_support_violations_equal_0": primary[
            "semantic_support_violations"
        ]
        == 0,
        "all_variable_claim_families_ltt_certified": all_certified,
        "synthetic_macro_supported_coverage_at_least_0.20": primary[
            "supported_coverage"
        ]
        >= 0.20,
        "synthetic_empirical_sfar_at_most_0.05": primary[
            "semantic_false_authorization_risk"
        ]
        <= 0.05,
    }
    regimes = np.asarray([row["regime"] for row in records])
    route_attempted = np.asarray([row["route_attempted"] for row in records], dtype=bool)
    route_direction = np.asarray([row["route_direction"] for row in records])
    cases_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cases_path,
        features=features,
        labels=labels.astype(np.uint8),
        probabilities=probabilities,
        admissible=admissible.astype(np.uint8),
        regimes=regimes,
        sample_sizes=np.asarray([row["sample_size"] for row in records]),
        noise_scales=np.asarray([row["noise_scale"] for row in records]),
        case_seeds=np.asarray([row["case_seed"] for row in records], dtype=np.int64),
        route_methods=np.asarray([row["route_method"] for row in records]),
        route_attempted=route_attempted.astype(np.uint8),
        route_directions=route_direction,
        policy_names=np.asarray(tuple(policies)),
        policy_decisions=np.stack([policies[name] for name in policies]),
    )
    payload = {
        "version": __version__,
        "git_revision": code_revision(),
        "analysis": "semantic_risk_control_synthetic_confirmation_v3",
        "protocol_id": protocol["protocol_id"],
        "protocol_version": str(protocol["version"]),
        "protocol_hash": _sha256(protocol_path),
        "frozen_score_model_hash": _sha256(score_model_path),
        "frozen_risk_policy_hash": _sha256(risk_policy_path),
        "score_model_refitted": False,
        "risk_policy_recalibrated": False,
        "test_mode": test_mode,
        "scale_matches_frozen_protocol": scale_matches,
        "cases": len(records),
        "variable_claims": list(variable_claims),
        "regimes": list(REGIMES),
        "sample_sizes": list(sample_sizes),
        "noise_scales": list(noise_scales),
        "seeds_per_cell": seeds_per_cell,
        "workers": workers,
        "case_artifact": str(cases_path),
        "case_artifact_sha256": _sha256(cases_path),
        "aggregate_by_policy": list(aggregate.values()),
        "direction_router": _direction_metrics(
            regimes, route_attempted, route_direction
        ),
        "primary_conditions": conditions,
        "computational_primary_passed": all(conditions.values()),
        "wall_time_seconds": time.perf_counter() - started,
        "decision": (
            "v3_synthetic_primary_passed"
            if all(conditions.values())
            else "v3_synthetic_primary_not_passed"
        ),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n")
    _write_markdown(payload, markdown)
    return output


def _write_markdown(payload: dict[str, Any], path: Path) -> None:
    lines = [
        "# Semantic risk control v3 synthetic confirmation",
        "",
        f"- Decision: `{payload['decision']}`",
        f"- Cases: `{payload['cases']}`",
        f"- Score refitted: `{payload['score_model_refitted']}`",
        f"- Risk policy recalibrated: `{payload['risk_policy_recalibrated']}`",
        "",
        "| Policy | Coverage | SFAR | False authorizations | Semantic violations |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in payload["aggregate_by_policy"]:
        lines.append(
            f"| `{row['policy']}` | {row['supported_coverage']:.4f} | "
            f"{row['semantic_false_authorization_risk']:.4f} | "
            f"{row['false_authorizations']} | {row['semantic_support_violations']} |"
        )
    lines.extend(("", "## Frozen primary conditions", ""))
    lines.extend(
        f"- `{key}`: `{value}`" for key, value in payload["primary_conditions"].items()
    )
    lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--score-model", type=Path, default=DEFAULT_SCORE_MODEL)
    parser.add_argument("--risk-policy", type=Path, default=DEFAULT_RISK_POLICY)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--markdown", type=Path, default=DEFAULT_MARKDOWN)
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--test-mode", action="store_true")
    args = parser.parse_args()
    path = run(
        protocol_path=args.protocol,
        score_model_path=args.score_model,
        risk_policy_path=args.risk_policy,
        output=args.output,
        markdown=args.markdown,
        cases_path=args.cases,
        workers=args.workers,
        test_mode=args.test_mode,
    )
    print(json.dumps({"output": str(path.resolve())}))


if __name__ == "__main__":
    main()
