"""Evaluate frozen v3 claim policies on the official Causal Chambers data.

Direct edges come from the package's published physical ground-truth graphs.
Matched non-edges are deterministic controls. The analysis is split at the
experiment level and is explicitly a transport audit: pair rows share variables
and observations, so no synthetic LTT guarantee is claimed for this domain.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import warnings
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from mousebrainbench import __version__
from mousebrainbench.artifacts import code_revision
from mousebrainbench.benchmarks.hybrid_development_features import encode_hybrid_features
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

DEFAULT_ROOT = Path("data/external/causal_chambers")
DEFAULT_SCORE_MODEL = Path("results/hybrid_selective_policy/model.json")
DEFAULT_RISK_POLICY = Path("results/semantic_risk_policy/model.json")
DEFAULT_OUTPUT = Path("results/causal_chambers_transport/summary.json")
DEFAULT_MARKDOWN = Path("results/causal_chambers_transport/summary.md")


def _sha256(path: Path) -> str:
    return f"sha256:{hashlib.sha256(path.read_bytes()).hexdigest()}"


def experiment_partition(chamber: str, experiment: str) -> str:
    """Apply the frozen SHA-256 modulo split without Python hash randomness."""

    value = int(hashlib.sha256(f"{chamber}:{experiment}".encode()).hexdigest(), 16) % 5
    return "calibration_context" if value in {0, 1, 2} else "locked_test"


def _ground_truth(chamber: str) -> tuple[tuple[str, ...], set[tuple[str, str]]]:
    from causalchamber.ground_truth.main import edges, variables

    return (
        tuple(variables(chamber, "standard")),
        set(edges(chamber, "standard")),
    )


def _eligible_columns(frame: pd.DataFrame, variables: Iterable[str]) -> tuple[str, ...]:
    output = []
    for name in variables:
        if name not in frame:
            continue
        values = pd.to_numeric(frame[name], errors="coerce").to_numpy(dtype=float)
        finite = values[np.isfinite(values)]
        if len(finite) >= 100 and np.std(finite) > 1e-8:
            output.append(name)
    return tuple(output)


def direct_and_control_pairs(
    columns: tuple[str, ...],
    edges: set[tuple[str, str]],
    *,
    namespace: str,
) -> tuple[tuple[str, str, bool], ...]:
    """Return every observable direct edge and an equal deterministic non-edge set."""

    direct = sorted((source, target) for source, target in edges if source in columns and target in columns)
    forbidden = edges | {(target, source) for source, target in edges}
    controls = sorted(
        (source, target)
        for source in columns
        for target in columns
        if source != target and (source, target) not in forbidden
    )
    controls = sorted(
        controls,
        key=lambda pair: hashlib.sha256(
            f"{namespace}:{pair[0]}:{pair[1]}".encode()
        ).hexdigest(),
    )[: len(direct)]
    return tuple(
        [(source, target, True) for source, target in direct]
        + [(source, target, False) for source, target in controls]
    )


def _cohort(
    frame: pd.DataFrame,
    source: str,
    target: str,
    controls: tuple[str, ...],
    remainder: int,
) -> GeneratedCohort:
    names = (source, target, *controls)
    values = frame.loc[np.arange(len(frame)) % 5 == remainder, list(names)].apply(
        pd.to_numeric, errors="coerce"
    )
    values = values.replace([np.inf, -np.inf], np.nan).dropna()
    if len(values) > 1000:
        values = values.iloc[np.linspace(0, len(values) - 1, 1000, dtype=int)]
    return GeneratedCohort(
        x=values[source].to_numpy(dtype=float),
        y=values[target].to_numpy(dtype=float),
        controls=values.loc[:, list(controls)].to_numpy(dtype=float),
    )


def _reference_prediction(
    frame: pd.DataFrame,
    source: str,
    target: str,
    controls: tuple[str, ...],
) -> bool:
    fourth = _cohort(frame, source, target, controls, 3)
    fifth = _cohort(frame, source, target, controls, 4)
    if min(len(fourth.x), len(fifth.x)) < 20:
        return False
    return bool(
        _safe_prediction(fourth, fifth)["passed"]
        and _safe_prediction(fifth, fourth)["passed"]
    )


def _safe_prediction(train: GeneratedCohort, test: GeneratedCohort) -> dict[str, Any]:
    """Convert near-constant physical partitions into explicit failed evidence."""

    if min(np.std(train.x), np.std(train.y), np.std(test.x), np.std(test.y)) < 1e-10:
        return {
            "passed": False,
            "correlation": 0.0,
            "p_value": 1.0,
            "r_squared": 0.0,
            "warning_types": ["near_constant_partition"],
        }
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = _prediction_diagnostic(train, test)
    if caught:
        result["passed"] = False
        result["warning_types"] = sorted(
            {type(item.message).__name__ for item in caught}
        )
    return result


def _pair_record(
    frame: pd.DataFrame,
    *,
    dataset: str,
    chamber: str,
    experiment: str,
    source: str,
    target: str,
    direct_edge: bool,
    eligible: tuple[str, ...],
    claim_names: tuple[str, ...],
) -> dict[str, Any] | None:
    control_candidates = [name for name in eligible if name not in {source, target}]
    controls = tuple(
        sorted(
            control_candidates,
            key=lambda name: hashlib.sha256(
                f"{dataset}:{experiment}:{source}:{target}:{name}".encode()
            ).hexdigest(),
        )[:3]
    )
    if len(controls) < 3:
        return None
    train = _cohort(frame, source, target, controls, 0)
    test = _cohort(frame, source, target, controls, 1)
    reproduction = _cohort(frame, source, target, controls, 2)
    if min(len(train.x), len(test.x), len(reproduction.x)) < 20:
        return None
    first = _safe_prediction(train, test)
    second = _safe_prediction(test, reproduction)
    topology = _topology_diagnostic(train, test)
    legacy = _direction_diagnostic(test)
    reference_predictive = _reference_prediction(frame, source, target, controls)
    direction_observations = {
        "method": "abstain",
        "attempted": False,
        "predicted_direction": "uncertain",
        "status": "requires_review",
        "blockers": [
            "pairwise_hidden_confounding_not_excluded",
            "experiment_specific_intervention_target_not_declared",
        ],
        "p_forward": None,
        "p_backward": None,
        "signed_margin": 0.0,
        "absolute_margin": 0.0,
    }
    blocks = blocks_by_name(
        (
            _evidence_block(
                "prediction",
                EvidenceStatus.PASSED if first["passed"] else EvidenceStatus.FAILED,
                "held-out correlation > 0.10, p < 0.01, and R-squared > 0",
                first,
            ),
            _evidence_block(
                "reproducible_compute",
                EvidenceStatus.PASSED,
                "official checksum-verified Causal Chambers CSV and deterministic pair",
                {"dataset": dataset, "experiment": experiment},
            ),
            _evidence_block(
                "internal_reproduction",
                EvidenceStatus.PASSED
                if first["passed"] and second["passed"]
                else EvidenceStatus.FAILED,
                "predictive rule passes in two disjoint row partitions",
                {"first": first, "second": second},
            ),
            _evidence_block(
                "external_replication",
                EvidenceStatus.NOT_APPLICABLE,
                "one physical chamber experiment is not cross-laboratory replication",
                {},
            ),
            _evidence_block(
                "topology_specificity",
                EvidenceStatus.PASSED if topology["passed"] else EvidenceStatus.FAILED,
                "candidate predictor exceeds three deterministic controls by 0.02",
                topology,
            ),
            EvidenceBlock.from_mapping(
                name="directed_identifiability",
                status=EvidenceStatus.REQUIRES_REVIEW,
                source="Causal Chambers pair adapter",
                rule="direction requires a valid assumption route or target-specific intervention",
                rationale="the published graph is reference truth, not input evidence",
                observations=direction_observations,
            ),
            _evidence_block(
                "structure_function_association",
                EvidenceStatus.NOT_APPLICABLE,
                "physical sensors are outside the mouse structure-function claim",
                {},
            ),
            _evidence_block(
                "causal_intervention",
                EvidenceStatus.REQUIRES_REVIEW,
                "the generic intervention flag does not identify a pair-specific randomized arm",
                {},
            ),
            _evidence_block(
                "whole_brain_coverage",
                EvidenceStatus.NOT_APPLICABLE,
                "physical chamber data do not represent a brain",
                {},
            ),
            _evidence_block(
                "independent_validation",
                EvidenceStatus.PASSED,
                "external physical resource not used to fit the frozen score model",
                {},
            ),
            _evidence_block(
                "entity_specificity",
                EvidenceStatus.NOT_APPLICABLE,
                "physical pair is not a biological digital twin entity",
                {},
            ),
            _evidence_block(
                "operational_compute",
                EvidenceStatus.NOT_APPLICABLE,
                "no digital-twin runtime target is declared",
                {},
            ),
        )
    )
    true_claims = {"computationally_reproducible"}
    if reference_predictive:
        true_claims |= {"predictive", "internally_reproduced"}
    if direct_edge:
        true_claims |= {"topology_specific", "directed", "causal"}
        if reference_predictive:
            true_claims.add("mechanistic")
    return {
        "dataset": dataset,
        "chamber": chamber,
        "experiment": experiment,
        "partition": experiment_partition(chamber, experiment),
        "source": source,
        "target": target,
        "direct_edge": direct_edge,
        "features": encode_hybrid_features(
            blocks,
            legacy_direction=legacy,
            anm=direction_observations,
            sample_size=min(len(train.x), len(test.x), len(reproduction.x)),
            noise_scale=0.0,
        ),
        "labels": np.asarray([claim in true_claims for claim in claim_names], dtype=np.uint8),
    }


def _evaluate_partition(
    records: list[dict[str, Any]],
    *,
    partition: str,
    score_model: dict[str, Any],
    risk_policy: SemanticRiskPolicy,
    variable_claims: tuple[str, ...],
) -> dict[str, Any]:
    selected = [row for row in records if row["partition"] == partition]
    if not selected:
        return {"partition": partition, "cases": 0, "decision": "empty_partition"}
    claim_names = tuple(score_model["claim_names"])
    variable_indices = np.asarray([claim_names.index(claim) for claim in variable_claims])
    features = np.vstack([row["features"] for row in selected])
    labels = np.vstack([row["labels"] for row in selected]).astype(bool)
    probabilities = predict_probabilities(
        score_model["model_sets"]["full"], features, claim_names
    )
    requirements = {
        item.claim: item.required_blocks for item in load_default_profile().requirements
    }
    admissible = semantic_admissibility_matrix(
        features,
        claim_names=claim_names,
        feature_names=score_model["feature_names"],
        support_vetoes=requirements,
    )
    decisions = authorize_with_policy(
        risk_policy,
        probabilities[:, variable_indices],
        admissible[:, variable_indices],
    )
    truths = labels[:, variable_indices]
    metrics = semantic_false_authorization_metrics(decisions, truths)
    return {
        "partition": partition,
        **metrics,
        "semantic_support_violations": int(
            ((decisions == 1) & ~admissible[:, variable_indices]).sum()
        ),
        "experiments": sorted(
            {f"{row['dataset']}/{row['experiment']}" for row in selected}
        ),
        "direct_edge_cases": sum(bool(row["direct_edge"]) for row in selected),
        "guarantee_transport_valid": False,
        "guarantee_limitation": (
            "the frozen synthetic LTT certificate does not cover physical-domain shift, "
            "and edge pairs are dependent within experiments"
        ),
    }


def run(
    *,
    root: Path = DEFAULT_ROOT,
    score_model_path: Path = DEFAULT_SCORE_MODEL,
    risk_policy_path: Path = DEFAULT_RISK_POLICY,
    output: Path = DEFAULT_OUTPUT,
    markdown: Path = DEFAULT_MARKDOWN,
) -> Path:
    """Build physical pair cases and evaluate frozen support thresholds."""

    score_model = json.loads(score_model_path.read_text())
    frozen = json.loads(risk_policy_path.read_text())
    risk_policy = SemanticRiskPolicy.from_dict(frozen["semantic_policy"])
    variable_claims = tuple(frozen["variable_claims"])
    claim_names = tuple(score_model["claim_names"])
    records: list[dict[str, Any]] = []
    dataset_hashes: dict[str, str] = {}
    for dataset, chamber in (("lt_test_v1", "lt"), ("wt_test_v1", "wt")):
        dataset_root = root / dataset
        if not dataset_root.exists():
            raise FileNotFoundError(f"missing official Causal Chambers dataset: {dataset_root}")
        variables, edges = _ground_truth(chamber)
        for csv_path in sorted(dataset_root.glob("*.csv")):
            dataset_hashes[str(csv_path)] = _sha256(csv_path)
            frame = pd.read_csv(csv_path)
            eligible = _eligible_columns(frame, variables)
            pairs = direct_and_control_pairs(
                eligible, edges, namespace=f"{dataset}:{csv_path.stem}"
            )
            for source, target, direct_edge in pairs:
                record = _pair_record(
                    frame,
                    dataset=dataset,
                    chamber=chamber,
                    experiment=csv_path.stem,
                    source=source,
                    target=target,
                    direct_edge=direct_edge,
                    eligible=eligible,
                    claim_names=claim_names,
                )
                if record is not None:
                    records.append(record)
    by_partition = [
        _evaluate_partition(
            records,
            partition=partition,
            score_model=score_model,
            risk_policy=risk_policy,
            variable_claims=variable_claims,
        )
        for partition in ("calibration_context", "locked_test")
    ]
    test = next(row for row in by_partition if row["partition"] == "locked_test")
    payload = {
        "version": __version__,
        "git_revision": code_revision(),
        "analysis": "causal_chambers_physical_transport_audit_v3",
        "source": "official causalchamber package and published ground-truth graphs",
        "causalchamber_version": importlib.metadata.version("causalchamber"),
        "license": "CC BY 4.0 data; MIT package",
        "datasets": ["lt_test_v1", "wt_test_v1"],
        "dataset_file_hashes": dataset_hashes,
        "frozen_score_model_hash": _sha256(score_model_path),
        "frozen_risk_policy_hash": _sha256(risk_policy_path),
        "score_model_refitted": False,
        "risk_policy_recalibrated": False,
        "pairs": len(records),
        "experiments": len({(row["dataset"], row["experiment"]) for row in records}),
        "variable_claims": list(variable_claims),
        "by_partition": by_partition,
        "locked_test_zero_semantic_violations": test.get("semantic_support_violations") == 0,
        "causal_or_directional_claims_from_pair_association_allowed": False,
        "decision": (
            "physical_transport_completed_without_semantic_violation"
            if test.get("cases", 0) > 0 and test.get("semantic_support_violations") == 0
            else "physical_transport_not_usable"
        ),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n")
    _write_markdown(payload, markdown)
    return output


def _write_markdown(payload: dict[str, Any], path: Path) -> None:
    lines = [
        "# Causal Chambers physical transport audit",
        "",
        f"- Decision: `{payload['decision']}`",
        f"- Pair cases: `{payload['pairs']}`",
        f"- Experiments: `{payload['experiments']}`",
        "- Synthetic guarantee transported: `false`",
        "",
        "| Partition | Cases | Coverage | SFAR | Semantic violations |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in payload["by_partition"]:
        lines.append(
            f"| `{row['partition']}` | {row['cases']} | "
            f"{row.get('supported_coverage', 0.0):.4f} | "
            f"{row.get('semantic_false_authorization_risk', 0.0):.4f} | "
            f"{row.get('semantic_support_violations', 0)} |"
        )
    lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--score-model", type=Path, default=DEFAULT_SCORE_MODEL)
    parser.add_argument("--risk-policy", type=Path, default=DEFAULT_RISK_POLICY)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--markdown", type=Path, default=DEFAULT_MARKDOWN)
    args = parser.parse_args()
    path = run(
        root=args.root,
        score_model_path=args.score_model,
        risk_policy_path=args.risk_policy,
        output=args.output,
        markdown=args.markdown,
    )
    print(json.dumps({"output": str(path.resolve())}))


if __name__ == "__main__":
    main()
