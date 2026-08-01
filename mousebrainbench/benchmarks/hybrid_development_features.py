"""Build the consumed, leakage-controlled feature matrix for hybrid v2.

All rows come from regimes whose outcomes have already been inspected. The
artifact is therefore suitable for model fitting, calibration, and development
audit, but never for version-2 confirmation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Callable, Mapping

import numpy as np

from mousebrainbench import __version__
from mousebrainbench.artifacts import code_revision
from mousebrainbench.benchmarks.causal_direction_anm import (
    SUBSAMPLE_SEED_NAMESPACE,
    anm_direction_evidence,
)
from mousebrainbench.benchmarks.oracle_sem_claim_benchmark import (
    REGIMES as ORACLE_REGIMES,
    _build_blocks as _build_oracle_blocks,
    _generate_cohort as _generate_oracle_cohort,
    _oracle_claims,
)
from mousebrainbench.benchmarks.prospective_claim_validation import (
    REGIME_TRUTHS,
    ProspectiveCell,
    _build_blocks as _build_prospective_blocks,
    _generate_cohort as _generate_prospective_cohort,
)
from mousebrainbench.benchmarks.prospective_probabilistic_baseline import (
    _feature_schema,
    encode_blocks,
)
from mousebrainbench.validation.evidence_contract import (
    CLAIM_REQUIREMENTS_V3,
    EvidenceBlock,
    EvidenceStatus,
    blocks_by_name,
)


DEFAULT_OUTPUT = Path("results/hybrid_development_features/summary.json")
DEFAULT_MATRIX = Path("results/hybrid_development_features/cases.npz")

CONTINUOUS_FEATURES = (
    "prediction_correlation",
    "prediction_negative_log10_p",
    "prediction_r_squared",
    "reproduction_first_correlation",
    "reproduction_second_correlation",
    "reproduction_first_r_squared",
    "reproduction_second_r_squared",
    "topology_candidate_r_squared",
    "topology_best_control_margin",
    "legacy_direction_forward_dependence",
    "legacy_direction_reverse_dependence",
    "legacy_direction_margin",
    "anm_p_forward",
    "anm_p_backward",
    "anm_signed_margin",
    "anm_absolute_margin",
    "causal_intervention_available",
    "causal_intervention_effect",
    "causal_intervention_negative_log10_p",
    "log10_sample_size",
    "known_noise_scale",
)

V1_SAMPLE_SIZES = (150, 600, 2400)
V1_NOISE_SCALES = (0.6, 1.2, 1.8)
V1_SEEDS_PER_CELL = 40
V1_SEED_NAMESPACE = 2_026_080_100


def _negative_log10(value: Any) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0.0
    if not np.isfinite(number) or number <= 0.0:
        return 300.0 if number == 0.0 else 0.0
    return float(-np.log10(max(number, 1e-300)))


def _observations(block: EvidenceBlock | None) -> dict[str, Any]:
    return dict(block.observations) if block is not None else {}


def replace_direction_block(
    blocks: Mapping[str, EvidenceBlock],
    evidence: Mapping[str, Any],
) -> dict[str, EvidenceBlock]:
    """Replace only the directional block while preserving all other evidence."""

    status = EvidenceStatus(str(evidence["status"]))
    updated = dict(blocks)
    updated["directed_identifiability"] = EvidenceBlock.from_mapping(
        name="directed_identifiability",
        status=status,
        source="causal-learn==0.1.4.8 ANM",
        rule=(
            "forward-minus-backward independence p-value margin >= 0.10; "
            "reverse margin <= -0.10; otherwise requires review"
        ),
        rationale=(
            "ANM direction is assumption-conditional and ambiguous cases are not forced"
        ),
        observations=dict(evidence),
    )
    return updated


def hybrid_feature_names() -> tuple[str, ...]:
    """Return the immutable status and continuous feature names in matrix order."""

    block_names, statuses = _feature_schema()
    categorical = tuple(
        feature
        for block in block_names
        for feature in (
            *(f"status:{block}:{status}" for status in statuses),
            f"missing:{block}",
        )
    )
    return (*categorical, *CONTINUOUS_FEATURES)


def encode_hybrid_features(
    blocks: Mapping[str, EvidenceBlock],
    *,
    legacy_direction: Mapping[str, Any],
    anm: Mapping[str, Any],
    sample_size: int,
    noise_scale: float,
) -> np.ndarray:
    """Encode evidence states and unnormalized diagnostic measurements."""

    prediction = _observations(blocks.get("prediction"))
    reproduction = _observations(blocks.get("internal_reproduction"))
    first = reproduction.get("first", {})
    second = reproduction.get("second", {})
    topology = _observations(blocks.get("topology_specificity"))
    causal = _observations(blocks.get("causal_intervention"))
    continuous = np.asarray(
        [
            float(prediction.get("correlation", 0.0)),
            _negative_log10(prediction.get("p_value")),
            float(prediction.get("r_squared", 0.0)),
            float(first.get("correlation", 0.0)),
            float(second.get("correlation", 0.0)),
            float(first.get("r_squared", 0.0)),
            float(second.get("r_squared", 0.0)),
            float(topology.get("candidate_r_squared", 0.0)),
            float(topology.get("best_control_margin", 0.0)),
            float(legacy_direction.get("forward_residual_dependence", 0.0)),
            float(legacy_direction.get("reverse_residual_dependence", 0.0)),
            float(legacy_direction.get("direction_margin", 0.0)),
            float(anm.get("p_forward") or 0.0),
            float(anm.get("p_backward") or 0.0),
            float(anm.get("signed_margin", 0.0)),
            float(anm.get("absolute_margin", 0.0)),
            float(bool(causal.get("available", False))),
            float(causal.get("mean_do_plus_minus_do_minus", 0.0)),
            _negative_log10(causal.get("p_value")),
            float(np.log10(sample_size)),
            float(noise_scale),
        ],
        dtype=float,
    )
    values = np.concatenate((encode_blocks(blocks), continuous))
    if not np.all(np.isfinite(values)):
        raise ValueError("hybrid feature vector contains non-finite values")
    if len(values) != len(hybrid_feature_names()):
        raise RuntimeError("hybrid feature schema and values disagree")
    return values


def _split_role(case_seed: int) -> str:
    remainder = case_seed % 5
    return (
        "model_fit"
        if remainder in {0, 1, 2}
        else "threshold_calibration"
        if remainder == 3
        else "locked_development_audit"
    )


def _oracle_test_cohort(regime: str, case_seed: int, n: int):
    children = np.random.SeedSequence(case_seed).spawn(4)
    return _generate_oracle_cohort(regime, n, np.random.default_rng(children[1]))


def _prospective_test_cohort(cell: ProspectiveCell, case_seed: int):
    children = np.random.SeedSequence(case_seed).spawn(4)
    return _generate_prospective_cohort(
        cell.regime,
        cell.sample_size,
        cell.noise_scale,
        np.random.default_rng(children[1]),
        1,
    )


def _case_record(
    *,
    source_partition: str,
    regime: str,
    case_seed: int,
    sample_size: int,
    noise_scale: float,
    blocks_tuple: tuple[EvidenceBlock, ...],
    test_cohort: Any,
    reference_claims: set[str] | frozenset[str],
    direction_function: Callable[..., dict[str, Any]],
) -> dict[str, Any]:
    original = blocks_by_name(blocks_tuple)
    legacy_direction = _observations(original.get("directed_identifiability"))
    anm = direction_function(
        test_cohort.x,
        test_cohort.y,
        seed=SUBSAMPLE_SEED_NAMESPACE + case_seed,
    )
    updated = replace_direction_block(original, anm)
    return {
        "source_partition": source_partition,
        "regime": regime,
        "case_seed": case_seed,
        "sample_size": sample_size,
        "noise_scale": noise_scale,
        "split_role": _split_role(case_seed),
        "features": encode_hybrid_features(
            updated,
            legacy_direction=legacy_direction,
            anm=anm,
            sample_size=sample_size,
            noise_scale=noise_scale,
        ),
        "labels": np.asarray(
            [requirement.claim in reference_claims for requirement in CLAIM_REQUIREMENTS_V3],
            dtype=np.uint8,
        ),
        "anm_status": str(anm["status"]),
        "anm_direction": str(anm["predicted_direction"]),
        "anm_execution_error": anm["execution_error"],
    }


def development_records(
    *,
    test_mode: bool = False,
    direction_function: Callable[..., dict[str, Any]] = anm_direction_evidence,
) -> list[dict[str, Any]]:
    """Generate all consumed rows or a small structurally identical test subset."""

    records: list[dict[str, Any]] = []
    oracle_seeds = 2 if test_mode else 100
    for regime_index, regime in enumerate(ORACLE_REGIMES):
        for seed in range(oracle_seeds):
            case_seed = 100_000 * regime_index + seed
            records.append(
                _case_record(
                    source_partition="oracle_v0",
                    regime=regime,
                    case_seed=case_seed,
                    sample_size=600,
                    noise_scale=0.65,
                    blocks_tuple=_build_oracle_blocks(regime, case_seed, 600),
                    test_cohort=_oracle_test_cohort(regime, case_seed, 600),
                    reference_claims=_oracle_claims(regime),
                    direction_function=direction_function,
                )
            )

    sample_sizes = (150,) if test_mode else V1_SAMPLE_SIZES
    noise_scales = (1.2,) if test_mode else V1_NOISE_SCALES
    seeds_per_cell = 2 if test_mode else V1_SEEDS_PER_CELL
    cells = [
        ProspectiveCell(regime, sample_size, noise_scale)
        for regime in REGIME_TRUTHS
        for sample_size in sample_sizes
        for noise_scale in noise_scales
    ]
    for cell_index, cell in enumerate(cells):
        for seed in range(seeds_per_cell):
            case_seed = V1_SEED_NAMESPACE + cell_index * 10_000 + seed
            records.append(
                _case_record(
                    source_partition="prospective_v1_consumed",
                    regime=cell.regime,
                    case_seed=case_seed,
                    sample_size=cell.sample_size,
                    noise_scale=cell.noise_scale,
                    blocks_tuple=_build_prospective_blocks(cell, case_seed=case_seed),
                    test_cohort=_prospective_test_cohort(cell, case_seed),
                    reference_claims=REGIME_TRUTHS[cell.regime],
                    direction_function=direction_function,
                )
            )
    return records


def _sha256(path: Path) -> str:
    return f"sha256:{hashlib.sha256(path.read_bytes()).hexdigest()}"


def run(
    *,
    output: Path = DEFAULT_OUTPUT,
    matrix: Path = DEFAULT_MATRIX,
    test_mode: bool = False,
    direction_function: Callable[..., dict[str, Any]] = anm_direction_evidence,
) -> Path:
    """Persist the consumed feature matrix with partition and schema provenance."""

    records = development_records(
        test_mode=test_mode,
        direction_function=direction_function,
    )
    features = np.vstack([row["features"] for row in records]).astype(np.float64)
    labels = np.vstack([row["labels"] for row in records]).astype(np.uint8)
    matrix.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        matrix,
        features=features,
        labels=labels,
        source_partition=np.asarray([row["source_partition"] for row in records]),
        regime=np.asarray([row["regime"] for row in records]),
        case_seed=np.asarray([row["case_seed"] for row in records], dtype=np.int64),
        sample_size=np.asarray([row["sample_size"] for row in records], dtype=np.int32),
        noise_scale=np.asarray([row["noise_scale"] for row in records], dtype=float),
        split_role=np.asarray([row["split_role"] for row in records]),
        anm_status=np.asarray([row["anm_status"] for row in records]),
        anm_direction=np.asarray([row["anm_direction"] for row in records]),
    )
    split_counts = {
        role: sum(row["split_role"] == role for row in records)
        for role in ("model_fit", "threshold_calibration", "locked_development_audit")
    }
    direction_counts = {
        status: sum(row["anm_status"] == status for row in records)
        for status in ("passed", "failed", "requires_review")
    }
    claim_names = [requirement.claim for requirement in CLAIM_REQUIREMENTS_V3]
    payload = {
        "version": __version__,
        "git_revision": code_revision(),
        "analysis": "hybrid_v2_consumed_development_feature_matrix",
        "data_role": "consumed_development_only_not_confirmatory",
        "test_mode": test_mode,
        "matrix_file": str(matrix),
        "matrix_sha256": _sha256(matrix),
        "cases": len(records),
        "feature_count": features.shape[1],
        "feature_names": list(hybrid_feature_names()),
        "claim_names": claim_names,
        "split_counts": split_counts,
        "source_partition_counts": {
            source: sum(row["source_partition"] == source for row in records)
            for source in sorted({row["source_partition"] for row in records})
        },
        "direction_status_counts": direction_counts,
        "direction_execution_errors": sum(
            row["anm_execution_error"] is not None for row in records
        ),
        "confirmatory_v2_cases_used": 0,
        "decision": (
            "hybrid_development_features_frozen_without_v2_confirmation_leakage"
            if not test_mode and len(records) == 3740
            else "hybrid_development_features_test_artifact"
        ),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2))
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--matrix", type=Path, default=DEFAULT_MATRIX)
    parser.add_argument("--test-mode", action="store_true")
    args = parser.parse_args()
    print(
        json.dumps(
            {"output": str(run(output=args.output, matrix=args.matrix, test_mode=args.test_mode).resolve())}
        )
    )


if __name__ == "__main__":
    main()

