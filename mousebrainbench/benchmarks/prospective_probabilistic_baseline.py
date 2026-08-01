"""Fit the locked probabilistic comparator on development regimes only.

The comparator is intentionally conventional: one L2-regularized logistic
regression is fitted for each declared claim. Its inputs are the categorical
statuses of every evidence block. Claims with a constant development label use
that empirical constant instead of an ill-posed classifier. The resulting
artifact is immutable input to the prospective benchmark, not a model-selection
surface.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import yaml
from scipy import optimize

from mousebrainbench import __version__
from mousebrainbench.artifacts import code_revision
from mousebrainbench.benchmarks.oracle_sem_claim_benchmark import (
    REGIMES,
    _build_blocks,
    _oracle_claims,
)
from mousebrainbench.knowledge import load_default_profile
from mousebrainbench.validation.evidence_contract import EvidenceBlock, EvidenceStatus, blocks_by_name


DEFAULT_PROTOCOL = Path("configs/benchmarks/prospective_claim_validation_v1.yaml")
DEFAULT_OUTPUT = Path("results/prospective_probabilistic_baseline/model.json")
DEVELOPMENT_SEEDS = 100
DEVELOPMENT_N_PER_COHORT = 600


def _protocol_hash(path: Path) -> str:
    """Hash the exact protocol bytes so semantic YAML reformatting is visible."""

    return f"sha256:{hashlib.sha256(path.read_bytes()).hexdigest()}"


def _load_protocol(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text())
    if payload.get("status") != "frozen_before_execution":
        raise ValueError("prospective protocol is not frozen")
    return payload


def _feature_schema() -> tuple[tuple[str, ...], tuple[str, ...]]:
    profile = load_default_profile()
    blocks = tuple(
        sorted(
            {
                block
                for requirement in profile.requirements
                for block in requirement.required_blocks
            }
        )
    )
    statuses = tuple(status.value for status in EvidenceStatus)
    return blocks, statuses


def encode_blocks(blocks: Mapping[str, EvidenceBlock]) -> np.ndarray:
    """One-hot encode all block states plus an explicit missing indicator."""

    block_names, statuses = _feature_schema()
    encoded: list[float] = []
    for name in block_names:
        observed = blocks.get(name)
        encoded.extend(
            float(observed is not None and observed.status.value == status)
            for status in statuses
        )
        encoded.append(float(observed is None))
    return np.asarray(encoded, dtype=float)


def development_records(
    *,
    seeds: int = DEVELOPMENT_SEEDS,
    n_per_cohort: int = DEVELOPMENT_N_PER_COHORT,
) -> list[dict[str, Any]]:
    """Recreate the locked development partition and its DGP-derived labels."""

    records: list[dict[str, Any]] = []
    for regime_index, regime in enumerate(REGIMES):
        for seed in range(seeds):
            case_seed = 100_000 * regime_index + seed
            blocks = blocks_by_name(_build_blocks(regime, case_seed, n_per_cohort))
            records.append(
                {
                    "regime": regime,
                    "case_seed": case_seed,
                    "features": encode_blocks(blocks),
                    "reference_claims": _oracle_claims(regime),
                }
            )
    return records


def _fit_logistic(
    features: np.ndarray,
    labels: np.ndarray,
    *,
    regularization: float,
) -> tuple[np.ndarray, float, bool]:
    """Fit a deterministic L2 logistic model and return weights and intercept."""

    if np.all(labels == labels[0]):
        return np.zeros(features.shape[1], dtype=float), float(labels[0]), True

    design = np.column_stack((np.ones(len(features)), features))

    def objective(parameters: np.ndarray) -> tuple[float, np.ndarray]:
        scores = design @ parameters
        loss = float(np.logaddexp(0.0, scores).sum() - labels @ scores)
        penalty = 0.5 * regularization * float(parameters[1:] @ parameters[1:])
        probabilities = 1.0 / (1.0 + np.exp(-np.clip(scores, -40.0, 40.0)))
        gradient = design.T @ (probabilities - labels)
        gradient[1:] += regularization * parameters[1:]
        return loss + penalty, gradient

    initial = np.zeros(design.shape[1], dtype=float)
    initial[0] = float(np.log((labels.mean() + 1e-6) / (1.0 - labels.mean() + 1e-6)))
    result = optimize.minimize(
        objective,
        initial,
        method="L-BFGS-B",
        jac=True,
        options={"maxiter": 2_000, "ftol": 1e-12, "gtol": 1e-8},
    )
    if not result.success:
        raise RuntimeError(f"development comparator optimization failed: {result.message}")
    return result.x[1:], float(result.x[0]), False


def train(
    *,
    output: Path = DEFAULT_OUTPUT,
    protocol_path: Path = DEFAULT_PROTOCOL,
    seeds: int = DEVELOPMENT_SEEDS,
    n_per_cohort: int = DEVELOPMENT_N_PER_COHORT,
) -> Path:
    """Fit and persist the development-only probabilistic comparator."""

    protocol = _load_protocol(protocol_path)
    declared_regimes = tuple(protocol["development_partition"]["regimes"])
    if declared_regimes != REGIMES:
        raise ValueError("protocol development regimes differ from implemented regimes")
    settings = protocol["probabilistic_comparator"]
    regularization = float(settings["regularization_strength"])
    records = development_records(seeds=seeds, n_per_cohort=n_per_cohort)
    features = np.vstack([record["features"] for record in records])
    profile = load_default_profile()
    models: dict[str, Any] = {}
    for requirement in profile.requirements:
        labels = np.asarray(
            [requirement.claim in record["reference_claims"] for record in records],
            dtype=float,
        )
        weights, intercept, constant = _fit_logistic(
            features,
            labels,
            regularization=regularization,
        )
        models[requirement.claim] = {
            "intercept": intercept,
            "weights": weights.tolist(),
            "constant_label_model": constant,
            "development_prevalence": float(labels.mean()),
        }

    block_names, statuses = _feature_schema()
    payload = {
        "version": __version__,
        "git_revision": code_revision(),
        "analysis": "development_trained_probabilistic_claim_comparator",
        "protocol_id": protocol["protocol_id"],
        "protocol_version": str(protocol["version"]),
        "protocol_hash": _protocol_hash(protocol_path),
        "profile_id": profile.profile_id,
        "profile_version": profile.version,
        "profile_hash": profile.source_hash,
        "training_partition": "development_partition_only",
        "training_regimes": list(REGIMES),
        "training_cases": len(records),
        "seeds_per_regime": seeds,
        "n_per_cohort": n_per_cohort,
        "regularization_strength": regularization,
        "authorization_threshold": float(settings["authorization_threshold"]),
        "feature_schema": {
            "blocks": list(block_names),
            "statuses": list(statuses),
            "per_block_missing_indicator": True,
            "claim_identity": "encoded by one separately fitted model per claim",
            "dimension": int(features.shape[1]),
        },
        "models": models,
        "prospective_data_used_for_training": False,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2))
    return output


def load_frozen_model(
    path: Path,
    *,
    protocol_path: Path = DEFAULT_PROTOCOL,
) -> dict[str, Any]:
    """Load a comparator only when protocol and knowledge-profile identities match."""

    payload = json.loads(path.read_text())
    profile = load_default_profile()
    expected = {
        "protocol_hash": _protocol_hash(protocol_path),
        "profile_id": profile.profile_id,
        "profile_version": profile.version,
        "profile_hash": profile.source_hash,
        "training_partition": "development_partition_only",
        "prospective_data_used_for_training": False,
    }
    for key, value in expected.items():
        if payload.get(key) != value:
            raise ValueError(f"frozen comparator has incompatible {key}")
    return payload


def predict_claims(model: Mapping[str, Any], features: np.ndarray) -> set[str]:
    """Authorize claims using frozen coefficients and the prespecified threshold."""

    threshold = float(model["authorization_threshold"])
    predictions: set[str] = set()
    for claim, row in model["models"].items():
        if row["constant_label_model"]:
            probability = float(row["intercept"])
        else:
            score = float(row["intercept"] + np.asarray(row["weights"]) @ features)
            probability = 1.0 / (1.0 + np.exp(-np.clip(score, -40.0, 40.0)))
        if probability >= threshold:
            predictions.add(claim)
    return predictions


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--seeds", type=int, default=DEVELOPMENT_SEEDS)
    parser.add_argument("--n-per-cohort", type=int, default=DEVELOPMENT_N_PER_COHORT)
    args = parser.parse_args()
    print(
        json.dumps(
            {
                "output": str(
                    train(
                        output=args.output,
                        protocol_path=args.protocol,
                        seeds=args.seeds,
                        n_per_cohort=args.n_per_cohort,
                    ).resolve()
                )
            }
        )
    )


if __name__ == "__main__":
    main()

