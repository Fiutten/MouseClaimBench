"""Post-confirmation sensitivity of semantic LTT certificates to calibration size.

This analysis is deliberately exploratory. It repeatedly recalibrates the LTT
policy on deterministic subsamples of the already consumed v2 development set
and evaluates each policy on the already inspected v3 synthetic set. It cannot
replace the frozen primary confirmation or support a new confirmatory claim.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from mousebrainbench import __version__
from mousebrainbench.artifacts import code_revision
from mousebrainbench.benchmarks.hybrid_selective_policy import (
    semantic_admissibility_matrix,
)
from mousebrainbench.knowledge import load_default_profile
from mousebrainbench.validation.semantic_risk_control import (
    authorize_with_policy,
    calibrate_semantic_risk,
    semantic_false_authorization_metrics,
)

DEFAULT_DEVELOPMENT = Path("results/hybrid_selective_confirmation/cases.npz")
DEFAULT_SCORE_MODEL = Path("results/hybrid_selective_policy/model.json")
DEFAULT_FROZEN_POLICY = Path("results/semantic_risk_policy/model.json")
DEFAULT_FRESH = Path("results/semantic_risk_confirmation/cases.npz")
DEFAULT_OUTPUT = Path("results/semantic_risk_sensitivity/summary.json")
DEFAULT_MARKDOWN = Path("results/semantic_risk_sensitivity/summary.md")

CALIBRATION_SIZES = (250, 500, 1_000, 2_000, 3_600)
DEFAULT_REPEATS = 20
SEED_NAMESPACE = 20260829


def deterministic_subsample_indices(
    population_size: int,
    sample_size: int,
    *,
    repeat: int,
    seed_namespace: int = SEED_NAMESPACE,
) -> np.ndarray:
    """Return a reproducible unordered subsample without outcome adaptation."""

    if not 0 < sample_size <= population_size:
        raise ValueError("sample size must be positive and no larger than population")
    if repeat < 0:
        raise ValueError("repeat must be non-negative")
    if sample_size == population_size:
        return np.arange(population_size, dtype=int)
    rng = np.random.default_rng(seed_namespace + sample_size * 10_000 + repeat)
    return np.sort(rng.choice(population_size, size=sample_size, replace=False))


def _sha256(path: Path) -> str:
    return f"sha256:{hashlib.sha256(path.read_bytes()).hexdigest()}"


def _complete_requirements() -> dict[str, tuple[str, ...]]:
    return {
        item.claim: item.required_blocks for item in load_default_profile().requirements
    }


def _per_claim_metrics(
    decisions: np.ndarray,
    labels: np.ndarray,
    claims: Sequence[str],
) -> list[dict[str, Any]]:
    return [
        {
            "claim": claim,
            **semantic_false_authorization_metrics(
                decisions[:, index : index + 1],
                labels[:, index : index + 1],
            ),
        }
        for index, claim in enumerate(claims)
    ]


def _aggregate_rows(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Summarize repeat variability without selecting the best repeat."""

    coverage = np.asarray([row["supported_coverage"] for row in rows], dtype=float)
    sfar = np.asarray(
        [row["semantic_false_authorization_risk"] for row in rows], dtype=float
    )
    certified = np.asarray([row["certified_claims"] for row in rows], dtype=float)
    return {
        "repeats": len(rows),
        "certified_claims_mean": float(certified.mean()),
        "certified_claims_min": int(certified.min()),
        "certified_claims_max": int(certified.max()),
        "fresh_supported_coverage_mean": float(coverage.mean()),
        "fresh_supported_coverage_std": float(coverage.std(ddof=0)),
        "fresh_supported_coverage_min": float(coverage.min()),
        "fresh_supported_coverage_max": float(coverage.max()),
        "fresh_sfar_mean": float(sfar.mean()),
        "fresh_sfar_std": float(sfar.std(ddof=0)),
        "fresh_sfar_min": float(sfar.min()),
        "fresh_sfar_max": float(sfar.max()),
        "fraction_repeats_at_or_below_target": float(
            np.mean([row["sfar_at_or_below_target"] for row in rows])
        ),
    }


def run(
    *,
    development_path: Path = DEFAULT_DEVELOPMENT,
    score_model_path: Path = DEFAULT_SCORE_MODEL,
    frozen_policy_path: Path = DEFAULT_FROZEN_POLICY,
    fresh_path: Path = DEFAULT_FRESH,
    output: Path = DEFAULT_OUTPUT,
    markdown: Path = DEFAULT_MARKDOWN,
    repeats: int = DEFAULT_REPEATS,
) -> Path:
    """Recalibrate on development subsets and report v3 sensitivity."""

    if repeats < 1:
        raise ValueError("repeats must be positive")
    score_model = json.loads(score_model_path.read_text())
    frozen = json.loads(frozen_policy_path.read_text())
    claim_names = tuple(str(value) for value in score_model["claim_names"])
    variable_claims = tuple(str(value) for value in frozen["variable_claims"])
    variable_indices = np.asarray([claim_names.index(claim) for claim in variable_claims])
    target_sfar = float(frozen["semantic_policy"]["target_sfar"])
    familywise_confidence = float(
        frozen["semantic_policy"]["familywise_confidence"]
    )

    development = np.load(development_path, allow_pickle=False)
    development_labels = development["labels"].astype(bool)[:, variable_indices]
    development_scores = development["full_calibrated_probabilities"].astype(float)[
        :, variable_indices
    ]
    development_gates = semantic_admissibility_matrix(
        development["features"].astype(float),
        claim_names=claim_names,
        feature_names=score_model["feature_names"],
        support_vetoes=_complete_requirements(),
    )[:, variable_indices]

    fresh = np.load(fresh_path, allow_pickle=False)
    fresh_labels = fresh["labels"].astype(bool)[:, variable_indices]
    fresh_scores = fresh["probabilities"].astype(float)[:, variable_indices]
    fresh_gates = fresh["admissible"].astype(bool)[:, variable_indices]
    if len(development_labels) != max(CALIBRATION_SIZES):
        raise ValueError("development artifact does not match declared maximum size")

    rows: list[dict[str, Any]] = []
    for size in CALIBRATION_SIZES:
        size_repeats = 1 if size == len(development_labels) else repeats
        for repeat in range(size_repeats):
            indices = deterministic_subsample_indices(
                len(development_labels), size, repeat=repeat
            )
            policy = calibrate_semantic_risk(
                development_scores[indices],
                development_labels[indices],
                development_gates[indices],
                claim_names=variable_claims,
                target_sfar=target_sfar,
                familywise_confidence=familywise_confidence,
            )
            decisions = authorize_with_policy(policy, fresh_scores, fresh_gates)
            metrics = semantic_false_authorization_metrics(decisions, fresh_labels)
            rows.append(
                {
                    "calibration_size": size,
                    "repeat": repeat,
                    "subsample_index_sha256": hashlib.sha256(
                        indices.astype(np.int64).tobytes()
                    ).hexdigest(),
                    "certified_claims": sum(
                        certificate.certified for certificate in policy.certificates
                    ),
                    "thresholds": {
                        certificate.claim: certificate.threshold
                        for certificate in policy.certificates
                    },
                    **metrics,
                    "sfar_at_or_below_target": metrics[
                        "semantic_false_authorization_risk"
                    ]
                    <= target_sfar,
                    "semantic_support_violations": int(
                        ((decisions == 1) & ~fresh_gates).sum()
                    ),
                    "per_claim": _per_claim_metrics(
                        decisions, fresh_labels, variable_claims
                    ),
                }
            )

    by_size = [
        {
            "calibration_size": size,
            **_aggregate_rows(
                [row for row in rows if row["calibration_size"] == size]
            ),
        }
        for size in CALIBRATION_SIZES
    ]
    payload = {
        "version": __version__,
        "git_revision": code_revision(),
        "analysis": "semantic_risk_calibration_size_sensitivity_v1",
        "analysis_role": "post_confirmation_exploratory_sensitivity",
        "confirmatory_reuse_prohibited": True,
        "development_artifact_sha256": _sha256(development_path),
        "fresh_artifact_sha256": _sha256(fresh_path),
        "score_model_sha256": _sha256(score_model_path),
        "frozen_policy_sha256": _sha256(frozen_policy_path),
        "target_sfar": target_sfar,
        "familywise_confidence": familywise_confidence,
        "variable_claims": list(variable_claims),
        "seed_namespace": SEED_NAMESPACE,
        "requested_repeats": repeats,
        "by_size": by_size,
        "runs": rows,
        "semantic_support_violations": int(
            sum(row["semantic_support_violations"] for row in rows)
        ),
        "interpretation_limits": [
            "The v3 outcomes had already been inspected before this sensitivity analysis.",
            "No calibration size or threshold may be selected from these results and relabelled confirmatory.",
            "The analysis measures stability within the synthetic benchmark population only.",
        ],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n")
    _write_markdown(payload, markdown)
    return output


def _write_markdown(payload: Mapping[str, Any], path: Path) -> None:
    lines = [
        "# Semantic risk calibration-size sensitivity",
        "",
        "This is a post-confirmation exploratory analysis. It is not a new confirmation.",
        "",
        "| Calibration cases | Repeats | Certified claims | Coverage (mean) | SFAR (mean) | Repeats SFAR <= target |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for row in payload["by_size"]:
        lines.append(
            f"| {row['calibration_size']} | {row['repeats']} | "
            f"{row['certified_claims_mean']:.2f} "
            f"[{row['certified_claims_min']}, {row['certified_claims_max']}] | "
            f"{row['fresh_supported_coverage_mean']:.4f} | "
            f"{row['fresh_sfar_mean']:.4f} | "
            f"{row['fraction_repeats_at_or_below_target']:.3f} |"
        )
    lines.extend(("", "## Interpretation limits", ""))
    lines.extend(f"- {item}" for item in payload["interpretation_limits"])
    lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--markdown", type=Path, default=DEFAULT_MARKDOWN)
    parser.add_argument("--repeats", type=int, default=DEFAULT_REPEATS)
    args = parser.parse_args()
    result = run(output=args.output, markdown=args.markdown, repeats=args.repeats)
    print(json.dumps({"output": str(result.resolve())}))


if __name__ == "__main__":
    main()
