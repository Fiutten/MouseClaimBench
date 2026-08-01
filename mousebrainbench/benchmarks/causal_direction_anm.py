"""Assumption-aware directional evidence using causal-learn's official ANM.

The adapter does not convert observational direction into causal proof. It
records forward and reverse residual-independence p-values, then applies the
margin fixed in the hybrid-selective v2 protocol. Ambiguous or failed numerical
execution is escalated to review rather than forced into a direction.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import warnings
from pathlib import Path
from typing import Any

import numpy as np

from mousebrainbench import __version__
from mousebrainbench.artifacts import code_revision
from mousebrainbench.benchmarks.tuebingen_causal_direction import (
    _load_meta,
    _weighted_accuracy_bootstrap,
    _wilson_interval,
)


DEFAULT_ROOT = Path("data/external/tuebingen_cause_effect")
DEFAULT_OUTPUT = Path("results/anm_direction_development/summary.json")
DEFAULT_MARKDOWN = Path("results/anm_direction_development/summary.md")
MAX_SAMPLES = 200
MARGIN_THRESHOLD = 0.10
SUBSAMPLE_SEED_NAMESPACE = 2_026_081_100


def _anm_class():
    """Import the optional established implementation with an actionable error."""

    try:
        from causallearn.search.FCMBased.ANM.ANM import ANM
    except ImportError as exc:
        raise RuntimeError(
            "ANM direction requires the `hybrid-validation` optional dependencies"
        ) from exc
    return ANM


def _standardize(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float).reshape(-1)
    scale = float(values.std())
    return (values - values.mean()) / scale if scale > 0.0 else values * 0.0


def _prepare_pair(
    x: np.ndarray,
    y: np.ndarray,
    *,
    seed: int,
    max_samples: int = MAX_SAMPLES,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Filter, deterministically subsample, and standardize one variable pair."""

    x_values = np.asarray(x, dtype=float).reshape(-1)
    y_values = np.asarray(y, dtype=float).reshape(-1)
    if len(x_values) != len(y_values):
        raise ValueError("direction variables must contain the same number of observations")
    finite = np.isfinite(x_values) & np.isfinite(y_values)
    x_values = x_values[finite]
    y_values = y_values[finite]
    available = len(x_values)
    if available < 20:
        raise ValueError("ANM direction requires at least 20 finite paired observations")
    if available > max_samples:
        indices = np.random.default_rng(seed).choice(available, max_samples, replace=False)
        x_values = x_values[indices]
        y_values = y_values[indices]
    if np.std(x_values) == 0.0 or np.std(y_values) == 0.0:
        raise ValueError("ANM direction requires non-constant variables")
    return (
        _standardize(x_values)[:, None],
        _standardize(y_values)[:, None],
        {
            "available_finite_samples": available,
            "used_samples": len(x_values),
            "subsampled": available > max_samples,
            "subsample_seed": seed,
            "max_samples": max_samples,
        },
    )


def anm_direction_evidence(
    x: np.ndarray,
    y: np.ndarray,
    *,
    seed: int,
    max_samples: int = MAX_SAMPLES,
    margin_threshold: float = MARGIN_THRESHOLD,
) -> dict[str, Any]:
    """Return a three-way forward, reverse, or review ANM direction decision."""

    try:
        x_values, y_values, sampling = _prepare_pair(
            x,
            y,
            seed=seed,
            max_samples=max_samples,
        )
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            p_forward, p_backward = _anm_class()().cause_or_effect(x_values, y_values)
        forward = float(p_forward)
        backward = float(p_backward)
        if not np.isfinite(forward) or not np.isfinite(backward):
            raise ValueError("ANM returned a non-finite independence p-value")
        signed_margin = forward - backward
        direction = (
            "forward"
            if signed_margin >= margin_threshold
            else "reverse"
            if signed_margin <= -margin_threshold
            else "uncertain"
        )
        status = (
            "passed"
            if direction == "forward"
            else "failed"
            if direction == "reverse"
            else "requires_review"
        )
        return {
            "method": "causal-learn additive noise model",
            "method_version": "0.1.4.8",
            "p_forward": forward,
            "p_backward": backward,
            "signed_margin": signed_margin,
            "absolute_margin": abs(signed_margin),
            "margin_threshold": margin_threshold,
            "predicted_direction": direction,
            "status": status,
            "sampling": sampling,
            "runtime_warning_types": sorted({type(item.message).__name__ for item in caught}),
            "assumptions": [
                "continuous variables",
                "additive noise",
                "no hidden common cause",
            ],
            "causal_proof_allowed": False,
            "execution_error": None,
        }
    except (ValueError, FloatingPointError, np.linalg.LinAlgError) as exc:
        return {
            "method": "causal-learn additive noise model",
            "method_version": "0.1.4.8",
            "p_forward": None,
            "p_backward": None,
            "signed_margin": 0.0,
            "absolute_margin": 0.0,
            "margin_threshold": margin_threshold,
            "predicted_direction": "uncertain",
            "status": "requires_review",
            "sampling": None,
            "runtime_warning_types": [],
            "assumptions": [
                "continuous variables",
                "additive noise",
                "no hidden common cause",
            ],
            "causal_proof_allowed": False,
            "execution_error": f"{type(exc).__name__}: {exc}",
        }


def _sha256(path: Path) -> str:
    return f"sha256:{hashlib.sha256(path.read_bytes()).hexdigest()}"


def run(
    *,
    root: Path = DEFAULT_ROOT,
    output: Path = DEFAULT_OUTPUT,
    markdown: Path = DEFAULT_MARKDOWN,
    max_pairs: int | None = None,
) -> Path:
    """Evaluate frozen ANM evidence on consumed univariate Tuebingen pairs."""

    meta_path = root / "pairmeta.txt"
    if not meta_path.exists():
        payload = {
            "version": __version__,
            "git_revision": code_revision(),
            "analysis": "anm_direction_tuebingen_development",
            "decision": "tuebingen_data_missing",
            "missing": [str(meta_path)],
        }
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(payload, indent=2))
        write_markdown(payload, markdown)
        return output

    meta = _load_meta(meta_path)
    rows: list[dict[str, Any]] = []
    excluded_multivariate = 0
    for pair_id, item in sorted(meta.items()):
        cause_dimensions = item["cause_end"] - item["cause_start"]
        effect_dimensions = item["effect_end"] - item["effect_start"]
        if cause_dimensions != 1 or effect_dimensions != 1:
            excluded_multivariate += 1
            continue
        if max_pairs is not None and len(rows) >= max_pairs:
            break
        path = root / f"pair{pair_id:04d}.txt"
        if not path.exists():
            continue
        values = np.loadtxt(path)
        cause = values[:, item["cause_start"]]
        effect = values[:, item["effect_start"]]
        evidence = anm_direction_evidence(
            cause,
            effect,
            seed=SUBSAMPLE_SEED_NAMESPACE + pair_id,
        )
        rows.append(
            {
                "pair_id": pair_id,
                "weight": item["weight"],
                "predicted_direction": evidence["predicted_direction"],
                "correct_direction": evidence["predicted_direction"] == "forward",
                "p_forward": evidence["p_forward"],
                "p_backward": evidence["p_backward"],
                "signed_margin": evidence["signed_margin"],
                "status": evidence["status"],
                "used_samples": (
                    evidence["sampling"]["used_samples"] if evidence["sampling"] else 0
                ),
                "execution_error": evidence["execution_error"],
            }
        )

    attempted = [row for row in rows if row["predicted_direction"] != "uncertain"]
    correct = sum(row["correct_direction"] for row in attempted)
    total_weight = sum(float(row["weight"]) for row in attempted)
    weighted_correct = sum(
        float(row["weight"]) for row in attempted if row["correct_direction"]
    )
    weighted_accuracy = weighted_correct / total_weight if total_weight else 0.0
    weighted_bootstrap = _weighted_accuracy_bootstrap(attempted)
    payload = {
        "version": __version__,
        "git_revision": code_revision(),
        "analysis": "anm_direction_tuebingen_development",
        "dataset": "Tuebingen cause-effect pairs",
        "dataset_role": "consumed_development_only_not_independent_confirmation",
        "pairmeta_sha256": _sha256(meta_path),
        "method": "causal-learn additive noise model",
        "method_version": "0.1.4.8",
        "margin_threshold": MARGIN_THRESHOLD,
        "max_samples_per_pair": MAX_SAMPLES,
        "univariate_pairs_loaded": len(rows),
        "multivariate_pairs_excluded": excluded_multivariate,
        "attempted_pairs": len(attempted),
        "coverage": len(attempted) / len(rows) if rows else 0.0,
        "accuracy": correct / len(attempted) if attempted else 0.0,
        "accuracy_wilson_95": _wilson_interval(correct, len(attempted)),
        "weighted_accuracy": weighted_accuracy,
        "weighted_accuracy_bootstrap": weighted_bootstrap,
        "execution_errors": sum(row["execution_error"] is not None for row in rows),
        "causal_performance_claim_allowed": False,
        "external_confirmation_claim_allowed": False,
        "rows": rows,
        "decision": (
            "anm_direction_development_characterized_with_explicit_abstention"
            if len(rows) >= 100 and len(attempted) >= 20
            else "anm_direction_development_insufficient"
        ),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2))
    write_markdown(payload, markdown)
    return output


def write_markdown(payload: dict[str, Any], markdown: Path) -> None:
    lines = [
        "# ANM directional evidence development audit",
        "",
        f"- Decision: `{payload['decision']}`",
        f"- Dataset role: `{payload.get('dataset_role')}`",
        f"- Univariate pairs: `{payload.get('univariate_pairs_loaded', 0)}`",
        f"- Coverage: `{payload.get('coverage', 0.0):.3f}`",
        f"- Accuracy: `{payload.get('accuracy', 0.0):.3f}`",
        f"- Weighted accuracy: `{payload.get('weighted_accuracy', 0.0):.3f}`",
        f"- Execution errors: `{payload.get('execution_errors', 0)}`",
        "",
        "Tuebingen has already been inspected and is used only for development. "
        "ANM output remains assumption-conditional and cannot establish causal proof.",
        "",
    ]
    markdown.parent.mkdir(parents=True, exist_ok=True)
    markdown.write_text("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--markdown", type=Path, default=DEFAULT_MARKDOWN)
    parser.add_argument("--max-pairs", type=int, default=None)
    args = parser.parse_args()
    print(
        json.dumps(
            {
                "output": str(
                    run(
                        root=args.root,
                        output=args.output,
                        markdown=args.markdown,
                        max_pairs=args.max_pairs,
                    ).resolve()
                )
            }
        )
    )


if __name__ == "__main__":
    main()

