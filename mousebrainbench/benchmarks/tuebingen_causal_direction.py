"""Tuebingen cause-effect external direction benchmark.

The adapter uses a deliberately lightweight residual-dependence heuristic. The
goal is not causal-discovery SOTA. The goal is to test whether claim auditing can
separate high association from justified directional wording on a public causal
benchmark.
"""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path
from typing import Any

import numpy as np

from mousebrainbench import __version__
from mousebrainbench.artifacts import code_revision


DEFAULT_ROOT = Path("data/external/tuebingen_cause_effect")
DEFAULT_OUTPUT = Path("results/tuebingen_causal_direction/summary.json")
DEFAULT_MARKDOWN = Path("results/tuebingen_causal_direction/summary.md")
BOOTSTRAP_SEED = 20260801
BOOTSTRAP_SAMPLES = 2000


def _load_meta(path: Path) -> dict[int, dict[str, Any]]:
    rows: dict[int, dict[str, Any]] = {}
    for line in path.read_text().splitlines():
        parts = line.split()
        if len(parts) < 6:
            continue
        pair = int(parts[0])
        cause_start, cause_end, effect_start, effect_end = (int(part) for part in parts[1:5])
        rows[pair] = {
            "cause_start": cause_start - 1,
            "cause_end": cause_end,
            "effect_start": effect_start - 1,
            "effect_end": effect_end,
            "weight": float(parts[5]),
        }
    return rows


def _standardize(values: np.ndarray) -> np.ndarray:
    std = np.std(values)
    if std == 0:
        return values * 0.0
    return (values - np.mean(values)) / std


def _polyfit_residual(source: np.ndarray, target: np.ndarray, degree: int = 2) -> np.ndarray:
    """Return standardized polynomial residuals for one direction."""

    source = _standardize(source.reshape(-1))
    target = _standardize(target.reshape(-1))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", np.exceptions.RankWarning)
        coeff = np.polyfit(source, target, deg=degree)
    predicted = np.polyval(coeff, source)
    return target - predicted


def _normalized_residual_error(source: np.ndarray, target: np.ndarray, degree: int = 2) -> float:
    """Return normalized polynomial residual error for one direction."""

    target = _standardize(target.reshape(-1))
    residual = _polyfit_residual(source, target, degree=degree)
    return float(np.mean(residual**2) / (np.var(target) + 1e-12))


def _anm_direction(cause_values: np.ndarray, effect_values: np.ndarray) -> tuple[str, float]:
    forward = min(_normalized_residual_error(cause_values, effect_values, degree=d) for d in (1, 2, 3))
    backward = min(_normalized_residual_error(effect_values, cause_values, degree=d) for d in (1, 2, 3))
    margin = abs(backward - forward)
    if margin < 0.02:
        return "uncertain", margin
    return ("forward", margin) if forward < backward else ("backward", margin)


def _igci_slope_score(source: np.ndarray, target: np.ndarray) -> float:
    """Lightweight IGCI-style slope entropy score.

    The Tuebingen benchmark paper discusses IGCI as a family of causal direction
    methods. This implementation is a transparent slope-complexity proxy, not a
    claim to reproduce every published IGCI variant.
    """

    source = _standardize(source.reshape(-1))
    target = _standardize(target.reshape(-1))
    order = np.argsort(source)
    x = source[order]
    y = target[order]
    dx = np.diff(x)
    dy = np.diff(y)
    mask = np.abs(dx) > 1e-9
    if not np.any(mask):
        return float("inf")
    slopes = np.abs(dy[mask] / dx[mask]) + 1e-12
    return float(np.mean(np.log(slopes)))


def _igci_direction(cause_values: np.ndarray, effect_values: np.ndarray) -> tuple[str, float]:
    forward = _igci_slope_score(cause_values, effect_values)
    backward = _igci_slope_score(effect_values, cause_values)
    margin = abs(backward - forward)
    if not np.isfinite(margin) or margin < 0.02:
        return "uncertain", 0.0 if not np.isfinite(margin) else margin
    return ("forward", margin) if forward < backward else ("backward", margin)


def _safe_abs_corr(left: np.ndarray, right: np.ndarray) -> float:
    if np.std(left) == 0 or np.std(right) == 0:
        return 0.0
    value = np.corrcoef(left.reshape(-1), right.reshape(-1))[0, 1]
    if not np.isfinite(value):
        return 0.0
    return abs(float(value))


def _residual_dependence_score(source: np.ndarray, residual: np.ndarray) -> float:
    """Approximate residual-source dependence without optional HSIC packages."""

    source = _standardize(source.reshape(-1))
    residual = _standardize(residual.reshape(-1))
    return float(
        max(
            _safe_abs_corr(source, residual),
            _safe_abs_corr(source, residual**2),
            _safe_abs_corr(source**2, residual),
            _safe_abs_corr(source**2, residual**2),
        )
    )


def _lingam_proxy_direction(cause_values: np.ndarray, effect_values: np.ndarray) -> tuple[str, float]:
    """Linear non-Gaussian causal-direction proxy.

    This is not a full DirectLiNGAM implementation. It is a transparent
    residual-independence baseline that is valid only as a reviewer-facing
    control against correlation-only causal wording.
    """

    forward_residual = _polyfit_residual(cause_values, effect_values, degree=1)
    backward_residual = _polyfit_residual(effect_values, cause_values, degree=1)
    forward_dep = _residual_dependence_score(cause_values, forward_residual)
    backward_dep = _residual_dependence_score(effect_values, backward_residual)
    margin = abs(backward_dep - forward_dep)
    if margin < 0.02:
        return "uncertain", margin
    return ("forward", margin) if forward_dep < backward_dep else ("backward", margin)


def _ensemble_direction(
    cause_values: np.ndarray, effect_values: np.ndarray
) -> tuple[str, float, str, str, str, int]:
    anm_direction, anm_margin = _anm_direction(cause_values, effect_values)
    igci_direction, igci_margin = _igci_direction(cause_values, effect_values)
    lingam_direction, lingam_margin = _lingam_proxy_direction(cause_values, effect_values)
    votes = [anm_direction, igci_direction, lingam_direction]
    forward_votes = sum(vote == "forward" for vote in votes)
    backward_votes = sum(vote == "backward" for vote in votes)
    if forward_votes >= 2:
        margins = [
            margin
            for vote, margin in (
                (anm_direction, anm_margin),
                (igci_direction, igci_margin),
                (lingam_direction, lingam_margin),
            )
            if vote == "forward"
        ]
        return "forward", min(margins), anm_direction, igci_direction, lingam_direction, forward_votes
    if backward_votes >= 2:
        margins = [
            margin
            for vote, margin in (
                (anm_direction, anm_margin),
                (igci_direction, igci_margin),
                (lingam_direction, lingam_margin),
            )
            if vote == "backward"
        ]
        return "backward", min(margins), anm_direction, igci_direction, lingam_direction, backward_votes
    if anm_direction != "uncertain" and anm_margin >= 0.12:
        return anm_direction, anm_margin, anm_direction, igci_direction, lingam_direction, 1
    if igci_direction != "uncertain" and igci_margin >= 0.12:
        return igci_direction, igci_margin, anm_direction, igci_direction, lingam_direction, 1
    if lingam_direction != "uncertain" and lingam_margin >= 0.12:
        return lingam_direction, lingam_margin, anm_direction, igci_direction, lingam_direction, 1
    return (
        "uncertain",
        max(anm_margin, igci_margin, lingam_margin),
        anm_direction,
        igci_direction,
        lingam_direction,
        max(forward_votes, backward_votes),
    )


def _wilson_interval(successes: int, total: int) -> list[float]:
    """Return a 95% Wilson interval for unweighted direction accuracy."""

    if total <= 0:
        return [0.0, 0.0]
    z = 1.959963984540054
    proportion = successes / total
    denominator = 1.0 + z**2 / total
    centre = (proportion + z**2 / (2.0 * total)) / denominator
    radius = (
        z
        * np.sqrt(proportion * (1.0 - proportion) / total + z**2 / (4.0 * total**2))
        / denominator
    )
    return [float(max(0.0, centre - radius)), float(min(1.0, centre + radius))]


def _weighted_accuracy_bootstrap(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Bootstrap weighted accuracy over attempted benchmark pairs."""

    if not rows:
        return {"samples": 0, "seed": BOOTSTRAP_SEED, "ci95": [0.0, 0.0]}
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    values = []
    for _ in range(BOOTSTRAP_SAMPLES):
        sampled = [rows[index] for index in rng.integers(0, len(rows), size=len(rows))]
        denominator = sum(float(row["weight"]) for row in sampled)
        numerator = sum(float(row["weight"]) for row in sampled if row["correct_direction"])
        values.append(numerator / denominator if denominator else 0.0)
    return {
        "samples": BOOTSTRAP_SAMPLES,
        "seed": BOOTSTRAP_SEED,
        "unit": "attempted cause-effect pair",
        "ci95": [float(value) for value in np.quantile(values, (0.025, 0.975))],
    }


def run(
    root: Path = DEFAULT_ROOT,
    output: Path = DEFAULT_OUTPUT,
    markdown: Path = DEFAULT_MARKDOWN,
    max_pairs: int | None = None,
) -> Path:
    """Run Tuebingen external causal-direction adapter."""

    meta_path = root / "pairmeta.txt"
    if not meta_path.exists():
        payload = {
            "version": __version__,
            "git_revision": code_revision(),
            "analysis": "tuebingen_causal_direction",
            "decision": "tuebingen_data_missing",
            "missing": [str(meta_path)],
        }
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(payload, indent=2))
        write_markdown(payload, markdown)
        return output

    meta = _load_meta(meta_path)
    rows: list[dict[str, Any]] = []
    for pair_id in sorted(meta):
        if max_pairs is not None and len(rows) >= max_pairs:
            break
        path = root / f"pair{pair_id:04d}.txt"
        if not path.exists():
            continue
        values = np.loadtxt(path)
        item = meta[pair_id]
        cause = values[:, item["cause_start"] : item["cause_end"]].mean(axis=1)
        effect = values[:, item["effect_start"] : item["effect_end"]].mean(axis=1)
        corr = abs(float(np.corrcoef(cause, effect)[0, 1]))
        (
            direction,
            margin,
            anm_direction,
            igci_direction,
            lingam_proxy_direction,
            consensus_votes,
        ) = _ensemble_direction(cause, effect)
        correct = direction == "forward"
        rows.append(
            {
                "pair_id": pair_id,
                "absolute_correlation": corr,
                "predicted_direction": direction,
                "anm_direction": anm_direction,
                "igci_direction": igci_direction,
                "lingam_proxy_direction": lingam_proxy_direction,
                "consensus_votes": consensus_votes,
                "direction_margin": margin,
                "correct_direction": correct,
                "weight": item["weight"],
                "correlation_only_would_overclaim_direction": corr >= 0.30,
            }
        )

    attempted = [row for row in rows if row["predicted_direction"] != "uncertain"]
    correct = sum(row["correct_direction"] for row in attempted)
    weighted_total = sum(row["weight"] for row in attempted)
    weighted_correct = sum(row["weight"] for row in attempted if row["correct_direction"])
    weighted_accuracy = weighted_correct / weighted_total if weighted_total else 0.0
    weighted_bootstrap = _weighted_accuracy_bootstrap(attempted)
    corr_overclaims = sum(row["correlation_only_would_overclaim_direction"] for row in rows)
    method_summary = {}
    for method_key in ("anm_direction", "igci_direction", "lingam_proxy_direction"):
        method_attempted = [row for row in rows if row[method_key] != "uncertain"]
        method_correct = sum(row[method_key] == "forward" for row in method_attempted)
        method_summary[method_key.replace("_direction", "")] = {
            "attempts": len(method_attempted),
            "coverage": len(method_attempted) / len(rows) if rows else 0.0,
            "accuracy": method_correct / len(method_attempted) if method_attempted else 0.0,
        }
    confidence_curve = []
    for threshold in (0.02, 0.05, 0.10, 0.15, 0.20):
        subset = [
            row
            for row in rows
            if row["predicted_direction"] != "uncertain" and row["direction_margin"] >= threshold
        ]
        confidence_curve.append(
            {
                "margin_threshold": threshold,
                "coverage": len(subset) / len(rows) if rows else 0.0,
                "accuracy": (
                    sum(row["correct_direction"] for row in subset) / len(subset) if subset else 0.0
                ),
            }
        )
    consensus_curve = []
    for min_votes in (1, 2, 3):
        subset = [
            row
            for row in rows
            if row["predicted_direction"] != "uncertain" and row["consensus_votes"] >= min_votes
        ]
        consensus_curve.append(
            {
                "min_consensus_votes": min_votes,
                "coverage": len(subset) / len(rows) if rows else 0.0,
                "accuracy": (
                    sum(row["correct_direction"] for row in subset) / len(subset) if subset else 0.0
                ),
            }
        )
    payload = {
        "version": __version__,
        "git_revision": code_revision(),
        "analysis": "tuebingen_causal_direction",
        "dataset": "Tuebingen cause-effect pairs",
        "num_pairs_loaded": len(rows),
        "num_direction_attempts": len(attempted),
        "direction_attempt_rate": len(attempted) / len(rows) if rows else 0.0,
        "direction_accuracy": correct / len(attempted) if attempted else 0.0,
        "direction_accuracy_wilson_95": _wilson_interval(correct, len(attempted)),
        "weighted_direction_accuracy": weighted_accuracy,
        "weighted_direction_accuracy_bootstrap": weighted_bootstrap,
        "weight_source": "official pair weights stored in Tuebingen pairmeta.txt",
        "weight_formula": "sum(pair_weight * correct) / sum(pair_weight) over attempted pairs",
        "correlation_overclaim_threshold": 0.30,
        "correlation_only_direction_overclaims": corr_overclaims,
        "method_summary": method_summary,
        "confidence_curve": confidence_curve,
        "consensus_curve": consensus_curve,
        "causal_performance_claim_allowed": False,
        "causal_control_claim_allowed": len(rows) >= 100 and corr_overclaims > 0,
        "rows": rows,
        "decision": (
            "tuebingen_external_direction_benchmark_ready"
            if len(rows) >= 100 and len(attempted) >= 20
            else "tuebingen_external_direction_benchmark_insufficient"
        ),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2))
    write_markdown(payload, markdown)
    return output


def write_markdown(payload: dict[str, Any], markdown: Path) -> None:
    """Write Tuebingen adapter report."""

    lines = [
        "# Tuebingen Causal Direction Adapter",
        "",
        f"- Decision: `{payload['decision']}`",
        f"- Pairs loaded: `{payload.get('num_pairs_loaded', 0)}`",
        f"- Direction attempts: `{payload.get('num_direction_attempts', 0)}`",
        f"- Direction attempt rate: `{payload.get('direction_attempt_rate', 0.0):.3f}`",
        f"- Direction accuracy: `{payload.get('direction_accuracy', 0.0):.3f}`",
        f"- Direction accuracy Wilson 95% CI: `{payload.get('direction_accuracy_wilson_95')}`",
        f"- Weighted accuracy: `{payload.get('weighted_direction_accuracy', 0.0):.3f}`",
        f"- Weighted accuracy bootstrap 95% CI: "
        f"`{payload.get('weighted_direction_accuracy_bootstrap', {}).get('ci95')}`",
        f"- Correlation-only direction overclaims: `{payload.get('correlation_only_direction_overclaims', 0)}`",
        f"- Causal performance claim allowed: `{payload.get('causal_performance_claim_allowed')}`",
        f"- Causal control claim allowed: `{payload.get('causal_control_claim_allowed')}`",
        "",
        "## Method Summary",
        "",
        "| Method | Attempts | Coverage | Accuracy |",
        "|---|---:|---:|---:|",
    ]
    for method, row in payload.get("method_summary", {}).items():
        lines.append(
            f"| `{method}` | `{row['attempts']}` | `{row['coverage']:.3f}` | "
            f"`{row['accuracy']:.3f}` |"
        )
    lines.extend(
        [
            "",
            "Interpretation: these methods are transparent causal-direction controls. "
            "They are used to audit causal wording, not to claim causal-discovery SOTA.",
        "",
        ]
    )
    markdown.parent.mkdir(parents=True, exist_ok=True)
    markdown.write_text("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--markdown", type=Path, default=DEFAULT_MARKDOWN)
    parser.add_argument("--max-pairs", type=int, default=None)
    args = parser.parse_args()
    print(json.dumps({"output": str(run(args.root, args.output, args.markdown, args.max_pairs).resolve())}))


if __name__ == "__main__":
    main()
