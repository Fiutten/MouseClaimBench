"""Sensitivity checks for the Q1 MouseBrainBench evidence package.

This module reads tracked result artifacts and asks whether the central
decisions survive modest threshold or effect-size perturbations. It avoids
rerunning expensive data pipelines; the goal is a reproducible robustness audit
of the evidence already produced.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import median
from typing import Any

from mousebrainbench import __version__
from mousebrainbench.artifacts import code_revision


DEFAULT_OUTPUT = Path("results/q1_sensitivity/summary.json")
DEFAULT_MARKDOWN = Path("results/q1_sensitivity/summary.md")


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _criterion_passed(value: float, threshold: float, direction: str) -> bool:
    if direction == "gt":
        return value > threshold
    if direction == "gte":
        return value >= threshold
    if direction == "lt":
        return value < threshold
    if direction == "lte":
        return value <= threshold
    raise ValueError(f"unknown criterion direction: {direction}")


def _allen_threshold_sensitivity(path: Path) -> dict[str, Any]:
    payload = _load(path)
    rows = []
    for multiplier in (0.75, 1.0, 1.25):
        blocks = []
        for block in payload["mis"]["blocks"]:
            criteria = []
            for criterion in block["criteria"]:
                threshold = float(criterion["threshold"]) * multiplier
                criteria.append(
                    _criterion_passed(
                        float(criterion["value"]),
                        threshold,
                        str(criterion["direction"]),
                    )
                )
            blocks.append({"name": block["name"], "passed": all(criteria)})
        rows.append(
            {
                "threshold_multiplier": multiplier,
                "passed_blocks": [block["name"] for block in blocks if block["passed"]],
                "mis_passed": all(block["passed"] for block in blocks),
            }
        )
    return {
        "artifact": str(path),
        "nominal_decision": payload["decision"],
        "rows": rows,
        "stable_negative": all(not row["mis_passed"] for row in rows),
    }


def _sensorium_static_sensitivity(path: Path) -> dict[str, Any]:
    payload = _load(path)
    repeated = payload["pretraining_test_repeated"]
    topo = payload["topographic_constraint"]
    rows = []
    for reliability_threshold in (0.5, 0.6, 0.7):
        for topographic_threshold in (0.05, 0.10, 0.15):
            reliability_passed = float(repeated["median_reliability"]) > reliability_threshold
            topographic_passed = float(topo["median_effect_over_null"]) > topographic_threshold
            rows.append(
                {
                    "reliability_threshold": reliability_threshold,
                    "topographic_threshold": topographic_threshold,
                    "reliability_passed": reliability_passed,
                    "topographic_passed": topographic_passed,
                    "partial_positive": reliability_passed and topographic_passed,
                }
            )
    return {
        "artifact": str(path),
        "median_reliability": repeated["median_reliability"],
        "median_topographic_effect": topo["median_effect_over_null"],
        "rows": rows,
        "stable_partial_positive": sum(row["partial_positive"] for row in rows),
        "n_checks": len(rows),
    }


def _dynamic_sensorium_sensitivity(path: Path) -> dict[str, Any]:
    payload = _load(path)
    cohorts = []
    for cohort in payload["cohorts"]:
        rows = cohort["model_rows"]
        mlp_minus_svd = [
            float(row["torch_mlp"]) - float(row["temporal_svd"])
            for row in rows
            if row.get("torch_mlp") is not None and row.get("temporal_svd") is not None
        ]
        mlp_minus_mean = [
            float(row["torch_mlp"]) - float(row["mean_response"])
            for row in rows
            if row.get("torch_mlp") is not None and row.get("mean_response") is not None
        ]
        thresholds = []
        for effect_threshold in (0.0, 0.002, 0.005, 0.01):
            thresholds.append(
                {
                    "effect_threshold": effect_threshold,
                    "mlp_beats_svd": sum(value > effect_threshold for value in mlp_minus_svd),
                    "mlp_beats_mean": sum(value > effect_threshold for value in mlp_minus_mean),
                }
            )
        cohorts.append(
            {
                "cohort": cohort["cohort"],
                "n_mice": cohort["n_mice"],
                "median_mlp_minus_svd": median(mlp_minus_svd),
                "median_mlp_minus_mean": median(mlp_minus_mean),
                "threshold_rows": thresholds,
                "interpretation": (
                    "NN predictive gains are present but small; robustness should be "
                    "reported as predictive evidence, not as mechanistic superiority."
                ),
            }
        )
    return {"artifact": str(path), "cohorts": cohorts}


def run_sensitivity(
    *,
    allen: Path = Path("results/allen_vbn_mechanistic_identifiability_score.json"),
    static: Path = Path("results/sensorium_static_model_comparator/summary.json"),
    dynamic: Path = Path("results/dynamic_sensorium_model_comparator/summary.json"),
) -> dict[str, Any]:
    """Build the full sensitivity payload from tracked artifacts."""

    return {
        "version": __version__,
        "git_revision": code_revision(),
        "allen_vbn": _allen_threshold_sensitivity(allen),
        "sensorium_static": _sensorium_static_sensitivity(static),
        "dynamic_sensorium": _dynamic_sensorium_sensitivity(dynamic),
        "decision": (
            "sensitivity_supports_methodological_claim_not_q1_mechanistic_claim"
        ),
        "interpretation": (
            "Allen remains negative under threshold perturbation; Sensorium static "
            "has robust partial reliability/topography evidence; Dynamic Sensorium "
            "has small but useful NN predictive gains. Together this supports a "
            "methodological benchmark claim, while still requiring an official "
            "baseline or causal/structural extension for a strong Q1 claim."
        ),
    }


def write_outputs(payload: dict[str, Any], output: Path, markdown: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2))
    lines = [
        "# Q1 Sensitivity Audit",
        "",
        f"- Decision: `{payload['decision']}`",
        f"- Allen stable negative: `{payload['allen_vbn']['stable_negative']}`",
        (
            "- Sensorium static partial positives: "
            f"`{payload['sensorium_static']['stable_partial_positive']}/"
            f"{payload['sensorium_static']['n_checks']}`"
        ),
        "",
        "## Dynamic Sensorium",
        "",
        "| Cohort | Median MLP - mean | Median MLP - SVD |",
        "|---|---:|---:|",
    ]
    for cohort in payload["dynamic_sensorium"]["cohorts"]:
        lines.append(
            f"| `{cohort['cohort']}` | `{cohort['median_mlp_minus_mean']:.6f}` | "
            f"`{cohort['median_mlp_minus_svd']:.6f}` |"
        )
    lines.extend(["", "## Interpretation", "", payload["interpretation"], ""])
    markdown.parent.mkdir(parents=True, exist_ok=True)
    markdown.write_text("\n".join(lines))


def run(output: Path = DEFAULT_OUTPUT, markdown: Path = DEFAULT_MARKDOWN) -> Path:
    payload = run_sensitivity()
    write_outputs(payload, output, markdown)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--markdown", type=Path, default=DEFAULT_MARKDOWN)
    args = parser.parse_args()
    print(json.dumps({"output": str(run(args.output, args.markdown).resolve())}))


if __name__ == "__main__":
    main()
