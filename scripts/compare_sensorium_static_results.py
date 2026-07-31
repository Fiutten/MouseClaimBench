"""Summarize Sensorium 2022 static results as a cross-dataset comparator."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import median
from typing import Any


def _median(values: list[float]) -> float | None:
    return median(values) if values else None


def _round(value: float | None) -> float | None:
    return None if value is None else round(float(value), 6)


def build_static_comparison(
    input_summary: Path,
    *,
    topographic_summary: Path | None = None,
) -> dict[str, Any]:
    """Build a compact comparison from existing Sensorium 2022 static artifacts."""

    payload = json.loads(input_summary.read_text())
    rows = payload["rows"]
    validation = [row for row in rows if row["eval_tier"] == "validation"]
    repeated_test = [
        row
        for row in rows
        if row["eval_tier"] == "test" and float(row.get("reliability", 0.0)) > 0.0
    ]

    def summarize(selected: list[dict[str, Any]]) -> dict[str, Any]:
        return {
            "n": len(selected),
            "median_reliability": _median([float(row["reliability"]) for row in selected]),
            "median_best_predictive_correlation": _median(
                [float(row["best_predictive_correlation"]) for row in selected]
            ),
            "median_best_minus_mean": _median(
                [float(row["best_predictive_minus_mean"]) for row in selected]
            ),
            "median_best_minus_scrambled": _median(
                [float(row["best_predictive_minus_scrambled"]) for row in selected]
            ),
            "mis_passed_count": sum(bool(row["mis_passed"]) for row in selected),
            "median_mis_score": _median([float(row["mis_score"]) for row in selected]),
        }

    topographic = json.loads(topographic_summary.read_text()) if topographic_summary else None
    return {
        "comparison": "sensorium2022_static_cross_dataset_comparator",
        "input_summary": str(input_summary),
        "topographic_summary": str(topographic_summary) if topographic_summary else None,
        "validation_all_mice": summarize(validation),
        "pretraining_test_repeated": summarize(repeated_test),
        "topographic_constraint": {
            "available": topographic is not None,
            "passed_count": topographic.get("passed_count") if topographic else None,
            "n_datasets": topographic.get("n_datasets") if topographic else None,
            "median_observed_spearman": topographic.get("median_observed_spearman")
            if topographic
            else None,
            "median_effect_over_null": topographic.get("median_effect_over_null")
            if topographic
            else None,
            "decision": topographic.get("decision") if topographic else None,
        },
        "rows": rows,
        "interpretation": (
            "Sensorium 2022 static is the current positive reliability case: "
            "repeated-test mice have estimable response reliability and positive "
            "stimulus-specific prediction. When the topographic constraint is "
            "provided, it adds structural evidence; MIS still requires caution "
            "because this is not an interventional causal circuit test."
        ),
    }


def write_outputs(payload: dict[str, Any], output_json: Path, output_md: Path) -> None:
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(payload, indent=2))
    lines = [
        "# Sensorium 2022 Static Comparator",
        "",
        payload["interpretation"],
        "",
        "| Cohorte | n | Reliability | Best corr | Best - mean | Best - scrambled | MIS passed | MIS score |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for label in ("validation_all_mice", "pretraining_test_repeated"):
        row = payload[label]
        lines.append(
            "| "
            f"{label} | `{row['n']}` | `{_round(row['median_reliability'])}` | "
            f"`{_round(row['median_best_predictive_correlation'])}` | "
            f"`{_round(row['median_best_minus_mean'])}` | "
            f"`{_round(row['median_best_minus_scrambled'])}` | "
            f"`{row['mis_passed_count']}` | `{_round(row['median_mis_score'])}` |"
        )
    topo = payload["topographic_constraint"]
    if topo["available"]:
        lines.extend(
            [
                "",
                "## Topographic Structural Constraint",
                "",
                "| n | Passed | Median Spearman | Median effect over null | Decision |",
                "|---:|---:|---:|---:|---|",
                "| "
                f"`{topo['n_datasets']}` | `{topo['passed_count']}` | "
                f"`{_round(topo['median_observed_spearman'])}` | "
                f"`{_round(topo['median_effect_over_null'])}` | "
                f"`{topo['decision']}` |",
            ]
        )
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_md.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-summary",
        type=Path,
        default=Path("results/sensorium_real/summary_sensorium2022_static_mis.json"),
    )
    parser.add_argument(
        "--topographic-summary",
        type=Path,
        default=Path("results/sensorium_topographic_constraint/summary_static_test.json"),
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("results/sensorium_static_model_comparator/summary.json"),
    )
    parser.add_argument(
        "--output-md",
        type=Path,
        default=Path("results/sensorium_static_model_comparator/summary.md"),
    )
    args = parser.parse_args()
    payload = build_static_comparison(
        args.input_summary,
        topographic_summary=args.topographic_summary if args.topographic_summary.exists() else None,
    )
    write_outputs(payload, args.output_json, args.output_md)
    print(json.dumps({"output_json": str(args.output_json)}))


if __name__ == "__main__":
    main()
