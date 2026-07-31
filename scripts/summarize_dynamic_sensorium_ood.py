"""Summarize released-response Dynamic Sensorium OOD probe results.

The OOD probe writes one JSON artifact per mouse. This script keeps the
aggregation reproducible: it reads only those per-mouse artifacts, computes
simple robust statistics, and preserves the individual rows so that every
reported number can be traced back to a concrete run.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import median
from typing import Any


def _finite_values(rows: list[dict[str, Any]], key: str) -> list[float]:
    """Return numeric values for ``key`` while ignoring missing/null fields."""

    values: list[float] = []
    for row in rows:
        value = row.get(key)
        if isinstance(value, int | float):
            values.append(float(value))
    return values


def _median_or_none(rows: list[dict[str, Any]], key: str) -> float | None:
    values = _finite_values(rows, key)
    return median(values) if values else None


def summarize(input_dir: Path, output: Path) -> Path:
    """Aggregate all per-mouse OOD comparison artifacts in ``input_dir``."""

    paths = sorted(
        path
        for path in input_dir.glob("*_ood_temporal_comparison.json")
        if not path.name.startswith("summary_")
    )
    if not paths:
        raise FileNotFoundError(f"No OOD comparison JSON files found in {input_dir}")

    rows = [json.loads(path.read_text()) for path in paths]
    positive_temporal_minus_summary = sum(
        row["temporal_minus_summary_correlation"] > 0 for row in rows
    )
    positive_temporal_minus_mean = sum(row["temporal_minus_mean"] > 0 for row in rows)
    positive_temporal_minus_scrambled = sum(
        row["temporal_minus_scrambled"] > 0 for row in rows
    )

    payload = {
        "dataset": "Dynamic Sensorium 2023 legacy released-response OOD dataset",
        "n_mice": len(rows),
        "input_dir": str(input_dir),
        "source_files": [str(path) for path in paths],
        "positive_temporal_minus_summary_count": positive_temporal_minus_summary,
        "positive_temporal_minus_mean_count": positive_temporal_minus_mean,
        "positive_temporal_minus_scrambled_count": positive_temporal_minus_scrambled,
        "median_summary_correlation": _median_or_none(rows, "summary_correlation"),
        "median_temporal_filterbank_correlation": _median_or_none(
            rows, "temporal_filterbank_correlation"
        ),
        "median_temporal_minus_summary_correlation": _median_or_none(
            rows, "temporal_minus_summary_correlation"
        ),
        "median_temporal_minus_mean": _median_or_none(rows, "temporal_minus_mean"),
        "median_temporal_minus_scrambled": _median_or_none(
            rows, "temporal_minus_scrambled"
        ),
        "reliability_estimable_count": sum(row["reliability_estimable"] for row in rows),
        "mis_passed_count": sum(row["mis_passed"] for row in rows),
        "interpretation": (
            "This aggregate is an OOD generalization check, not a mechanistic "
            "identification claim. Reliability remains non-estimable in these "
            "artifacts, and no structural or causal constraint is active."
        ),
        "rows": rows,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2))
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("results/dynamic_sensorium_ood"),
        help="Directory containing *_ood_temporal_comparison.json files.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "results/dynamic_sensorium_ood/"
            "summary_dynamic_sensorium_legacy_ood_temporal_comparison.json"
        ),
        help="Aggregate JSON artifact to write.",
    )
    args = parser.parse_args()
    output = summarize(args.input_dir, args.output)
    print(json.dumps({"output": str(output.resolve())}))


if __name__ == "__main__":
    main()
