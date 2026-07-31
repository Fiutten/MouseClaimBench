"""Run the transparent temporal-filterbank Dynamic Sensorium comparison.

This script compares the existing calibrated residual adapter using the
original summary descriptors against the richer temporal-filterbank descriptors.
It does not claim Sensorium state-of-the-art performance; the purpose is to
test whether explicit video timing adds reproducible signal inside the
MouseBrainBench/MIS evaluation contract.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from mousebrainbench.artifacts import code_revision
from mousebrainbench.benchmarks.sensorium_predictive_mis import run


DEFAULT_ALPHA_GRID = (0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0)


def _mouse_id(root: Path) -> str:
    """Keep historical result filenames stable across full Sensorium directory names."""

    return root.name.split("-Video-")[0].split("dynamic")[-1].split("-")[0]


def _load_metrics(path: Path) -> dict[str, object]:
    return json.loads(path.read_text())


def run_temporal_filterbank_comparison(
    extracted_root: Path,
    *,
    output_dir: Path,
    summary_output: Path,
    eval_tier: str,
    alpha_grid: tuple[float, ...],
    force: bool = False,
    git_revision: str | None = None,
) -> Path:
    """Execute per-mouse temporal comparisons and write a compact JSON summary.

    The revision is captured once before any tracked output is written so all
    per-mouse artifacts describe the same source state.
    """

    roots = sorted(path for path in extracted_root.glob("dynamic*") if path.is_dir())
    if not roots:
        raise FileNotFoundError(f"no Dynamic Sensorium directories found under {extracted_root}")
    output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    artifact_revision = git_revision or code_revision()

    for root in roots:
        mouse = f"dynamic{_mouse_id(root)}"
        summary_path = output_dir / f"{mouse}_oracle_calibrated_residual_mis.json"
        temporal_path = output_dir / f"{mouse}_oracle_temporal_filterbank_calibrated_residual_mis.json"

        if force or not summary_path.exists():
            run(
                root,
                output=summary_path,
                modality="dynamic",
                eval_tiers=(eval_tier,),
                feature_mode="summary",
                adapter="calibrated_residual_ridge",
                adapter_alpha_grid=alpha_grid,
                git_revision=artifact_revision,
            )
        if force or not temporal_path.exists():
            run(
                root,
                output=temporal_path,
                modality="dynamic",
                eval_tiers=(eval_tier,),
                feature_mode="temporal_filterbank",
                adapter="calibrated_residual_ridge",
                adapter_alpha_grid=alpha_grid,
                git_revision=artifact_revision,
            )

        summary_payload = _load_metrics(summary_path)
        temporal_payload = _load_metrics(temporal_path)
        summary_metrics = summary_payload["metrics"]
        temporal_metrics = temporal_payload["metrics"]
        rows.append(
            {
                "mouse": mouse,
                "summary_feature_mode_correlation": summary_metrics[
                    "calibrated_residual_ridge_correlation"
                ],
                "temporal_filterbank_correlation": temporal_metrics[
                    "calibrated_residual_ridge_correlation"
                ],
                "temporal_minus_summary_correlation": temporal_metrics[
                    "calibrated_residual_ridge_correlation"
                ]
                - summary_metrics["calibrated_residual_ridge_correlation"],
                "summary_minus_mean": summary_metrics["calibrated_residual_ridge_minus_mean"],
                "temporal_minus_mean": temporal_metrics["calibrated_residual_ridge_minus_mean"],
                "summary_minus_scrambled": summary_metrics[
                    "calibrated_residual_ridge_minus_scrambled"
                ],
                "temporal_minus_scrambled": temporal_metrics[
                    "calibrated_residual_ridge_minus_scrambled"
                ],
                "temporal_adapter_beta": temporal_payload["adapter"]["diagnostics"].get(
                    "adapter_beta"
                ),
                "temporal_adapter_alpha": temporal_payload["adapter"]["diagnostics"].get(
                    "adapter_alpha"
                ),
            }
        )

    summary = {
        "dataset": "Dynamic Sensorium 2023 public release",
        "eval_tier": eval_tier,
        "adapter": "calibrated_residual_ridge",
        "comparison": "summary stimulus descriptors vs temporal_filterbank descriptors",
        "n_mice": len(rows),
        "positive_temporal_minus_summary_count": int(
            sum(row["temporal_minus_summary_correlation"] > 0 for row in rows)
        ),
        "median_temporal_minus_summary_correlation": float(
            np.median([row["temporal_minus_summary_correlation"] for row in rows])
        ),
        "median_temporal_minus_mean": float(np.median([row["temporal_minus_mean"] for row in rows])),
        "median_temporal_minus_scrambled": float(
            np.median([row["temporal_minus_scrambled"] for row in rows])
        ),
        "rows": rows,
        "interpretation": (
            "Temporal descriptors are treated as a stronger transparent baseline. "
            "They improve most, not all, mice and must be retested on OOD/released-repeat data."
        ),
    }
    summary_output.parent.mkdir(parents=True, exist_ok=True)
    summary_output.write_text(json.dumps(summary, indent=2))
    return summary_output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--extracted-root",
        type=Path,
        default=Path("data/dynamic_sensorium/extracted"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/dynamic_sensorium_adapter"),
    )
    parser.add_argument(
        "--summary-output",
        type=Path,
        default=Path(
            "results/dynamic_sensorium_adapter/"
            "summary_dynamic_sensorium2023_temporal_filterbank_mis.json"
        ),
    )
    parser.add_argument("--eval-tier", default="oracle")
    parser.add_argument(
        "--alpha-grid",
        default=",".join(str(value) for value in DEFAULT_ALPHA_GRID),
        help="Comma-separated train-CV alpha candidates for the calibrated adapter.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Regenerate per-mouse artifacts even when output files already exist.",
    )
    parser.add_argument(
        "--git-revision",
        default=None,
        help="Explicit clean Git revision shared by every generated per-mouse artifact.",
    )
    args = parser.parse_args()
    alpha_grid = tuple(float(value) for value in args.alpha_grid.split(","))
    output = run_temporal_filterbank_comparison(
        args.extracted_root,
        output_dir=args.output_dir,
        summary_output=args.summary_output,
        eval_tier=args.eval_tier,
        alpha_grid=alpha_grid,
        force=args.force,
        git_revision=args.git_revision,
    )
    print(json.dumps({"output": str(output.resolve())}))


if __name__ == "__main__":
    main()
