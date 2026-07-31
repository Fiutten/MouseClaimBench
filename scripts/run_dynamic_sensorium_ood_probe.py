"""Run a released-response OOD probe on the legacy Dynamic Sensorium dataset.

The legacy Sensorium 2023 five-mouse release includes non-zero responses for
``live_test_*`` and ``final_test_*`` tiers. This script uses those tiers as an
explicit out-of-distribution gate for the transparent MouseBrainBench predictive
benchmark. It intentionally keeps the model simple; the scientific question is
whether the evaluation framework distinguishes in-distribution prediction,
OOD generalization, and mechanistic evidence.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from mousebrainbench.benchmarks.sensorium_predictive_mis import run


DEFAULT_ALPHA_GRID = (0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0)
DEFAULT_OOD_TIERS = (
    "live_test_main",
    "live_test_bonus",
    "final_test_main",
    "final_test_bonus",
)


def run_ood_probe(
    root: Path,
    *,
    output_dir: Path,
    alpha_grid: tuple[float, ...],
    eval_tiers: tuple[str, ...] = DEFAULT_OOD_TIERS,
    git_revision: str | None = None,
) -> Path:
    """Run summary and temporal-filterbank OOD benchmarks for one mouse.

    ``git_revision`` is passed through explicitly so that both child benchmark
    artifacts are stamped with the same code revision. Without this, the first
    write can make the working tree dirty before the second child run records
    its provenance.
    """

    output_dir.mkdir(parents=True, exist_ok=True)
    stem = root.name.split("-Video-")[0]
    summary_output = output_dir / f"{stem}_ood_summary_calibrated_residual_mis.json"
    temporal_output = output_dir / f"{stem}_ood_temporal_filterbank_calibrated_residual_mis.json"
    run(
        root,
        output=summary_output,
        modality="dynamic",
        eval_tiers=eval_tiers,
        feature_mode="summary",
        adapter="calibrated_residual_ridge",
        adapter_alpha_grid=alpha_grid,
        has_ood_generalization_gate=True,
        git_revision=git_revision,
    )
    run(
        root,
        output=temporal_output,
        modality="dynamic",
        eval_tiers=eval_tiers,
        feature_mode="temporal_filterbank",
        adapter="calibrated_residual_ridge",
        adapter_alpha_grid=alpha_grid,
        has_ood_generalization_gate=True,
        git_revision=git_revision,
    )

    summary_payload = json.loads(summary_output.read_text())
    temporal_payload = json.loads(temporal_output.read_text())
    summary_metrics = summary_payload["metrics"]
    temporal_metrics = temporal_payload["metrics"]
    comparison = {
        "dataset": "Dynamic Sensorium 2023 legacy released-response OOD dataset",
        "mouse": stem,
        "eval_tiers": list(eval_tiers),
        "summary_output": str(summary_output),
        "temporal_output": str(temporal_output),
        "summary_correlation": summary_metrics["calibrated_residual_ridge_correlation"],
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
        "reliability_estimable": temporal_metrics["reliability_estimable"],
        "mis_passed": temporal_payload["mis"]["passed"],
        "interpretation": (
            "Positive OOD prediction is evidence of generalization, not sufficient "
            "mechanistic identification without repeat reliability or structural/causal constraints."
        ),
    }
    comparison_output = output_dir / f"{stem}_ood_temporal_comparison.json"
    comparison_output.write_text(json.dumps(comparison, indent=2))
    return comparison_output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("results/dynamic_sensorium_ood"))
    parser.add_argument(
        "--alpha-grid",
        default=",".join(str(value) for value in DEFAULT_ALPHA_GRID),
        help="Comma-separated train-CV alpha candidates for the calibrated adapter.",
    )
    parser.add_argument(
        "--eval-tier",
        action="append",
        dest="eval_tiers",
        default=None,
        help="OOD tier to evaluate. Defaults to all released live/final OOD tiers.",
    )
    parser.add_argument(
        "--git-revision",
        default=None,
        help="Explicit code revision to stamp into generated benchmark artifacts.",
    )
    args = parser.parse_args()
    output = run_ood_probe(
        args.root,
        output_dir=args.output_dir,
        alpha_grid=tuple(float(value) for value in args.alpha_grid.split(",")),
        eval_tiers=tuple(args.eval_tiers) if args.eval_tiers else DEFAULT_OOD_TIERS,
        git_revision=args.git_revision,
    )
    print(json.dumps({"output": str(output.resolve())}))


if __name__ == "__main__":
    main()
