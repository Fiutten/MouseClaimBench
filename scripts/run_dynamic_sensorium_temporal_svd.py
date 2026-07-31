"""Compare temporal-filterbank ridge against a train-only SVD temporal adapter.

This script is the next step after the transparent temporal-filterbank
baseline. It keeps the same MouseBrainBench/MIS contract and changes only the
adapter: ``temporal_svd_residual_ridge`` fits a low-dimensional temporal
stimulus subspace on the training split, calibrates the residual scale by
train-only CV, and evaluates held-out tiers without using their responses for
selection.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from mousebrainbench.benchmarks.sensorium_predictive_mis import run


DEFAULT_ALPHA_GRID = (0.1, 1.0, 3.0, 10.0, 30.0, 100.0)


def _stem(root: Path) -> str:
    return root.name.split("-Video-")[0]


def _load(path: Path) -> dict[str, object]:
    return json.loads(path.read_text())


def _median(values: list[float]) -> float:
    return float(np.median(values)) if values else 0.0


def run_temporal_svd_comparison(
    roots: list[Path],
    *,
    output_dir: Path,
    summary_output: Path,
    eval_tiers: tuple[str, ...],
    alpha_grid: tuple[float, ...],
    ood_gate: bool,
    git_revision: str | None = None,
    force: bool = False,
) -> Path:
    """Run per-mouse SVD adapter comparisons and write an aggregate summary."""

    if not roots:
        raise FileNotFoundError("no Dynamic Sensorium roots were provided")
    output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []

    for root in roots:
        mouse = _stem(root)
        output = output_dir / f"{mouse}_temporal_svd_residual_mis.json"
        if force or not output.exists():
            run(
                root,
                output=output,
                modality="dynamic",
                eval_tiers=eval_tiers,
                feature_mode="temporal_filterbank",
                adapter="temporal_svd_residual_ridge",
                adapter_alpha_grid=alpha_grid,
                has_ood_generalization_gate=ood_gate,
                git_revision=git_revision,
            )
        payload = _load(output)
        metrics = payload["metrics"]
        diagnostics = payload["adapter"]["diagnostics"]
        rows.append(
            {
                "mouse": mouse,
                "output": str(output),
                "temporal_svd_correlation": metrics[
                    "temporal_svd_residual_ridge_correlation"
                ],
                "temporal_svd_minus_mean": metrics[
                    "temporal_svd_residual_ridge_minus_mean"
                ],
                "temporal_svd_minus_scrambled": metrics[
                    "temporal_svd_residual_ridge_minus_scrambled"
                ],
                "best_model": metrics["best_model"],
                "selected_alpha": diagnostics.get("adapter_alpha"),
                "selected_beta": diagnostics.get("adapter_beta"),
                "selected_components": diagnostics.get("adapter_components"),
                "mis_passed": payload["mis"]["passed"],
                "reliability_estimable": metrics["reliability_estimable"],
            }
        )

    minus_mean = [float(row["temporal_svd_minus_mean"]) for row in rows]
    minus_scrambled = [float(row["temporal_svd_minus_scrambled"]) for row in rows]
    summary = {
        "dataset": "Dynamic Sensorium temporal SVD adapter comparison",
        "eval_tiers": list(eval_tiers),
        "adapter": "temporal_svd_residual_ridge",
        "n_mice": len(rows),
        "positive_temporal_svd_minus_mean_count": int(sum(value > 0 for value in minus_mean)),
        "positive_temporal_svd_minus_scrambled_count": int(
            sum(value > 0 for value in minus_scrambled)
        ),
        "median_temporal_svd_minus_mean": _median(minus_mean),
        "median_temporal_svd_minus_scrambled": _median(minus_scrambled),
        "reliability_estimable_count": int(sum(row["reliability_estimable"] for row in rows)),
        "mis_passed_count": int(sum(row["mis_passed"] for row in rows)),
        "rows": rows,
        "interpretation": (
            "The SVD adapter is a stronger temporal baseline, but still only "
            "predictive evidence. MIS should remain negative without repeat "
            "reliability or structural/causal constraints."
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
        required=True,
        help="Directory containing Dynamic Sensorium mouse directories.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--summary-output", type=Path, required=True)
    parser.add_argument(
        "--eval-tier",
        action="append",
        dest="eval_tiers",
        required=True,
        help="Held-out tier to evaluate. Repeat for multiple tiers.",
    )
    parser.add_argument(
        "--alpha-grid",
        default=",".join(str(value) for value in DEFAULT_ALPHA_GRID),
        help="Comma-separated alpha candidates selected by train-only CV.",
    )
    parser.add_argument("--git-revision", default=None)
    parser.add_argument("--ood-gate", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    roots = sorted(path for path in args.extracted_root.glob("dynamic*") if path.is_dir())
    output = run_temporal_svd_comparison(
        roots,
        output_dir=args.output_dir,
        summary_output=args.summary_output,
        eval_tiers=tuple(args.eval_tiers),
        alpha_grid=tuple(float(value) for value in args.alpha_grid.split(",")),
        ood_gate=args.ood_gate,
        git_revision=args.git_revision,
        force=args.force,
    )
    print(json.dumps({"output": str(output.resolve())}))


if __name__ == "__main__":
    main()
