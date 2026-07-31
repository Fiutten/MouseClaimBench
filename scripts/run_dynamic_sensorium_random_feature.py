"""Run a stronger local nonlinear baseline on Dynamic Sensorium artifacts.

The model is deliberately labelled as a local nonlinear baseline, not an
official Sensorium model: it uses train-only random Fourier features plus the
same calibrated residual ridge contract used by the rest of MouseBrainBench.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import median
from typing import Any

from mousebrainbench.benchmarks.sensorium_predictive_mis import run


def _stem(root: Path) -> str:
    return root.name.split("-Video-")[0]


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _median(values: list[float]) -> float | None:
    return median(values) if values else None


def run_random_feature_comparison(
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
    """Run per-mouse random-feature residual baselines and aggregate results."""

    if not roots:
        raise FileNotFoundError("no Dynamic Sensorium roots were provided")
    output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for root in roots:
        mouse = _stem(root)
        output = output_dir / f"{mouse}_random_feature_residual_mis.json"
        if force or not output.exists():
            run(
                root,
                output=output,
                modality="dynamic",
                eval_tiers=eval_tiers,
                feature_mode="temporal_filterbank",
                adapter="random_feature_residual_ridge",
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
                "random_feature_correlation": metrics[
                    "random_feature_residual_ridge_correlation"
                ],
                "random_feature_minus_mean": metrics[
                    "random_feature_residual_ridge_minus_mean"
                ],
                "random_feature_minus_scrambled": metrics[
                    "random_feature_residual_ridge_minus_scrambled"
                ],
                "best_model": metrics["best_model"],
                "selected_alpha": diagnostics.get("adapter_alpha"),
                "selected_beta": diagnostics.get("adapter_beta"),
                "selected_components": diagnostics.get("adapter_components"),
                "selected_gamma": diagnostics.get("adapter_gamma"),
                "mis_passed": payload["mis"]["passed"],
                "reliability_estimable": metrics["reliability_estimable"],
            }
        )

    minus_mean = [float(row["random_feature_minus_mean"]) for row in rows]
    minus_scrambled = [float(row["random_feature_minus_scrambled"]) for row in rows]
    summary = {
        "dataset": "Dynamic Sensorium random-feature residual baseline comparison",
        "eval_tiers": list(eval_tiers),
        "adapter": "random_feature_residual_ridge",
        "n_mice": len(rows),
        "positive_random_feature_minus_mean_count": int(
            sum(value > 0 for value in minus_mean)
        ),
        "positive_random_feature_minus_scrambled_count": int(
            sum(value > 0 for value in minus_scrambled)
        ),
        "median_random_feature_minus_mean": _median(minus_mean),
        "median_random_feature_minus_scrambled": _median(minus_scrambled),
        "reliability_estimable_count": int(sum(row["reliability_estimable"] for row in rows)),
        "mis_passed_count": int(sum(row["mis_passed"] for row in rows)),
        "rows": rows,
        "interpretation": (
            "Random Fourier residual ridge is a stronger local nonlinear baseline, "
            "not an official Sensorium/SOTA model. It remains predictive evidence "
            "unless reliability and mechanistic constraints pass."
        ),
    }
    summary_output.parent.mkdir(parents=True, exist_ok=True)
    summary_output.write_text(json.dumps(summary, indent=2))
    return summary_output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--extracted-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--summary-output", type=Path, required=True)
    parser.add_argument("--eval-tier", action="append", dest="eval_tiers", required=True)
    parser.add_argument("--alpha-grid", default="1,10,100")
    parser.add_argument("--git-revision", default=None)
    parser.add_argument("--ood-gate", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    roots = sorted(path for path in args.extracted_root.glob("dynamic*") if path.is_dir())
    output = run_random_feature_comparison(
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
