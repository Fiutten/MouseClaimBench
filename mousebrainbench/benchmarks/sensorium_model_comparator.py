"""Compare Sensorium predictive artifacts without retraining models.

The comparator is intentionally conservative: it reads tracked benchmark
artifacts, aligns mice by their stable ``dynamic<id>`` identifier, and reports
paired deltas. It does not tune hyperparameters or reinterpret prediction as a
mechanistic claim.
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from statistics import median
from typing import Any


MODEL_ORDER = (
    "mean_response",
    "summary_adapter",
    "temporal_filterbank",
    "temporal_svd",
    "random_feature",
    "torch_mlp",
    "official_sensorium_bounded",
)


@dataclass(frozen=True)
class ModelObservation:
    """Single model result for one mouse and one evaluation cohort."""

    mouse: str
    model: str
    correlation: float
    minus_mean: float | None = None
    minus_scrambled: float | None = None
    reliability_estimable: bool = False
    mis_passed: bool = False


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def mouse_key(mouse: str) -> str:
    """Return the stable Dynamic Sensorium mouse id used across artifact names."""

    match = re.search(r"dynamic\d+", mouse)
    if not match:
        return mouse
    return match.group(0)


def _median(values: list[float]) -> float | None:
    return median(values) if values else None


def _round(value: float | None, digits: int = 6) -> float | None:
    return None if value is None else round(float(value), digits)


def _cohort_label(path: Path, fallback: str) -> str:
    name = path.name
    if "legacy_ood" in name:
        return "dynamic_sensorium_legacy_ood"
    if "dynamic_sensorium2023" in name:
        return "dynamic_sensorium2023_oracle"
    return fallback


def _read_temporal_summary(path: Path) -> dict[str, dict[str, ModelObservation]]:
    payload = _load_json(path)
    rows: dict[str, dict[str, ModelObservation]] = {}
    for row in payload["rows"]:
        mouse = str(row["mouse"])
        key = mouse_key(mouse)
        summary_corr = row.get("summary_correlation", row.get("summary_feature_mode_correlation"))
        temporal_corr = row["temporal_filterbank_correlation"]
        temporal_minus_mean = row["temporal_minus_mean"]
        mean_corr = float(temporal_corr) - float(temporal_minus_mean)
        rows.setdefault(key, {})
        rows[key]["mean_response"] = ModelObservation(
            mouse=mouse,
            model="mean_response",
            correlation=mean_corr,
        )
        rows[key]["summary_adapter"] = ModelObservation(
            mouse=mouse,
            model="summary_adapter",
            correlation=float(summary_corr),
            minus_mean=row.get("summary_minus_mean"),
            minus_scrambled=row.get("summary_minus_scrambled"),
            reliability_estimable=bool(row.get("reliability_estimable", False)),
            mis_passed=bool(row.get("mis_passed", False)),
        )
        rows[key]["temporal_filterbank"] = ModelObservation(
            mouse=mouse,
            model="temporal_filterbank",
            correlation=float(temporal_corr),
            minus_mean=float(temporal_minus_mean),
            minus_scrambled=row.get("temporal_minus_scrambled"),
            reliability_estimable=bool(row.get("reliability_estimable", False)),
            mis_passed=bool(row.get("mis_passed", False)),
        )
    return rows


def _merge_svd_summary(
    rows: dict[str, dict[str, ModelObservation]],
    path: Path,
) -> None:
    payload = _load_json(path)
    for row in payload["rows"]:
        mouse = str(row["mouse"])
        key = mouse_key(mouse)
        rows.setdefault(key, {})
        rows[key]["temporal_svd"] = ModelObservation(
            mouse=mouse,
            model="temporal_svd",
            correlation=float(row["temporal_svd_correlation"]),
            minus_mean=float(row["temporal_svd_minus_mean"]),
            minus_scrambled=float(row["temporal_svd_minus_scrambled"]),
            reliability_estimable=bool(row.get("reliability_estimable", False)),
            mis_passed=bool(row.get("mis_passed", False)),
        )


def _merge_random_feature_summary(
    rows: dict[str, dict[str, ModelObservation]],
    path: Path,
) -> None:
    payload = _load_json(path)
    for row in payload["rows"]:
        mouse = str(row["mouse"])
        key = mouse_key(mouse)
        rows.setdefault(key, {})
        rows[key]["random_feature"] = ModelObservation(
            mouse=mouse,
            model="random_feature",
            correlation=float(row["random_feature_correlation"]),
            minus_mean=float(row["random_feature_minus_mean"]),
            minus_scrambled=float(row["random_feature_minus_scrambled"]),
            reliability_estimable=bool(row.get("reliability_estimable", False)),
            mis_passed=bool(row.get("mis_passed", False)),
        )


def _merge_torch_mlp_summary(
    rows: dict[str, dict[str, ModelObservation]],
    path: Path,
) -> None:
    payload = _load_json(path)
    for row in payload["rows"]:
        mouse = str(row["mouse"])
        key = mouse_key(mouse)
        rows.setdefault(key, {})
        rows[key]["torch_mlp"] = ModelObservation(
            mouse=mouse,
            model="torch_mlp",
            correlation=float(row["torch_mlp_correlation"]),
            minus_mean=float(row["torch_mlp_minus_mean"]),
            minus_scrambled=None,
            reliability_estimable=False,
            mis_passed=False,
        )


def _merge_official_sensorium_summary(
    rows: dict[str, dict[str, ModelObservation]],
    path: Path,
) -> None:
    """Merge bounded official Sensorium architecture results when available."""

    payload = _load_json(path)
    for row in payload["rows"]:
        mouse = str(row["mouse"])
        key = mouse_key(mouse)
        if row.get("official_sensorium_correlation") is None:
            continue
        rows.setdefault(key, {})
        rows[key]["official_sensorium_bounded"] = ModelObservation(
            mouse=mouse,
            model="official_sensorium_bounded",
            correlation=float(row["official_sensorium_correlation"]),
            minus_mean=row.get("official_sensorium_minus_mean"),
            minus_scrambled=None,
            reliability_estimable=False,
            mis_passed=False,
        )


def _pairwise(rows: dict[str, dict[str, ModelObservation]], left: str, right: str) -> dict[str, Any]:
    deltas: list[float] = []
    paired: list[dict[str, Any]] = []
    for mouse in sorted(rows):
        models = rows[mouse]
        if left not in models or right not in models:
            continue
        delta = models[right].correlation - models[left].correlation
        deltas.append(delta)
        paired.append(
            {
                "mouse": mouse,
                "left": left,
                "right": right,
                "left_correlation": models[left].correlation,
                "right_correlation": models[right].correlation,
                "delta": delta,
            }
        )
    return {
        "left": left,
        "right": right,
        "n_paired": len(deltas),
        "right_wins": sum(delta > 0 for delta in deltas),
        "left_wins": sum(delta < 0 for delta in deltas),
        "ties": sum(delta == 0 for delta in deltas),
        "median_delta": _median(deltas),
        "mean_delta": sum(deltas) / len(deltas) if deltas else None,
        "rows": paired,
    }


def _best_models(rows: dict[str, dict[str, ModelObservation]]) -> list[dict[str, Any]]:
    best_rows: list[dict[str, Any]] = []
    for mouse in sorted(rows):
        candidates = [rows[mouse][model] for model in MODEL_ORDER if model in rows[mouse]]
        if not candidates:
            continue
        best = max(candidates, key=lambda item: item.correlation)
        best_rows.append(
            {
                "mouse": mouse,
                "best_model": best.model,
                "best_correlation": best.correlation,
                "available_models": [item.model for item in candidates],
            }
        )
    return best_rows


def _model_table(rows: dict[str, dict[str, ModelObservation]]) -> list[dict[str, Any]]:
    table: list[dict[str, Any]] = []
    for mouse in sorted(rows):
        row: dict[str, Any] = {"mouse": mouse}
        for model in MODEL_ORDER:
            row[model] = rows[mouse][model].correlation if model in rows[mouse] else None
        table.append(row)
    return table


def _evidence_label(
    *,
    cohort_name: str,
    rows: dict[str, dict[str, ModelObservation]],
    svd_vs_mean: dict[str, Any],
    svd_vs_temporal: dict[str, Any],
) -> str:
    n = len(rows)
    mis_passed = sum(
        observation.mis_passed
        for models in rows.values()
        for observation in models.values()
        if observation.model != "mean_response"
    )
    reliability_estimable = sum(
        observation.reliability_estimable
        for models in rows.values()
        for observation in models.values()
        if observation.model != "mean_response"
    )
    svd_predictive = (
        svd_vs_mean["n_paired"] == n
        and svd_vs_mean["right_wins"] == n
        and (svd_vs_mean["median_delta"] or 0.0) > 0.0
    )
    svd_method_gain = (
        svd_vs_temporal["n_paired"] == n
        and svd_vs_temporal["right_wins"] == n
        and (svd_vs_temporal["median_delta"] or 0.0) > 0.0
    )
    is_ood = "ood" in cohort_name.lower()
    if svd_predictive and svd_method_gain and is_ood and mis_passed == 0:
        return "positive_ood_prediction_without_mechanistic_identifiability"
    if svd_predictive and mis_passed == 0:
        return "positive_prediction_without_mechanistic_identifiability"
    if reliability_estimable == 0 and mis_passed == 0:
        return "predictive_benchmark_requires_reliability_or_causal_constraints"
    return "mixed_evidence"


def compare_sensorium_artifacts(
    *,
    temporal_summary: Path,
    svd_summary: Path,
    random_feature_summary: Path | None = None,
    torch_mlp_summary: Path | None = None,
    official_sensorium_summary: Path | None = None,
    cohort_name: str | None = None,
) -> dict[str, Any]:
    """Compare summary/temporal/SVD artifacts for one Sensorium cohort."""

    rows = _read_temporal_summary(temporal_summary)
    _merge_svd_summary(rows, svd_summary)
    if random_feature_summary is not None:
        _merge_random_feature_summary(rows, random_feature_summary)
    if torch_mlp_summary is not None:
        _merge_torch_mlp_summary(rows, torch_mlp_summary)
    if official_sensorium_summary is not None and official_sensorium_summary.exists():
        _merge_official_sensorium_summary(rows, official_sensorium_summary)
    cohort = cohort_name or _cohort_label(svd_summary, svd_summary.stem)
    pairwise = {
        "summary_adapter_vs_temporal_filterbank": _pairwise(
            rows, "summary_adapter", "temporal_filterbank"
        ),
        "mean_response_vs_temporal_filterbank": _pairwise(
            rows, "mean_response", "temporal_filterbank"
        ),
        "mean_response_vs_temporal_svd": _pairwise(rows, "mean_response", "temporal_svd"),
        "temporal_filterbank_vs_temporal_svd": _pairwise(
            rows, "temporal_filterbank", "temporal_svd"
        ),
        "temporal_svd_vs_random_feature": _pairwise(
            rows, "temporal_svd", "random_feature"
        ),
        "temporal_svd_vs_torch_mlp": _pairwise(rows, "temporal_svd", "torch_mlp"),
        "torch_mlp_vs_official_sensorium_bounded": _pairwise(
            rows, "torch_mlp", "official_sensorium_bounded"
        ),
        "temporal_svd_vs_official_sensorium_bounded": _pairwise(
            rows, "temporal_svd", "official_sensorium_bounded"
        ),
    }
    best_rows = _best_models(rows)
    model_win_counts: dict[str, int] = {}
    for row in best_rows:
        model_win_counts[row["best_model"]] = model_win_counts.get(row["best_model"], 0) + 1

    return {
        "cohort": cohort,
        "n_mice": len(rows),
        "temporal_summary": str(temporal_summary),
        "svd_summary": str(svd_summary),
        "random_feature_summary": str(random_feature_summary) if random_feature_summary else None,
        "torch_mlp_summary": str(torch_mlp_summary) if torch_mlp_summary else None,
        "official_sensorium_summary": (
            str(official_sensorium_summary) if official_sensorium_summary else None
        ),
        "models": list(MODEL_ORDER),
        "model_rows": _model_table(rows),
        "best_models": best_rows,
        "model_win_counts": model_win_counts,
        "pairwise": pairwise,
        "mis_passed_count": sum(
            observation.mis_passed
            for models in rows.values()
            for observation in models.values()
            if observation.model != "mean_response"
        ),
        "reliability_estimable_count": sum(
            observation.reliability_estimable
            for models in rows.values()
            for observation in models.values()
            if observation.model != "mean_response"
        ),
        "evidence_label": _evidence_label(
            cohort_name=cohort,
            rows=rows,
            svd_vs_mean=pairwise["mean_response_vs_temporal_svd"],
            svd_vs_temporal=pairwise["temporal_filterbank_vs_temporal_svd"],
        ),
        "interpretation": (
            "Predictive deltas are paired by mouse. A positive SVD result is "
            "treated as predictive/OOD evidence only; MIS and reliability fields "
            "control whether the result can support a mechanistic claim."
        ),
    }


def _format_count(pairwise: dict[str, Any]) -> str:
    return f"{pairwise['right_wins']}/{pairwise['n_paired']}"


def _markdown_for_cohort(cohort: dict[str, Any]) -> str:
    pairwise = cohort["pairwise"]
    lines = [
        f"## {cohort['cohort']}",
        "",
        f"- Ratones comparados: `{cohort['n_mice']}`",
        f"- Evidencia: `{cohort['evidence_label']}`",
        f"- MIS passed: `{cohort['mis_passed_count']}`",
        f"- Reliability estimable: `{cohort['reliability_estimable_count']}`",
        "",
        "| Comparación | Victorias modelo derecho | Mediana delta | Media delta |",
        "|---|---:|---:|---:|",
    ]
    for key, item in pairwise.items():
        lines.append(
            "| "
            f"{key} | `{_format_count(item)}` | "
            f"`{_round(item['median_delta'])}` | `{_round(item['mean_delta'])}` |"
        )
    lines.extend(
        [
            "",
            (
            "| Mouse | Mean | Summary | Temporal filterbank | Temporal SVD | "
            "Random feature | Torch MLP | Official bounded | Best |"
        ),
            "|---|---:|---:|---:|---:|---:|---:|---:|---|",
        ]
    )
    best_by_mouse = {row["mouse"]: row["best_model"] for row in cohort["best_models"]}
    for row in cohort["model_rows"]:
        lines.append(
            "| "
            f"`{row['mouse']}` | "
            f"`{_round(row['mean_response'])}` | "
            f"`{_round(row['summary_adapter'])}` | "
            f"`{_round(row['temporal_filterbank'])}` | "
            f"`{_round(row['temporal_svd'])}` | "
            f"`{_round(row['random_feature'])}` | "
            f"`{_round(row['torch_mlp'])}` | "
            f"`{_round(row.get('official_sensorium_bounded'))}` | "
            f"`{best_by_mouse.get(row['mouse'])}` |"
        )
    return "\n".join(lines)


def write_comparison_outputs(payload: dict[str, Any], output_json: Path, output_md: Path) -> None:
    """Write machine-readable and human-readable comparison artifacts."""

    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(payload, indent=2))
    lines = [
        "# Dynamic Sensorium Model Comparator",
        "",
        payload["interpretation"],
        "",
    ]
    for cohort in payload["cohorts"]:
        lines.append(_markdown_for_cohort(cohort))
        lines.append("")
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_md.write_text("\n".join(lines).rstrip() + "\n")


def compare_default_dynamic_sensorium(
    *,
    output_json: Path,
    output_md: Path,
) -> dict[str, Any]:
    """Compare the current tracked Dynamic Sensorium main and OOD artifacts."""

    cohorts = [
        compare_sensorium_artifacts(
            temporal_summary=Path(
                "results/dynamic_sensorium_adapter/"
                "summary_dynamic_sensorium2023_temporal_filterbank_mis.json"
            ),
            svd_summary=Path(
                "results/dynamic_sensorium_temporal_svd/"
                "summary_dynamic_sensorium2023_temporal_svd.json"
            ),
            random_feature_summary=Path(
                "results/dynamic_sensorium_random_feature/"
                "summary_dynamic_sensorium2023_random_feature.json"
            ),
            torch_mlp_summary=Path(
                "results/dynamic_sensorium_torch_mlp/"
                "summary_dynamic_sensorium2023_torch_mlp.json"
            ),
            official_sensorium_summary=Path(
                "results/sensorium_official_baseline_audit/"
                "official_trained_baseline_summary.json"
            ),
            cohort_name="dynamic_sensorium2023_oracle",
        ),
        compare_sensorium_artifacts(
            temporal_summary=Path(
                "results/dynamic_sensorium_ood/"
                "summary_dynamic_sensorium_legacy_ood_temporal_comparison.json"
            ),
            svd_summary=Path(
                "results/dynamic_sensorium_temporal_svd/"
                "summary_dynamic_sensorium_legacy_ood_temporal_svd.json"
            ),
            random_feature_summary=Path(
                "results/dynamic_sensorium_random_feature/"
                "summary_dynamic_sensorium_legacy_ood_random_feature.json"
            ),
            torch_mlp_summary=Path(
                "results/dynamic_sensorium_torch_mlp/"
                "summary_dynamic_sensorium_legacy_ood_torch_mlp.json"
            ),
            cohort_name="dynamic_sensorium_legacy_ood",
        ),
    ]
    payload = {
        "comparison": "dynamic_sensorium_predictive_model_comparator",
        "cohorts": cohorts,
        "interpretation": (
            "This comparator asks whether progressively stronger transparent "
            "models improve held-out prediction and whether that improvement is "
            "allowed to count as mechanistic evidence. Current Dynamic Sensorium "
            "artifacts remain predictive evidence only because MIS does not pass."
        ),
    }
    write_comparison_outputs(payload, output_json, output_md)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("results/dynamic_sensorium_model_comparator/summary.json"),
    )
    parser.add_argument(
        "--output-md",
        type=Path,
        default=Path("results/dynamic_sensorium_model_comparator/summary.md"),
    )
    args = parser.parse_args()
    payload = compare_default_dynamic_sensorium(
        output_json=args.output_json,
        output_md=args.output_md,
    )
    print(json.dumps({"output_json": str(args.output_json), "n_cohorts": len(payload["cohorts"])}))


if __name__ == "__main__":
    main()
