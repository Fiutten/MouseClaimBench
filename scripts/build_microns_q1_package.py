"""Build manuscript-grade MICRONS Q1 tables and bootstrap stability checks.

This script summarizes the internally reproduced MICRONS stratified signal
without changing the underlying tests. The primary endpoint was fixed after the
discovery analysis and evaluated in two non-overlapping hold-outs: ``all_pairs`` with
``readout_location`` similarity. Bootstrap intervals use a unit-cluster weighted
bootstrap over the directed pair frame, avoiding naive pair-level independence.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from mousebrainbench import __version__
from mousebrainbench.artifacts import code_revision
from run_microns_stratified_structure_function import (  # noqa: PLC0415
    _add_metric_columns,
    _build_pair_frame,
    _safe_cut_from_reference,
)
from run_microns_structure_function_pilot import _prepare_edges, _prepare_units  # noqa: PLC0415


DEFAULT_OUTPUT = Path("results/microns_q1_package/summary.json")
DEFAULT_MARKDOWN = Path("results/microns_q1_package/summary.md")
PRIMARY_METRIC = "readout_location"
PRIMARY_STRATUM = "all_pairs"


@dataclass(frozen=True)
class CohortSpec:
    """Input and result files for one MICRONS discovery or hold-out cohort."""

    name: str
    units: Path
    edges: Path
    stratified_result: Path


DEFAULT_COHORTS = (
    CohortSpec(
        name="discovery",
        units=Path("data/microns/expanded/dt_coreg_units_v1507_sample1000.csv"),
        edges=Path("data/microns/expanded/dt_coreg_edges_v1507_sample1000.csv"),
        stratified_result=Path("results/microns_structure_function_pilot/stratified_summary.json"),
    ),
    CohortSpec(
        name="holdout_offset1000",
        units=Path("data/microns/expanded/dt_coreg_units_v1507_holdout_offset1000.csv"),
        edges=Path("data/microns/expanded/dt_coreg_edges_v1507_holdout_offset1000.csv"),
        stratified_result=Path(
            "results/microns_structure_function_pilot/stratified_holdout_offset1000_summary.json"
        ),
    ),
    CohortSpec(
        name="holdout_offset2000",
        units=Path("data/microns/expanded/dt_coreg_units_v1507_holdout_offset2000.csv"),
        edges=Path("data/microns/expanded/dt_coreg_edges_v1507_holdout_offset2000.csv"),
        stratified_result=Path(
            "results/microns_structure_function_pilot/stratified_holdout_offset2000_summary.json"
        ),
    ),
)


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _primary_test(stratified: dict[str, Any]) -> dict[str, Any]:
    """Return the primary all-pairs/readout-location test from a stratified result."""

    for row in stratified["all_tests"]:
        if row["stratum"] == PRIMARY_STRATUM and row["metric"] == PRIMARY_METRIC:
            return row
    raise ValueError(f"Missing primary test {PRIMARY_STRATUM}/{PRIMARY_METRIC}")


def _weighted_mean(values: np.ndarray, weights: np.ndarray) -> float:
    mask = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    if not np.any(mask):
        return float("nan")
    return float(np.average(values[mask], weights=weights[mask]))


def _matched_delta(frame: pd.DataFrame, metric: str, weights: np.ndarray, group_col: str) -> float:
    """Compute a deterministic matched-null delta under bootstrap weights."""

    connected_mask = frame["connected"].to_numpy(bool)
    observed = _weighted_mean(frame.loc[connected_mask, metric].to_numpy(float), weights[connected_mask])
    if math.isnan(observed):
        return float("nan")
    connected = frame.loc[connected_mask, [group_col]].copy()
    connected["_weight"] = weights[connected_mask]
    unconnected = frame.loc[~connected_mask, [group_col, metric]].copy()
    unconnected["_weight"] = weights[~connected_mask]
    connected_counts = connected.groupby(group_col, dropna=True)["_weight"].sum()
    null_chunks: list[float] = []
    null_weights: list[float] = []
    for group, group_weight in connected_counts.items():
        pool = unconnected[unconnected[group_col] == group]
        if pool.empty:
            pool = unconnected
        pool_mean = _weighted_mean(pool[metric].to_numpy(float), pool["_weight"].to_numpy(float))
        if not math.isnan(pool_mean) and group_weight > 0:
            null_chunks.append(pool_mean)
            null_weights.append(float(group_weight))
    if not null_chunks:
        return float("nan")
    matched_null = float(np.average(np.asarray(null_chunks), weights=np.asarray(null_weights)))
    return observed - matched_null


def _unit_bootstrap(
    frame: pd.DataFrame,
    *,
    roots: np.ndarray,
    samples: int,
    seed: int,
) -> dict[str, list[float] | int]:
    """Compute a weighted unit-cluster bootstrap over the directed pair frame.

    Units are resampled with replacement. Directed pair weights are induced by
    the multiplicities of their pre- and post-synaptic roots in each unit sample.
    The implementation therefore reweights the existing pair frame rather than
    physically rebuilding every eligible pair for each bootstrap sample.
    """

    rng = np.random.default_rng(seed)
    pre_roots = frame["pre_root"].to_numpy(np.int64)
    post_roots = frame["post_root"].to_numpy(np.int64)
    distance_deltas: list[float] = []
    degree_deltas: list[float] = []
    for _ in range(samples):
        sampled = rng.choice(roots, size=len(roots), replace=True)
        counts = pd.Series(sampled).value_counts()
        pre_counts = pd.Series(pre_roots).map(counts).fillna(0).to_numpy(float)
        post_counts = pd.Series(post_roots).map(counts).fillna(0).to_numpy(float)
        weights = pre_counts * post_counts
        if not np.any(weights > 0):
            continue
        distance_deltas.append(
            _matched_delta(frame, PRIMARY_METRIC, weights, "distance_bin")
        )
        degree_deltas.append(
            _matched_delta(frame, PRIMARY_METRIC, weights, "degree_bin")
        )
    return {
        "samples_requested": samples,
        "samples_usable": len(distance_deltas),
        "distance_matched_deltas": distance_deltas,
        "degree_matched_deltas": degree_deltas,
    }


def _interval(values: list[float]) -> dict[str, float | int]:
    array = np.asarray([value for value in values if math.isfinite(value)], dtype=float)
    if len(array) == 0:
        return {"n": 0, "median": float("nan"), "ci95_low": float("nan"), "ci95_high": float("nan")}
    return {
        "n": int(len(array)),
        "median": float(np.median(array)),
        "ci95_low": float(np.quantile(array, 0.025)),
        "ci95_high": float(np.quantile(array, 0.975)),
    }


def _cohort_summary(spec: CohortSpec, *, bootstrap_samples: int, seed: int) -> dict[str, Any]:
    """Build one cohort's manuscript table row and bootstrap intervals."""

    units = _prepare_units(spec.units)
    edges = _prepare_edges(spec.edges, set(units["pt_root_id"].astype("int64")))
    frame = _build_pair_frame(units, edges)
    _add_metric_columns(frame, units)
    unconnected = frame[~frame["connected"]]
    frame["distance_bin"] = _safe_cut_from_reference(frame["distance"], unconnected["distance"], 8)
    frame["pre_degree_bin"] = _safe_cut_from_reference(frame["pre_degree"], unconnected["pre_degree"], 5)
    frame["post_degree_bin"] = _safe_cut_from_reference(frame["post_degree"], unconnected["post_degree"], 5)
    frame["degree_bin"] = (
        frame["pre_degree_bin"].astype(str) + "|" + frame["post_degree_bin"].astype(str)
    )
    stratified = _load_json(spec.stratified_result)
    primary = _primary_test(stratified)
    bootstrap = _unit_bootstrap(
        frame,
        roots=units["pt_root_id"].astype("int64").to_numpy(),
        samples=bootstrap_samples,
        seed=seed,
    )
    return {
        "cohort": spec.name,
        "units": str(spec.units),
        "edges": str(spec.edges),
        "stratified_result": str(spec.stratified_result),
        "n_units": int(len(units)),
        "n_synapses_loaded": int(len(edges)),
        "n_connected_edge_pairs": int(stratified["n_connected_edge_pairs"]),
        "n_candidate_pairs": int(stratified["n_candidate_pairs"]),
        "n_statistical_tests": int(stratified["n_statistical_tests"]),
        "n_confirmed_positive_after_fdr": int(stratified["n_confirmed_positive_after_fdr"]),
        "primary_test": {
            "stratum": primary["stratum"],
            "metric": primary["metric"],
            "n_connected_pairs": int(primary["n_connected_pairs"]),
            "distance_matched_delta": float(primary["distance_matched_delta"]),
            "distance_matched_q_one_sided": float(primary["distance_matched_q_one_sided"]),
            "degree_matched_delta": float(primary["degree_matched_delta"]),
            "degree_matched_q_one_sided": float(primary["degree_matched_q_one_sided"]),
            "confirmed_positive_after_fdr": bool(primary["confirmed_positive_after_fdr"]),
        },
        "unit_bootstrap": {
            "samples_requested": bootstrap["samples_requested"],
            "samples_usable": bootstrap["samples_usable"],
            "distance_matched_delta": _interval(bootstrap["distance_matched_deltas"]),
            "degree_matched_delta": _interval(bootstrap["degree_matched_deltas"]),
        },
    }


def run(
    *,
    output: Path = DEFAULT_OUTPUT,
    markdown: Path = DEFAULT_MARKDOWN,
    bootstrap_samples: int = 300,
    seed: int = 71,
) -> dict[str, Any]:
    """Build internally reproduced MICRONS package artifacts."""

    cohorts = [
        _cohort_summary(spec, bootstrap_samples=bootstrap_samples, seed=seed + idx)
        for idx, spec in enumerate(DEFAULT_COHORTS)
    ]
    package_ready = all(
        cohort["primary_test"]["confirmed_positive_after_fdr"]
        and cohort["unit_bootstrap"]["distance_matched_delta"]["ci95_low"] > 0
        and cohort["unit_bootstrap"]["degree_matched_delta"]["ci95_low"] > 0
        for cohort in cohorts
    )
    payload = {
        "version": __version__,
        "git_revision": code_revision(),
        "analysis": "microns_q1_internally_reproduced_structure_function_package",
        "primary_endpoint": f"{PRIMARY_STRATUM}/{PRIMARY_METRIC}",
        "bootstrap": {
            "cluster": "unit",
            "samples": bootstrap_samples,
            "seed": seed,
        },
        "q1_package_ready": bool(package_ready),
        "cohorts": cohorts,
        "claims_allowed": [
            "Internally reproduced local MICRONS structure-function association "
            "across two non-overlapping hold-out windows from the same resource.",
            "Connected pairs show closer readout-location similarity than distance- and degree-matched controls.",
            "MouseBrainBench provides a reproducible claim-audit benchmark for partial mouse-brain digital models.",
        ],
        "claims_blocked": [
            "Causal mechanism.",
            "Whole-brain mouse digital twin.",
            "Behavioral digital twin.",
            "Sensorium SOTA predictor.",
        ],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2))
    _write_markdown(payload, markdown)
    return payload


def _round(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def _write_markdown(payload: dict[str, Any], markdown: Path) -> None:
    lines = [
        "# MICRONS Q1 Internally Reproduced Structure-Function Package",
        "",
        f"- Primary endpoint: `{payload['primary_endpoint']}`",
        "- Endpoint status: fixed after discovery and evaluated in two non-overlapping hold-outs.",
        f"- Q1 package ready: `{payload['q1_package_ready']}`",
        f"- Bootstrap cluster: `{payload['bootstrap']['cluster']}`; weighted over the directed pair frame.",
        f"- Bootstrap samples: `{payload['bootstrap']['samples']}`",
        "",
        "## Definitive Cohort Table",
        "",
        "| Cohort | Units | Synapses | Edge pairs | Confirmed tests | Distance delta | Distance q | Degree delta | Degree q |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for cohort in payload["cohorts"]:
        primary = cohort["primary_test"]
        lines.append(
            "| "
            f"`{cohort['cohort']}` | `{cohort['n_units']}` | `{cohort['n_synapses_loaded']}` | "
            f"`{cohort['n_connected_edge_pairs']}` | `{cohort['n_confirmed_positive_after_fdr']}` | "
            f"`{_round(primary['distance_matched_delta'])}` | "
            f"`{_round(primary['distance_matched_q_one_sided'])}` | "
            f"`{_round(primary['degree_matched_delta'])}` | "
            f"`{_round(primary['degree_matched_q_one_sided'])}` |"
        )
    lines.extend(
        [
            "",
            "## Unit-Cluster Bootstrap Stability",
            "",
            "| Cohort | Distance median | Distance CI95 | Degree median | Degree CI95 |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for cohort in payload["cohorts"]:
        distance = cohort["unit_bootstrap"]["distance_matched_delta"]
        degree = cohort["unit_bootstrap"]["degree_matched_delta"]
        lines.append(
            "| "
            f"`{cohort['cohort']}` | `{_round(distance['median'])}` | "
            f"`[{_round(distance['ci95_low'])}, {_round(distance['ci95_high'])}]` | "
            f"`{_round(degree['median'])}` | "
            f"`[{_round(degree['ci95_low'])}, {_round(degree['ci95_high'])}]` |"
        )
    lines.extend(["", "## Claims Allowed", ""])
    lines.extend(f"- {claim}" for claim in payload["claims_allowed"])
    lines.extend(["", "## Claims Blocked", ""])
    lines.extend(f"- {claim}" for claim in payload["claims_blocked"])
    lines.append("")
    markdown.parent.mkdir(parents=True, exist_ok=True)
    markdown.write_text("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--markdown", type=Path, default=DEFAULT_MARKDOWN)
    parser.add_argument("--bootstrap-samples", type=int, default=300)
    parser.add_argument("--seed", type=int, default=71)
    args = parser.parse_args()
    payload = run(
        output=args.output,
        markdown=args.markdown,
        bootstrap_samples=args.bootstrap_samples,
        seed=args.seed,
    )
    print(json.dumps({"output": str(args.output), "q1_package_ready": payload["q1_package_ready"]}))


if __name__ == "__main__":
    main()
