"""Run stratified MICrONS structure-function checks with multiplicity control.

The expanded all-pairs MICrONS pilot is negative after distance matching. This
script asks whether that negative global result hides a narrower effect inside
deterministically generated anatomical, cell-type, reliability, or readout
strata. The analysis is exploratory unless an endpoint is fixed after discovery
and then evaluated in independent hold-outs. Some categorical strata are
data-dependent because deterministic selection rules use connected-pair category
frequencies. Broad counts of confirmed strata therefore provide exploratory
context rather than independent biological discoveries. A single uncorrected or
post-hoc subgroup must not be promoted to a mechanistic claim.
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

from run_microns_structure_function_pilot import (  # noqa: PLC0415
    FUNCTION_COLUMNS,
    SPATIAL_COLUMNS,
    _circular_distance_degrees,
    _parse_position,
    _permutation_p_value,
    _prepare_edges,
    _prepare_units,
    _zscore,
)


DEFAULT_UNITS = Path("data/microns/expanded/dt_coreg_units_v1507_sample1000.csv")
DEFAULT_EDGES = Path("data/microns/expanded/dt_coreg_edges_v1507_sample1000.csv")
DEFAULT_OUTPUT = Path("results/microns_structure_function_pilot/stratified_summary.json")
DEFAULT_MARKDOWN = Path("results/microns_structure_function_pilot/stratified_summary.md")


@dataclass(frozen=True)
class StratifiedConfig:
    """Runtime controls for exploratory but controlled MICrONS strata tests."""

    n_permutations: int = 1000
    seed: int = 23
    min_connected_pairs: int = 50
    max_category_strata: int = 24


def _safe_qcut(values: pd.Series, q: int) -> pd.Series:
    """Quantile-bin numeric values while tolerating degenerate distributions."""

    clean = values.replace([np.inf, -np.inf], np.nan)
    if clean.nunique(dropna=True) < 2:
        return pd.Series([pd.NA] * len(values), index=values.index, dtype="Int64")
    bins = pd.qcut(clean, q=min(q, clean.nunique(dropna=True)), labels=False, duplicates="drop")
    return bins.astype("Int64")


def _safe_cut_from_reference(values: pd.Series, reference: pd.Series, q: int) -> pd.Series:
    """Assign values to quantile bins computed from a reference distribution."""

    clean_ref = reference.replace([np.inf, -np.inf], np.nan).dropna()
    if clean_ref.nunique() < 2:
        return pd.Series([pd.NA] * len(values), index=values.index, dtype="Int64")
    _, edges = pd.qcut(clean_ref, q=min(q, clean_ref.nunique()), duplicates="drop", retbins=True)
    if len(edges) < 2:
        return pd.Series([pd.NA] * len(values), index=values.index, dtype="Int64")
    edges[0] = -np.inf
    edges[-1] = np.inf
    return pd.cut(values, bins=edges, labels=False, include_lowest=True).astype("Int64")


def _fdr_bh(p_values: list[float]) -> list[float]:
    """Benjamini-Hochberg FDR correction with monotonic adjusted p-values."""

    n = len(p_values)
    if n == 0:
        return []
    order = np.argsort(np.asarray(p_values, dtype=float))
    adjusted = np.empty(n, dtype=float)
    running = 1.0
    for rank_from_end, idx in enumerate(order[::-1], start=1):
        rank = n - rank_from_end + 1
        value = min(running, p_values[idx] * n / rank)
        running = value
        adjusted[idx] = value
    return [float(min(1.0, value)) for value in adjusted]


def _metric_matrices(units: pd.DataFrame) -> dict[str, np.ndarray]:
    """Build predefined functional-similarity matrices used by all strata.

    Higher values always mean "more similar". Negative distances are used for
    Euclidean metrics so that larger values still indicate closer responses.
    """

    continuous_columns = ["cc_abs", "cc_max", "cc_norm", "OSI", "DSI", "gOSI", "gDSI"]
    continuous = _zscore(units[continuous_columns].to_numpy(float))
    continuous_similarity = -np.linalg.norm(continuous[:, None, :] - continuous[None, :, :], axis=2)
    ori_similarity = _circular_distance_degrees(
        units["pref_ori"].to_numpy(float)[:, None],
        units["pref_ori"].to_numpy(float)[None, :],
        period=180.0,
    )
    dir_similarity = _circular_distance_degrees(
        units["pref_dir"].to_numpy(float)[:, None],
        units["pref_dir"].to_numpy(float)[None, :],
        period=360.0,
    )
    cc_norm_similarity = -np.abs(
        units["cc_norm"].to_numpy(float)[:, None] - units["cc_norm"].to_numpy(float)[None, :]
    )
    readout = units[["readout_loc_x", "readout_loc_y"]].to_numpy(float)
    readout_similarity = -np.linalg.norm(readout[:, None, :] - readout[None, :, :], axis=2)
    return {
        "aggregate_functional": continuous_similarity + ori_similarity + dir_similarity,
        "response_profile": continuous_similarity,
        "orientation": ori_similarity,
        "direction": dir_similarity,
        "cc_norm": cc_norm_similarity,
        "readout_location": readout_similarity,
    }


def _build_pair_frame(units: pd.DataFrame, edges: pd.DataFrame) -> pd.DataFrame:
    """Create the ordered non-self pair table once for all stratified tests."""

    roots = units["pt_root_id"].astype("int64").to_numpy()
    root_to_index = {int(root): idx for idx, root in enumerate(roots)}
    n_units = len(roots)
    pre_idx, post_idx = np.where(~np.eye(n_units, dtype=bool))
    xyz = units[list(SPATIAL_COLUMNS)].to_numpy(float)
    grouped = (
        edges.groupby(["pre_pt_root_id", "post_pt_root_id"])
        .agg(synapse_count=("size", "size"), synapse_size_sum=("size", "sum"))
        .reset_index()
    )
    connected_matrix = np.zeros((n_units, n_units), dtype=bool)
    synapse_count = np.zeros((n_units, n_units), dtype=np.int32)
    synapse_size = np.zeros((n_units, n_units), dtype=float)
    for row in grouped.itertuples(index=False):
        i = root_to_index.get(int(row.pre_pt_root_id))
        j = root_to_index.get(int(row.post_pt_root_id))
        if i is not None and j is not None and i != j:
            connected_matrix[i, j] = True
            synapse_count[i, j] = int(row.synapse_count)
            synapse_size[i, j] = float(row.synapse_size_sum)

    frame = pd.DataFrame(
        {
            "pre_idx": pre_idx,
            "post_idx": post_idx,
            "pre_root": roots[pre_idx],
            "post_root": roots[post_idx],
            "connected": connected_matrix[pre_idx, post_idx],
            "synapse_count": synapse_count[pre_idx, post_idx],
            "synapse_size_sum": synapse_size[pre_idx, post_idx],
            "distance": np.linalg.norm(xyz[pre_idx] - xyz[post_idx], axis=1),
        }
    )
    pre_degree = grouped.groupby("pre_pt_root_id")["post_pt_root_id"].nunique().to_dict()
    post_degree = grouped.groupby("post_pt_root_id")["pre_pt_root_id"].nunique().to_dict()
    frame["pre_degree"] = frame["pre_root"].map(pre_degree).fillna(0).astype(int)
    frame["post_degree"] = frame["post_root"].map(post_degree).fillna(0).astype(int)

    for column in ("coarse_cell_type", "fine_cell_type"):
        if column in units.columns:
            values = units[column].fillna("unknown").astype(str).to_numpy()
            frame[f"pre_{column}"] = values[pre_idx]
            frame[f"post_{column}"] = values[post_idx]
            frame[f"same_{column}"] = frame[f"pre_{column}"] == frame[f"post_{column}"]

    readout_x = units["readout_loc_x"].to_numpy(float)
    readout_y = units["readout_loc_y"].to_numpy(float)
    x_mid = float(np.nanmedian(readout_x))
    y_mid = float(np.nanmedian(readout_y))
    quadrants = np.array(
        [
            f"{'R' if x >= x_mid else 'L'}{'U' if y >= y_mid else 'D'}"
            for x, y in zip(readout_x, readout_y, strict=True)
        ],
        dtype=object,
    )
    frame["pre_readout_quadrant"] = quadrants[pre_idx]
    frame["post_readout_quadrant"] = quadrants[post_idx]
    frame["same_readout_quadrant"] = frame["pre_readout_quadrant"] == frame["post_readout_quadrant"]
    frame["distance_tercile"] = _safe_qcut(frame["distance"], 3).astype(str)
    reliability = units["cc_norm"].to_numpy(float)
    reliability_threshold = float(np.nanmedian(reliability))
    reliable = reliability >= reliability_threshold
    frame["both_high_reliability"] = reliable[pre_idx] & reliable[post_idx]
    return frame


def _add_metric_columns(frame: pd.DataFrame, units: pd.DataFrame) -> list[str]:
    """Attach metric values to pair rows and return metric names."""

    metrics = _metric_matrices(units)
    pre_idx = frame["pre_idx"].to_numpy(int)
    post_idx = frame["post_idx"].to_numpy(int)
    for name, matrix in metrics.items():
        frame[name] = matrix[pre_idx, post_idx]
    return list(metrics)


def _candidate_strata(frame: pd.DataFrame, config: StratifiedConfig) -> dict[str, pd.Series]:
    """Return deterministic candidate masks with enough connected pairs to test.

    Some categorical strata are selected by fixed rules over connected-pair
    category counts. They are therefore rule-based exploratory strata rather
    than fully preregistered biological hypotheses.
    """

    candidates: dict[str, pd.Series] = {
        "all_pairs": pd.Series(True, index=frame.index),
        "same_coarse_cell_type": frame.get("same_coarse_cell_type", pd.Series(False, index=frame.index)),
        "different_coarse_cell_type": ~frame.get(
            "same_coarse_cell_type", pd.Series(True, index=frame.index)
        ),
        "same_fine_cell_type": frame.get("same_fine_cell_type", pd.Series(False, index=frame.index)),
        "same_readout_quadrant": frame["same_readout_quadrant"],
        "different_readout_quadrant": ~frame["same_readout_quadrant"],
        "both_high_reliability": frame["both_high_reliability"],
    }
    for tercile in sorted(frame["distance_tercile"].dropna().unique()):
        if tercile != "<NA>":
            candidates[f"distance_tercile:{tercile}"] = frame["distance_tercile"] == tercile

    category_specs = [
        ("coarse_pair", "pre_coarse_cell_type", "post_coarse_cell_type"),
        ("readout_quadrant_pair", "pre_readout_quadrant", "post_readout_quadrant"),
    ]
    for prefix, pre_col, post_col in category_specs:
        if pre_col not in frame.columns or post_col not in frame.columns:
            continue
        connected = frame[frame["connected"]].copy()
        connected["category"] = connected[pre_col].astype(str) + "->" + connected[post_col].astype(str)
        for category, count in connected["category"].value_counts().head(config.max_category_strata).items():
            if int(count) >= config.min_connected_pairs:
                candidates[f"{prefix}:{category}"] = (
                    frame[pre_col].astype(str) + "->" + frame[post_col].astype(str) == category
                )
    return {
        name: mask.fillna(False).astype(bool)
        for name, mask in candidates.items()
        if int((frame.loc[mask.fillna(False).astype(bool), "connected"]).sum())
        >= config.min_connected_pairs
    }


def _matched_null(
    *,
    connected: pd.DataFrame,
    unconnected: pd.DataFrame,
    metric: str,
    group_columns: list[str],
    n_permutations: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Sample null means matched on one or more discretized pair covariates."""

    fallback = unconnected[metric].to_numpy(float)
    if not group_columns or unconnected.empty:
        return np.asarray(
            [
                rng.choice(fallback, size=len(connected), replace=len(fallback) < len(connected)).mean()
                for _ in range(n_permutations)
            ],
            dtype=float,
        )
    connected_groups = connected[group_columns].astype(str).agg("|".join, axis=1)
    unconnected_groups = unconnected[group_columns].astype(str).agg("|".join, axis=1)
    pools = {
        group: values[metric].to_numpy(float)
        for group, values in unconnected.assign(_group=unconnected_groups).groupby("_group")
    }
    group_counts = connected_groups.value_counts().to_dict()
    means: list[float] = []
    for _ in range(n_permutations):
        chunks = []
        for group, count in group_counts.items():
            pool = pools.get(group, fallback)
            chunks.append(rng.choice(pool, size=int(count), replace=len(pool) < int(count)))
        means.append(float(np.concatenate(chunks).mean()))
    return np.asarray(means, dtype=float)


def _analyze_one(
    *,
    subset: pd.DataFrame,
    metric: str,
    stratum: str,
    config: StratifiedConfig,
    rng: np.random.Generator,
) -> dict[str, Any] | None:
    """Analyze one metric within one stratum against random and matched nulls."""

    connected = subset[subset["connected"]].copy()
    unconnected = subset[~subset["connected"]].copy()
    if len(connected) < config.min_connected_pairs or unconnected.empty:
        return None
    observed = float(connected[metric].mean())
    random_null = np.asarray(
        [
            rng.choice(
                unconnected[metric].to_numpy(float),
                size=len(connected),
                replace=len(unconnected) < len(connected),
            ).mean()
            for _ in range(config.n_permutations)
        ],
        dtype=float,
    )
    subset = subset.copy()
    subset["distance_bin"] = _safe_cut_from_reference(subset["distance"], unconnected["distance"], 8)
    subset["pre_degree_bin"] = _safe_cut_from_reference(subset["pre_degree"], unconnected["pre_degree"], 5)
    subset["post_degree_bin"] = _safe_cut_from_reference(subset["post_degree"], unconnected["post_degree"], 5)
    connected = subset[subset["connected"]].copy()
    unconnected = subset[~subset["connected"]].copy()
    distance_null = _matched_null(
        connected=connected,
        unconnected=unconnected,
        metric=metric,
        group_columns=["distance_bin"],
        n_permutations=config.n_permutations,
        rng=rng,
    )
    degree_null = _matched_null(
        connected=connected,
        unconnected=unconnected,
        metric=metric,
        group_columns=["pre_degree_bin", "post_degree_bin"],
        n_permutations=config.n_permutations,
        rng=rng,
    )
    return {
        "stratum": stratum,
        "metric": metric,
        "n_pairs": int(len(subset)),
        "n_connected_pairs": int(len(connected)),
        "n_unconnected_pairs": int(len(unconnected)),
        "observed_connected_mean": observed,
        "random_null_mean": float(random_null.mean()),
        "random_delta": float(observed - random_null.mean()),
        "random_p_one_sided": _permutation_p_value(observed, random_null),
        "distance_matched_null_mean": float(distance_null.mean()),
        "distance_matched_delta": float(observed - distance_null.mean()),
        "distance_matched_p_one_sided": _permutation_p_value(observed, distance_null),
        "degree_matched_null_mean": float(degree_null.mean()),
        "degree_matched_delta": float(observed - degree_null.mean()),
        "degree_matched_p_one_sided": _permutation_p_value(observed, degree_null),
    }


def _finalize_tests(tests: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Attach FDR-adjusted q-values and a strict confirmation flag."""

    for key in ("random_p_one_sided", "distance_matched_p_one_sided", "degree_matched_p_one_sided"):
        q_values = _fdr_bh([float(row[key]) for row in tests])
        for row, q_value in zip(tests, q_values, strict=True):
            row[key.replace("_p_", "_q_")] = q_value
    for row in tests:
        row["confirmed_positive_after_fdr"] = bool(
            row["random_delta"] > 0.0
            and row["distance_matched_delta"] > 0.0
            and row["degree_matched_delta"] > 0.0
            and row["distance_matched_q_one_sided"] <= 0.05
            and row["degree_matched_q_one_sided"] <= 0.05
        )
    return tests


def run(
    *,
    units_path: Path = DEFAULT_UNITS,
    edges_path: Path = DEFAULT_EDGES,
    output: Path = DEFAULT_OUTPUT,
    markdown: Path = DEFAULT_MARKDOWN,
    config: StratifiedConfig = StratifiedConfig(),
) -> dict[str, Any]:
    """Execute the stratified expanded MICrONS analysis."""

    units = _prepare_units(units_path)
    # Expanded CAVE tables store positions as a bracketed string; normalize here
    # so downstream code can rely on explicit numeric coordinate columns.
    if not set(SPATIAL_COLUMNS).issubset(units.columns) and "pt_position" in units.columns:
        positions = np.vstack(units["pt_position"].map(_parse_position).to_numpy())
        units["pt_position_x"] = positions[:, 0]
        units["pt_position_y"] = positions[:, 1]
        units["pt_position_z"] = positions[:, 2]
    units = units.dropna(subset=["pt_root_id", *SPATIAL_COLUMNS, *FUNCTION_COLUMNS]).copy()
    edges = _prepare_edges(edges_path, set(units["pt_root_id"].astype("int64")))
    frame = _build_pair_frame(units, edges)
    metrics = _add_metric_columns(frame, units)
    strata = _candidate_strata(frame, config)
    rng = np.random.default_rng(config.seed)
    tests: list[dict[str, Any]] = []
    for stratum, mask in strata.items():
        subset = frame.loc[mask].copy()
        for metric in metrics:
            result = _analyze_one(
                subset=subset,
                metric=metric,
                stratum=stratum,
                config=config,
                rng=rng,
            )
            if result is not None:
                tests.append(result)
    tests = _finalize_tests(tests)
    confirmed = [row for row in tests if row["confirmed_positive_after_fdr"]]
    top_distance = sorted(tests, key=lambda row: row["distance_matched_q_one_sided"])[:10]
    payload = {
        "analysis": "microns_stratified_structure_function",
        "units": str(units_path),
        "edges": str(edges_path),
        "n_units": int(len(units)),
        "n_synapses_loaded": int(len(edges)),
        "n_connected_edge_pairs": int(
            edges[["pre_pt_root_id", "post_pt_root_id"]].drop_duplicates().shape[0]
        ),
        "n_candidate_pairs": int(len(frame)),
        "n_tested_strata": int(len(strata)),
        "n_statistical_tests": int(len(tests)),
        "n_confirmed_positive_after_fdr": int(len(confirmed)),
        "positive_stratified_structure_function_result": bool(confirmed),
        "scientific_decision": (
            "positive_stratified_structure_function_signal"
            if confirmed
            else "negative_or_exploratory_stratified_structure_function_signal"
        ),
        "config": {
            "n_permutations": config.n_permutations,
            "seed": config.seed,
            "min_connected_pairs": config.min_connected_pairs,
            "max_category_strata": config.max_category_strata,
        },
        "tested_strata": sorted(strata),
        "confirmed_positive_tests": confirmed,
        "top_distance_matched_tests": top_distance,
        "all_tests": tests,
        "interpretation": (
            "A stratified MICrONS effect is accepted only if it has positive "
            "random, distance-matched, and degree-matched deltas and survives "
            "FDR correction on the matched controls. Otherwise the result remains "
            "negative or exploratory and must not be used as a Q1 mechanistic claim."
        ),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2))
    _write_markdown(payload, markdown)
    return payload


def _round(value: Any) -> str:
    if isinstance(value, float):
        if math.isnan(value):
            return "nan"
        return f"{value:.6g}"
    return str(value)


def _write_markdown(payload: dict[str, Any], markdown: Path) -> None:
    """Write a compact human-readable stratification report."""

    lines = [
        "# MICRONS Stratified Structure-Function Analysis",
        "",
        f"- Scientific decision: `{payload['scientific_decision']}`",
        f"- Positive after FDR: `{payload['positive_stratified_structure_function_result']}`",
        f"- Units: `{payload['n_units']}`",
        f"- Synapses loaded: `{payload['n_synapses_loaded']}`",
        f"- Connected edge pairs: `{payload['n_connected_edge_pairs']}`",
        f"- Candidate pairs: `{payload['n_candidate_pairs']}`",
        f"- Tested strata: `{payload['n_tested_strata']}`",
        f"- Statistical tests: `{payload['n_statistical_tests']}`",
        f"- Confirmed positives: `{payload['n_confirmed_positive_after_fdr']}`",
        "",
        "## Top Distance-Matched Tests",
        "",
        "| Stratum | Metric | Connected | Distance delta | Distance q | Degree delta | Degree q | Confirmed |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in payload["top_distance_matched_tests"][:10]:
        lines.append(
            "| "
            f"`{row['stratum']}` | `{row['metric']}` | `{row['n_connected_pairs']}` | "
            f"`{_round(row['distance_matched_delta'])}` | "
            f"`{_round(row['distance_matched_q_one_sided'])}` | "
            f"`{_round(row['degree_matched_delta'])}` | "
            f"`{_round(row['degree_matched_q_one_sided'])}` | "
            f"`{row['confirmed_positive_after_fdr']}` |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            str(payload["interpretation"]),
            "",
        ]
    )
    markdown.parent.mkdir(parents=True, exist_ok=True)
    markdown.write_text("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--units", type=Path, default=DEFAULT_UNITS)
    parser.add_argument("--edges", type=Path, default=DEFAULT_EDGES)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--markdown", type=Path, default=DEFAULT_MARKDOWN)
    parser.add_argument("--n-permutations", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=23)
    parser.add_argument("--min-connected-pairs", type=int, default=50)
    parser.add_argument("--max-category-strata", type=int, default=24)
    args = parser.parse_args()
    payload = run(
        units_path=args.units,
        edges_path=args.edges,
        output=args.output,
        markdown=args.markdown,
        config=StratifiedConfig(
            n_permutations=args.n_permutations,
            seed=args.seed,
            min_connected_pairs=args.min_connected_pairs,
            max_category_strata=args.max_category_strata,
        ),
    )
    print(
        json.dumps(
            {
                "output": str(args.output),
                "positive": payload["positive_stratified_structure_function_result"],
                "n_tests": payload["n_statistical_tests"],
            }
        )
    )


if __name__ == "__main__":
    main()
