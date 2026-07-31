"""Run a bounded MICrONS structure-function pilot when real edges exist.

The analysis asks a narrow, falsifiable question: are functionally similar
co-registered MICrONS units more likely to be connected by real synaptic edges
than expected from spatially matched or random pairs? It deliberately avoids
claiming a full connectome/function model from the small pilot.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


DEFAULT_MATCHED = Path("data/microns/static_small/matched_function_em_units.csv")
DEFAULT_EDGES = Path("data/microns/static_small/cave_synaptic_edges.csv")
DEFAULT_OUTPUT = Path("results/microns_structure_function_pilot/summary.json")
DEFAULT_MARKDOWN = Path("results/microns_structure_function_pilot/summary.md")


FUNCTION_COLUMNS = ("cc_abs", "cc_max", "cc_norm", "OSI", "DSI", "gOSI", "gDSI", "pref_ori", "pref_dir")
SPATIAL_COLUMNS = ("pt_position_x", "pt_position_y", "pt_position_z")


@dataclass(frozen=True)
class PairRecord:
    """One candidate ordered pair for structure-function testing."""

    pre_root: int
    post_root: int
    connected: bool
    distance: float
    functional_similarity: float
    synapse_count: int
    synapse_size_sum: float
    pre_degree: int
    post_degree: int
    same_coarse_type: bool | None = None


def _zscore(values: np.ndarray) -> np.ndarray:
    """Column-wise z-score with zero-variance protection."""

    mean = np.nanmean(values, axis=0)
    std = np.nanstd(values, axis=0)
    std[std == 0] = 1.0
    return (values - mean) / std


def _circular_distance_degrees(a: np.ndarray, b: np.ndarray, period: float) -> np.ndarray:
    """Return normalized circular similarity for angle-valued features."""

    raw = np.abs(a - b) % period
    dist = np.minimum(raw, period - raw)
    return 1.0 - (dist / (period / 2.0))


def _parse_position(value: Any) -> np.ndarray:
    """Parse a MICrONS position stored as array, list, or bracketed string."""

    if isinstance(value, np.ndarray):
        return value.astype(float)
    if isinstance(value, (list, tuple)):
        return np.asarray(value, dtype=float)
    cleaned = str(value).strip().replace("[", " ").replace("]", " ")
    return np.fromstring(cleaned, sep=" ", dtype=float)


def _prepare_units(matched_units: Path) -> pd.DataFrame:
    """Load matched MICrONS units with usable spatial and functional fields."""

    units = pd.read_csv(matched_units)
    if not set(SPATIAL_COLUMNS).issubset(units.columns) and "pt_position" in units.columns:
        positions = np.vstack(units["pt_position"].map(_parse_position).to_numpy())
        units["pt_position_x"] = positions[:, 0]
        units["pt_position_y"] = positions[:, 1]
        units["pt_position_z"] = positions[:, 2]
    required = ["pt_root_id", *SPATIAL_COLUMNS, *FUNCTION_COLUMNS]
    usable = units.dropna(subset=required).copy()
    usable["pt_root_id"] = usable["pt_root_id"].astype("int64")
    usable = usable.drop_duplicates(subset=["pt_root_id"], keep="first")
    return usable


def _prepare_edges(edges_path: Path, valid_roots: set[int]) -> pd.DataFrame:
    """Load directed synapses restricted to candidate roots."""

    edges = pd.read_csv(edges_path)
    required = {"pre_pt_root_id", "post_pt_root_id"}
    if not required.issubset(edges.columns):
        raise ValueError(f"Edges file lacks required columns: {sorted(required)}")
    edges = edges.dropna(subset=["pre_pt_root_id", "post_pt_root_id"]).copy()
    edges["pre_pt_root_id"] = edges["pre_pt_root_id"].astype("int64")
    edges["post_pt_root_id"] = edges["post_pt_root_id"].astype("int64")
    edges = edges[
        (edges["pre_pt_root_id"] != edges["post_pt_root_id"])
        & edges["pre_pt_root_id"].isin(valid_roots)
        & edges["post_pt_root_id"].isin(valid_roots)
    ].copy()
    if "size" not in edges.columns:
        edges["size"] = 1.0
    return edges


def _functional_similarity_matrix(units: pd.DataFrame) -> np.ndarray:
    """Compute conservative functional similarity from static response properties."""

    continuous = _zscore(units[["cc_abs", "cc_max", "cc_norm", "OSI", "DSI", "gOSI", "gDSI"]].to_numpy(float))
    diff = continuous[:, None, :] - continuous[None, :, :]
    continuous_similarity = -np.linalg.norm(diff, axis=2)
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
    return continuous_similarity + ori_similarity + dir_similarity


def _pair_records(units: pd.DataFrame, edges: pd.DataFrame) -> list[PairRecord]:
    """Build all ordered non-self candidate pairs with structure/function fields."""

    roots = units["pt_root_id"].astype("int64").to_numpy()
    xyz = units[list(SPATIAL_COLUMNS)].to_numpy(float)
    similarity = _functional_similarity_matrix(units)
    grouped = (
        edges.groupby(["pre_pt_root_id", "post_pt_root_id"])
        .agg(synapse_count=("size", "size"), synapse_size_sum=("size", "sum"))
        .reset_index()
    )
    pair_weights = {
        (int(row.pre_pt_root_id), int(row.post_pt_root_id)): (
            int(row.synapse_count),
            float(row.synapse_size_sum),
        )
        for row in grouped.itertuples(index=False)
    }
    pre_degree = grouped.groupby("pre_pt_root_id")["post_pt_root_id"].nunique().to_dict()
    post_degree = grouped.groupby("post_pt_root_id")["pre_pt_root_id"].nunique().to_dict()
    coarse_types = (
        units.set_index("pt_root_id")["coarse_cell_type"].to_dict()
        if "coarse_cell_type" in units.columns
        else {}
    )
    records: list[PairRecord] = []
    for i, pre in enumerate(roots):
        distances = np.linalg.norm(xyz[i] - xyz, axis=1)
        for j, post in enumerate(roots):
            if i == j:
                continue
            count, size_sum = pair_weights.get((int(pre), int(post)), (0, 0.0))
            pre_type = coarse_types.get(int(pre))
            post_type = coarse_types.get(int(post))
            records.append(
                PairRecord(
                    pre_root=int(pre),
                    post_root=int(post),
                    connected=count > 0,
                    distance=float(distances[j]),
                    functional_similarity=float(similarity[i, j]),
                    synapse_count=count,
                    synapse_size_sum=size_sum,
                    pre_degree=int(pre_degree.get(int(pre), 0)),
                    post_degree=int(post_degree.get(int(post), 0)),
                    same_coarse_type=(
                        bool(pre_type == post_type)
                        if pd.notna(pre_type) and pd.notna(post_type)
                        else None
                    ),
                )
            )
    return records


def _permutation_p_value(observed: float, null: np.ndarray) -> float:
    """One-sided permutation p-value for connected-pair functional similarity."""

    return float((np.sum(null >= observed) + 1) / (len(null) + 1))


def _spearman(x: pd.Series, y: pd.Series) -> dict[str, float | None]:
    """Return Spearman rho/p-value when SciPy can estimate it."""

    from scipy.stats import spearmanr  # noqa: PLC0415

    if x.nunique() < 2 or y.nunique() < 2:
        return {"rho": None, "p": None}
    result = spearmanr(x, y)
    return {"rho": float(result.statistic), "p": float(result.pvalue)}


def _grouped_null_means(
    *,
    connected_groups: pd.Series,
    unconnected_groups: pd.Series,
    unconnected_values: pd.Series,
    n_permutations: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Sample matched-null means from precomputed group arrays."""

    fallback = unconnected_values.to_numpy(float)
    pools = {
        group: values["functional_similarity"].to_numpy(float)
        for group, values in pd.DataFrame(
            {"group": unconnected_groups, "functional_similarity": unconnected_values}
        )
        .dropna(subset=["group"])
        .groupby("group")
    }
    group_counts = connected_groups.value_counts(dropna=True).to_dict()
    means: list[float] = []
    for _ in range(n_permutations):
        sampled_chunks = []
        for group, count in group_counts.items():
            pool = pools.get(group, fallback)
            sampled_chunks.append(rng.choice(pool, size=int(count), replace=len(pool) < count))
        if sampled_chunks:
            sampled = np.concatenate(sampled_chunks)
        else:
            sampled = rng.choice(fallback, size=len(connected_groups), replace=True)
        means.append(float(sampled.mean()))
    return np.asarray(means, dtype=float)


def _analyze(records: list[PairRecord], *, n_permutations: int, seed: int) -> dict[str, Any]:
    """Compare real connected pairs against random and distance-matched nulls."""

    rng = np.random.default_rng(seed)
    frame = pd.DataFrame([record.__dict__ for record in records])
    connected = frame[frame["connected"]]
    unconnected = frame[~frame["connected"]]
    if connected.empty:
        return {
            "approved_analysis": False,
            "reason": "No real synaptic edges among candidate roots.",
            "n_pairs": int(len(frame)),
            "n_connected_pairs": 0,
        }

    observed = float(connected["functional_similarity"].mean())
    random_null = np.array(
        [
            unconnected["functional_similarity"]
            .sample(n=len(connected), replace=len(unconnected) < len(connected), random_state=int(rng.integers(0, 2**31 - 1)))
            .mean()
            for _ in range(n_permutations)
        ],
        dtype=float,
    )

    _, distance_edges = pd.qcut(
        unconnected["distance"], q=min(10, len(unconnected)), duplicates="drop", retbins=True
    )
    unconnected_distance_bins = pd.cut(
        unconnected["distance"], bins=distance_edges, labels=False, include_lowest=True
    )
    connected_distance_bins = pd.cut(
        connected["distance"], bins=distance_edges, labels=False, include_lowest=True
    )
    distance_null = _grouped_null_means(
        connected_groups=connected_distance_bins,
        unconnected_groups=unconnected_distance_bins,
        unconnected_values=unconnected["functional_similarity"],
        n_permutations=n_permutations,
        rng=rng,
    )

    _, pre_degree_edges = pd.qcut(
        unconnected["pre_degree"], q=5, duplicates="drop", retbins=True
    )
    _, post_degree_edges = pd.qcut(
        unconnected["post_degree"], q=5, duplicates="drop", retbins=True
    )
    unconnected_pre = pd.cut(
        unconnected["pre_degree"], bins=pre_degree_edges, labels=False, include_lowest=True
    )
    unconnected_post = pd.cut(
        unconnected["post_degree"], bins=post_degree_edges, labels=False, include_lowest=True
    )
    connected_pre = pd.cut(
        connected["pre_degree"], bins=pre_degree_edges, labels=False, include_lowest=True
    )
    connected_post = pd.cut(
        connected["post_degree"], bins=post_degree_edges, labels=False, include_lowest=True
    )
    degree_null = _grouped_null_means(
        connected_groups=connected_pre.astype(str) + "_" + connected_post.astype(str),
        unconnected_groups=unconnected_pre.astype(str) + "_" + unconnected_post.astype(str),
        unconnected_values=unconnected["functional_similarity"],
        n_permutations=n_permutations,
        rng=rng,
    )

    random_delta = float(observed - random_null.mean())
    random_p = _permutation_p_value(observed, random_null)
    distance_delta = float(observed - distance_null.mean())
    distance_p = _permutation_p_value(observed, distance_null)
    degree_delta = float(observed - degree_null.mean())
    degree_p = _permutation_p_value(observed, degree_null)
    positive_result = (
        random_delta > 0.0
        and distance_delta > 0.0
        and degree_delta > 0.0
        and distance_p <= 0.05
        and degree_p <= 0.05
    )
    weighted = connected.copy()
    count_corr = _spearman(weighted["synapse_count"], weighted["functional_similarity"])
    size_corr = _spearman(weighted["synapse_size_sum"], weighted["functional_similarity"])
    return {
        "approved_analysis": True,
        "positive_structure_function_result": positive_result,
        "scientific_decision": (
            "positive_structure_function_pilot"
            if positive_result
            else "negative_or_inconclusive_structure_function_pilot"
        ),
        "n_pairs": int(len(frame)),
        "n_connected_pairs": int(len(connected)),
        "n_unconnected_pairs": int(len(unconnected)),
        "observed_connected_mean_functional_similarity": observed,
        "random_null_mean": float(random_null.mean()),
        "random_null_delta": random_delta,
        "random_null_p_one_sided": random_p,
        "distance_matched_null_mean": float(distance_null.mean()),
        "distance_matched_delta": distance_delta,
        "distance_matched_p_one_sided": distance_p,
        "degree_matched_null_mean": float(degree_null.mean()),
        "degree_matched_delta": degree_delta,
        "degree_matched_p_one_sided": degree_p,
        "connected_synapse_count_spearman": count_corr,
        "connected_synapse_size_spearman": size_corr,
        "connected_synapse_count_mean": float(connected["synapse_count"].mean()),
        "connected_synapse_size_sum_mean": float(connected["synapse_size_sum"].mean()),
        "n_permutations": n_permutations,
        "seed": seed,
    }


def run(
    *,
    matched_units: Path = DEFAULT_MATCHED,
    edges: Path = DEFAULT_EDGES,
    output: Path = DEFAULT_OUTPUT,
    markdown: Path = DEFAULT_MARKDOWN,
    n_permutations: int = 1000,
    seed: int = 17,
) -> dict[str, Any]:
    """Execute the MICrONS structure-function pilot analysis."""

    if not edges.exists():
        payload = {
            "analysis": "microns_structure_function_pilot",
            "approved_analysis": False,
            "reason": f"Missing real edge file: {edges}",
            "matched_units": str(matched_units),
            "edges": str(edges),
        }
    else:
        units = _prepare_units(matched_units)
        edge_frame = _prepare_edges(edges, set(units["pt_root_id"].astype("int64")))
        records = _pair_records(units, edge_frame)
        payload = {
            "analysis": "microns_structure_function_pilot",
            "matched_units": str(matched_units),
            "edges": str(edges),
            "n_units": int(len(units)),
            "n_synapses_loaded": int(len(edge_frame)),
            "n_edge_pairs_loaded": int(
                edge_frame[["pre_pt_root_id", "post_pt_root_id"]].drop_duplicates().shape[0]
            ),
            "coarse_cell_type_coverage": (
                int(units["coarse_cell_type"].notna().sum())
                if "coarse_cell_type" in units.columns
                else 0
            ),
            **_analyze(records, n_permutations=n_permutations, seed=seed),
            "interpretation": (
                "A positive result requires connected pairs to exceed both random "
                "and distance-matched nulls. This pilot tests local structure-function "
                "coupling only; it is not a whole-MICrONS model."
            ),
        }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2))
    lines = [
        "# MICrONS Structure-Function Pilot",
        "",
        f"- Approved analysis: `{payload['approved_analysis']}`",
        f"- Reason: {payload.get('reason', 'real edge file available')}",
    ]
    if payload.get("approved_analysis"):
        lines.extend(
            [
                f"- Units: `{payload['n_units']}`",
                f"- Connected pairs: `{payload['n_connected_pairs']}`",
                f"- Scientific decision: `{payload['scientific_decision']}`",
                f"- Random-null delta: `{payload['random_null_delta']}`",
                f"- Distance-matched delta: `{payload['distance_matched_delta']}`",
                f"- Distance-matched p: `{payload['distance_matched_p_one_sided']}`",
                f"- Degree-matched delta: `{payload['degree_matched_delta']}`",
                f"- Degree-matched p: `{payload['degree_matched_p_one_sided']}`",
            ]
        )
    markdown.parent.mkdir(parents=True, exist_ok=True)
    markdown.write_text("\n".join(lines) + "\n")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matched-units", type=Path, default=DEFAULT_MATCHED)
    parser.add_argument("--edges", type=Path, default=DEFAULT_EDGES)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--markdown", type=Path, default=DEFAULT_MARKDOWN)
    parser.add_argument("--n-permutations", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()
    payload = run(
        matched_units=args.matched_units,
        edges=args.edges,
        output=args.output,
        markdown=args.markdown,
        n_permutations=args.n_permutations,
        seed=args.seed,
    )
    print(json.dumps({"output": str(args.output), "approved_analysis": payload["approved_analysis"]}))


if __name__ == "__main__":
    main()
