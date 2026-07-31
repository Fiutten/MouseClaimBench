"""Harder robustness checks for the fixed MICRONS primary endpoint.

This benchmark does not select new MICRONS endpoints. It reuses the fixed
``all_pairs/readout_location`` endpoint and asks whether the discovery and
hold-out cohorts remain positive under stricter controls:

1. combined distance plus pre/post degree matching;
2. within-distance-bin readout similarity shuffling.

The result remains local and observational even when positive.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from mousebrainbench import __version__
from mousebrainbench.artifacts import code_revision

SCRIPTS_DIR = Path(__file__).resolve().parents[2] / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from scripts.run_microns_stratified_structure_function import (  # noqa: E402
    _add_metric_columns,
    _build_pair_frame,
    _matched_null,
    _parse_position,
    _permutation_p_value,
    _safe_cut_from_reference,
)
from scripts.run_microns_structure_function_pilot import (  # noqa: E402
    FUNCTION_COLUMNS,
    SPATIAL_COLUMNS,
    _prepare_edges,
    _prepare_units,
)


DEFAULT_OUTPUT = Path("results/microns_primary_robustness/summary.json")
DEFAULT_MARKDOWN = Path("results/microns_primary_robustness/summary.md")
PRIMARY_METRIC = "readout_location"


@dataclass(frozen=True)
class CohortSpec:
    """Input files for one MICRONS primary-endpoint robustness cohort."""

    name: str
    units: Path
    edges: Path


COHORTS = (
    CohortSpec(
        name="discovery",
        units=Path("data/microns/expanded/dt_coreg_units_v1507_sample1000.csv"),
        edges=Path("data/microns/expanded/dt_coreg_edges_v1507_sample1000.csv"),
    ),
    CohortSpec(
        name="holdout_offset1000",
        units=Path("data/microns/expanded/dt_coreg_units_v1507_holdout_offset1000.csv"),
        edges=Path("data/microns/expanded/dt_coreg_edges_v1507_holdout_offset1000.csv"),
    ),
    CohortSpec(
        name="holdout_offset2000",
        units=Path("data/microns/expanded/dt_coreg_units_v1507_holdout_offset2000.csv"),
        edges=Path("data/microns/expanded/dt_coreg_edges_v1507_holdout_offset2000.csv"),
    ),
)


def _load_units(path: Path):
    units = _prepare_units(path)
    if not set(SPATIAL_COLUMNS).issubset(units.columns) and "pt_position" in units.columns:
        positions = np.vstack(units["pt_position"].map(_parse_position).to_numpy())
        units["pt_position_x"] = positions[:, 0]
        units["pt_position_y"] = positions[:, 1]
        units["pt_position_z"] = positions[:, 2]
    return units.dropna(subset=["pt_root_id", *SPATIAL_COLUMNS, *FUNCTION_COLUMNS]).copy()


def _shuffle_within_distance_bins(
    *,
    values: np.ndarray,
    connected_mask: np.ndarray,
    distance_bins: np.ndarray,
    observed: float,
    n_permutations: int,
    rng: np.random.Generator,
) -> dict[str, float]:
    """Shuffle readout similarity within distance bins without copying full frames."""

    null_means = []
    bins = distance_bins.astype(str)
    bin_specs = []
    for bin_value in np.unique(bins):
        if bin_value == "<NA>":
            continue
        bin_mask = bins == bin_value
        connected_in_bin = connected_mask & bin_mask
        connected_count = int(connected_in_bin.sum())
        if connected_count:
            bin_specs.append((values[bin_mask], connected_count))
    total_connected = int(connected_mask.sum())
    for _ in range(n_permutations):
        sampled_sum = 0.0
        sampled_count = 0
        for pool, connected_count in bin_specs:
            sampled = rng.choice(pool, size=connected_count, replace=False)
            sampled_sum += float(sampled.sum())
            sampled_count += connected_count
        if sampled_count != total_connected:
            raise ValueError("distance-bin shuffle failed to cover all connected pairs")
        null_means.append(sampled_sum / sampled_count)
    null = np.asarray(null_means, dtype=float)
    return {
        "null_mean": float(null.mean()),
        "delta": float(observed - null.mean()),
        "p_one_sided": _permutation_p_value(observed, null),
    }


def _cohort_result(
    spec: CohortSpec,
    *,
    n_permutations: int,
    seed: int,
) -> dict[str, Any]:
    units = _load_units(spec.units)
    edges = _prepare_edges(spec.edges, set(units["pt_root_id"].astype("int64")))
    frame = _build_pair_frame(units, edges)
    _add_metric_columns(frame, units)
    unconnected_reference = frame[~frame["connected"]].copy()
    frame["distance_bin"] = _safe_cut_from_reference(
        frame["distance"], unconnected_reference["distance"], 8
    )
    frame["pre_degree_bin"] = _safe_cut_from_reference(
        frame["pre_degree"], unconnected_reference["pre_degree"], 5
    )
    frame["post_degree_bin"] = _safe_cut_from_reference(
        frame["post_degree"], unconnected_reference["post_degree"], 5
    )
    connected = frame[frame["connected"]].copy()
    unconnected = frame[~frame["connected"]].copy()
    observed = float(connected[PRIMARY_METRIC].mean())
    rng = np.random.default_rng(seed)
    combined_null = _matched_null(
        connected=connected,
        unconnected=unconnected,
        metric=PRIMARY_METRIC,
        group_columns=["distance_bin", "pre_degree_bin", "post_degree_bin"],
        n_permutations=n_permutations,
        rng=rng,
    )
    combined = {
        "null_mean": float(combined_null.mean()),
        "delta": float(observed - combined_null.mean()),
        "p_one_sided": _permutation_p_value(observed, combined_null),
    }
    shuffle = _shuffle_within_distance_bins(
        values=frame[PRIMARY_METRIC].to_numpy(float),
        connected_mask=frame["connected"].to_numpy(bool),
        distance_bins=frame["distance_bin"].astype(str).to_numpy(),
        observed=observed,
        n_permutations=n_permutations,
        rng=rng,
    )
    robust = bool(combined["delta"] > 0 and combined["p_one_sided"] <= 0.05 and shuffle["delta"] > 0)
    return {
        "cohort": spec.name,
        "units": str(spec.units),
        "edges": str(spec.edges),
        "n_units": int(len(units)),
        "n_synapses_loaded": int(len(edges)),
        "n_connected_pairs": int(len(connected)),
        "n_candidate_pairs": int(len(frame)),
        "observed_connected_mean": observed,
        "combined_distance_degree_control": combined,
        "within_distance_readout_shuffle": shuffle,
        "robust_primary_endpoint": robust,
    }


def run(
    *,
    output: Path = DEFAULT_OUTPUT,
    markdown: Path = DEFAULT_MARKDOWN,
    n_permutations: int = 500,
    seed: int = 311,
) -> Path:
    """Run MICRONS primary-endpoint robustness checks."""

    cohorts = [
        _cohort_result(spec, n_permutations=n_permutations, seed=seed + idx)
        for idx, spec in enumerate(COHORTS)
    ]
    all_robust = all(cohort["robust_primary_endpoint"] for cohort in cohorts)
    payload = {
        "version": __version__,
        "git_revision": code_revision(),
        "analysis": "microns_primary_endpoint_robustness",
        "primary_endpoint": f"all_pairs/{PRIMARY_METRIC}",
        "n_permutations": n_permutations,
        "cohorts": cohorts,
        "all_cohorts_robust": bool(all_robust),
        "decision": (
            "microns_primary_endpoint_survives_harder_controls"
            if all_robust
            else "microns_primary_endpoint_requires_caution_under_harder_controls"
        ),
        "interpretation": (
            "Positive results remain local and observational. Combined matching and "
            "within-distance shuffling reduce obvious degree and spatial confounds, "
            "but they do not establish causality or independent biological replication."
        ),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2))
    write_markdown(payload, markdown)
    return output


def write_markdown(payload: dict[str, Any], markdown: Path) -> None:
    """Write a compact MICRONS robustness report."""

    lines = [
        "# MICRONS Primary Endpoint Robustness",
        "",
        f"- Decision: `{payload['decision']}`",
        f"- Primary endpoint: `{payload['primary_endpoint']}`",
        f"- All cohorts robust: `{payload['all_cohorts_robust']}`",
        "",
        "| Cohort | Units | Connected pairs | Combined delta | Combined p | Shuffle delta | Robust |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for cohort in payload["cohorts"]:
        combined = cohort["combined_distance_degree_control"]
        shuffle = cohort["within_distance_readout_shuffle"]
        lines.append(
            f"| `{cohort['cohort']}` | `{cohort['n_units']}` | `{cohort['n_connected_pairs']}` | "
            f"`{combined['delta']:.6g}` | `{combined['p_one_sided']:.6g}` | "
            f"`{shuffle['delta']:.6g}` | `{cohort['robust_primary_endpoint']}` |"
        )
    lines.extend(["", "## Interpretation", "", str(payload["interpretation"]), ""])
    markdown.parent.mkdir(parents=True, exist_ok=True)
    markdown.write_text("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--markdown", type=Path, default=DEFAULT_MARKDOWN)
    parser.add_argument("--n-permutations", type=int, default=500)
    parser.add_argument("--seed", type=int, default=311)
    args = parser.parse_args()
    path = run(
        output=args.output,
        markdown=args.markdown,
        n_permutations=args.n_permutations,
        seed=args.seed,
    )
    print(json.dumps({"output": str(path.resolve())}))


if __name__ == "__main__":
    main()
