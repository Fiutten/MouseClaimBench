"""Network-dependent inference for the fixed MICRONS primary endpoint.

The discovery cohort fixes the positive direction. The same connected-pair
coefficient is then evaluated confirmatorily in two non-overlapping hold-outs.
Pair-level OLS uncertainty is reported only as a naive reference. Primary
uncertainty uses a directed dyadic sandwich in which two observations may be
dependent whenever their pairs share either neuron. A Freedman--Lane
simultaneous node-label permutation is the corroborating test.

Positive output remains a local observational association. Neither estimator
identifies a biological causal mechanism.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy import stats

from mousebrainbench import __version__
from mousebrainbench.artifacts import code_revision

SCRIPTS_DIR = Path(__file__).resolve().parents[2] / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from scripts.run_microns_stratified_structure_function import (
    _add_metric_columns,
    _build_pair_frame,
    _parse_position,
)
from scripts.run_microns_structure_function_pilot import (
    FUNCTION_COLUMNS,
    SPATIAL_COLUMNS,
    _prepare_edges,
    _prepare_units,
)

DEFAULT_OUTPUT = Path("results/microns_network_inference/summary.json")
DEFAULT_MARKDOWN = Path("results/microns_network_inference/summary.md")
PRIMARY_METRIC = "readout_location"
CONTROL_NAMES = (
    "log1p_distance",
    "squared_log1p_distance",
    "log1p_pre_degree",
    "log1p_post_degree",
    "same_coarse_cell_type",
)


@dataclass(frozen=True)
class CohortSpec:
    """Relative MICRONS files for one fixed evaluation window."""

    name: str
    units: Path
    edges: Path


COHORTS = (
    CohortSpec(
        "discovery",
        Path("data/microns/expanded/dt_coreg_units_v1507_sample1000.csv"),
        Path("data/microns/expanded/dt_coreg_edges_v1507_sample1000.csv"),
    ),
    CohortSpec(
        "holdout_offset1000",
        Path("data/microns/expanded/dt_coreg_units_v1507_holdout_offset1000.csv"),
        Path("data/microns/expanded/dt_coreg_edges_v1507_holdout_offset1000.csv"),
    ),
    CohortSpec(
        "holdout_offset2000",
        Path("data/microns/expanded/dt_coreg_units_v1507_holdout_offset2000.csv"),
        Path("data/microns/expanded/dt_coreg_edges_v1507_holdout_offset2000.csv"),
    ),
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return f"sha256:{digest.hexdigest()}"


def _load_units(path: Path):
    units = _prepare_units(path)
    if not set(SPATIAL_COLUMNS).issubset(units.columns) and "pt_position" in units.columns:
        positions = np.vstack(units["pt_position"].map(_parse_position).to_numpy())
        units["pt_position_x"] = positions[:, 0]
        units["pt_position_y"] = positions[:, 1]
        units["pt_position_z"] = positions[:, 2]
    return units.dropna(subset=["pt_root_id", *SPATIAL_COLUMNS, *FUNCTION_COLUMNS]).copy()


def _design(frame) -> tuple[np.ndarray, np.ndarray, tuple[str, ...]]:
    """Build the fixed full model with standardized continuous controls."""

    log_distance = np.log1p(frame["distance"].to_numpy(float))
    log_pre_degree = np.log1p(frame["pre_degree"].to_numpy(float))
    log_post_degree = np.log1p(frame["post_degree"].to_numpy(float))

    def standardized(values: np.ndarray) -> np.ndarray:
        scale = float(values.std())
        return (values - values.mean()) / scale if scale > 0.0 else values * 0.0

    distance_z = standardized(log_distance)
    controls = np.column_stack(
        (
            distance_z,
            np.square(distance_z),
            standardized(log_pre_degree),
            standardized(log_post_degree),
            frame.get("same_coarse_cell_type", False).to_numpy(float),
        )
    )
    focal = frame["connected"].to_numpy(float)
    full = np.column_stack((np.ones(len(frame)), focal, controls))
    names = ("intercept", "connected", *CONTROL_NAMES)
    return full, controls, names


def _dyadic_meat(
    scores: np.ndarray,
    pre: np.ndarray,
    post: np.ndarray,
    n_units: int,
) -> np.ndarray:
    """Return the score meat for dependence among dyads sharing any unit.

    Summed incident-unit scores count an observation and reciprocal directed
    pairs twice because they share both members. Subtracting the score outer
    product aggregated by unordered dyad removes exactly that duplicate count.
    """

    if np.any(pre == post):
        raise ValueError("dyadic covariance requires non-self pairs")
    incident = np.zeros((n_units, scores.shape[1]), dtype=float)
    np.add.at(incident, pre, scores)
    np.add.at(incident, post, scores)
    incident_meat = incident.T @ incident

    pair_index = np.full((n_units, n_units), -1, dtype=np.int32)
    pair_index[pre, post] = np.arange(len(pre), dtype=np.int32)
    upper = pre < post
    reverse = pair_index[post[upper], pre[upper]]
    if np.any(reverse < 0):
        raise ValueError("directed pair frame is not reciprocal and complete")
    dyad_scores = scores[upper] + scores[reverse]
    repeated_dyad_meat = dyad_scores.T @ dyad_scores
    return incident_meat - repeated_dyad_meat


def _linear_inference(
    design: np.ndarray,
    outcome: np.ndarray,
    pre: np.ndarray,
    post: np.ndarray,
    n_units: int,
) -> dict[str, Any]:
    """Fit OLS and compare naive HC1 with directed dyadic uncertainty."""

    xtx_inverse = np.linalg.inv(design.T @ design)
    coefficients = xtx_inverse @ design.T @ outcome
    residuals = outcome - design @ coefficients
    scores = design * residuals[:, None]
    observations, parameters = design.shape
    hc1_factor = observations / (observations - parameters)
    naive_covariance = hc1_factor * xtx_inverse @ (scores.T @ scores) @ xtx_inverse
    dyadic_factor = hc1_factor * n_units / (n_units - 1)
    dyadic_covariance = (
        dyadic_factor
        * xtx_inverse
        @ _dyadic_meat(scores, pre, post, n_units)
        @ xtx_inverse
    )
    naive_variance = float(naive_covariance[1, 1])
    dyadic_variance = float(dyadic_covariance[1, 1])
    if naive_variance <= 0.0 or dyadic_variance <= 0.0:
        raise ValueError("connected coefficient has non-positive estimated variance")
    coefficient = float(coefficients[1])
    naive_se = float(np.sqrt(naive_variance))
    dyadic_se = float(np.sqrt(dyadic_variance))
    dyadic_t = coefficient / dyadic_se
    return {
        "coefficients": coefficients,
        "residuals": residuals,
        "connected_coefficient": coefficient,
        "naive_hc1_standard_error": naive_se,
        "naive_hc1_two_sided_p_value": float(
            2.0 * stats.norm.sf(abs(coefficient / naive_se))
        ),
        "dyadic_cluster_standard_error": dyadic_se,
        "dyadic_cluster_t_statistic": float(dyadic_t),
        "dyadic_cluster_degrees_of_freedom": n_units - 1,
        "dyadic_cluster_two_sided_p_value": float(
            2.0 * stats.t.sf(abs(dyadic_t), df=n_units - 1)
        ),
        "dyadic_to_naive_se_ratio": dyadic_se / naive_se,
    }


def _freedman_lane_node_permutation(
    *,
    design: np.ndarray,
    controls: np.ndarray,
    outcome: np.ndarray,
    pre: np.ndarray,
    post: np.ndarray,
    n_units: int,
    n_permutations: int,
    rng: np.random.Generator,
) -> dict[str, Any]:
    """Run a simultaneous row-column Freedman--Lane residual permutation."""

    reduced = np.column_stack((np.ones(len(controls)), controls))
    reduced_coefficients = np.linalg.lstsq(reduced, outcome, rcond=None)[0]
    residuals = outcome - reduced @ reduced_coefficients
    focal = design[:, 1]
    focal_residual = focal - reduced @ np.linalg.lstsq(reduced, focal, rcond=None)[0]
    denominator = float(focal_residual @ focal_residual)
    if denominator <= 0.0:
        raise ValueError("connected indicator is collinear with fixed controls")
    observed = float(focal_residual @ outcome / denominator)
    residual_matrix = np.zeros((n_units, n_units), dtype=float)
    residual_matrix[pre, post] = residuals
    null = np.empty(n_permutations, dtype=float)
    for index in range(n_permutations):
        permutation = rng.permutation(n_units)
        permuted_residuals = residual_matrix[permutation[pre], permutation[post]]
        null[index] = float(focal_residual @ permuted_residuals / denominator)
    p_value = (1.0 + float(np.sum(null >= observed))) / (n_permutations + 1.0)
    return {
        "statistic": "partial connected-pair regression coefficient",
        "observed_coefficient": observed,
        "null_mean": float(null.mean()),
        "null_standard_deviation": float(null.std(ddof=1)),
        "one_sided_p_value": p_value,
        "n_permutations": n_permutations,
        "permutation_action": "same random node permutation applied to row and column labels",
        "exchangeability_assumption": (
            "reduced-model residual arrays are invariant under simultaneous node relabeling"
        ),
    }


def _cohort_result(
    spec: CohortSpec,
    *,
    data_root: Path,
    n_permutations: int,
    seed: int,
) -> dict[str, Any]:
    units_path = data_root / spec.units
    edges_path = data_root / spec.edges
    units = _load_units(units_path)
    edges = _prepare_edges(edges_path, set(units["pt_root_id"].astype("int64")))
    frame = _build_pair_frame(units, edges)
    _add_metric_columns(frame, units)
    design, controls, coefficient_names = _design(frame)
    outcome = frame[PRIMARY_METRIC].to_numpy(float)
    pre = frame["pre_idx"].to_numpy(int)
    post = frame["post_idx"].to_numpy(int)
    inference = _linear_inference(design, outcome, pre, post, len(units))
    permutation = _freedman_lane_node_permutation(
        design=design,
        controls=controls,
        outcome=outcome,
        pre=pre,
        post=post,
        n_units=len(units),
        n_permutations=n_permutations,
        rng=np.random.default_rng(seed),
    )
    passed = bool(
        inference["connected_coefficient"] > 0.0
        and inference["dyadic_cluster_two_sided_p_value"] <= 0.05
        and permutation["one_sided_p_value"] <= 0.05
    )
    return {
        "cohort": spec.name,
        "units_file": str(units_path),
        "units_sha256": _sha256(units_path),
        "edges_file": str(edges_path),
        "edges_sha256": _sha256(edges_path),
        "n_units": len(units),
        "n_candidate_directed_pairs": len(frame),
        "n_connected_directed_pairs": int(frame["connected"].sum()),
        "coefficient_names": list(coefficient_names),
        "connected_coefficient": inference["connected_coefficient"],
        "naive_hc1_standard_error": inference["naive_hc1_standard_error"],
        "naive_hc1_two_sided_p_value": inference[
            "naive_hc1_two_sided_p_value"
        ],
        "dyadic_cluster_standard_error": inference[
            "dyadic_cluster_standard_error"
        ],
        "dyadic_cluster_t_statistic": inference[
            "dyadic_cluster_t_statistic"
        ],
        "dyadic_cluster_degrees_of_freedom": inference[
            "dyadic_cluster_degrees_of_freedom"
        ],
        "dyadic_cluster_two_sided_p_value": inference[
            "dyadic_cluster_two_sided_p_value"
        ],
        "dyadic_to_naive_se_ratio": inference["dyadic_to_naive_se_ratio"],
        "permutation_seed": seed,
        "freedman_lane_node_permutation": permutation,
        "network_inference_passed": passed,
    }


def _confirmation_status(cohorts: list[dict[str, Any]]) -> dict[str, object]:
    """Separate discovery direction selection from hold-out confirmation."""

    if len(cohorts) != 3:
        raise ValueError("MICRONS confirmation requires one discovery and two hold-outs")
    discovery_direction_positive = bool(cohorts[0]["connected_coefficient"] > 0.0)
    holdout_results = tuple(
        bool(row["network_inference_passed"]) for row in cohorts[1:]
    )
    return {
        "discovery_direction_positive": discovery_direction_positive,
        "holdout_results": holdout_results,
        "confirmation_passed": discovery_direction_positive and all(holdout_results),
    }


def run(
    *,
    output: Path = DEFAULT_OUTPUT,
    markdown: Path = DEFAULT_MARKDOWN,
    data_root: Path = Path("."),
    n_permutations: int = 1_000,
    seed: int = 2_026_080_201,
) -> Path:
    """Select direction in discovery and test it in two fixed hold-outs."""

    cohorts = [
        _cohort_result(
            spec,
            data_root=data_root,
            n_permutations=n_permutations,
            seed=seed + index,
        )
        for index, spec in enumerate(COHORTS)
    ]
    confirmation = _confirmation_status(cohorts)
    all_descriptive_criteria_met = all(
        row["network_inference_passed"] for row in cohorts
    )
    payload = {
        "version": __version__,
        "git_revision": code_revision(),
        "analysis": "microns_fixed_endpoint_network_dependent_inference",
        "primary_endpoint": f"all_pairs/{PRIMARY_METRIC}",
        "controls": list(CONTROL_NAMES),
        "full_model": "readout_similarity ~ connected + fixed_controls",
        "reduced_model": "readout_similarity ~ fixed_controls",
        "primary_inference": "directed dyadic cluster-robust linear model",
        "corroborating_inference": "Freedman-Lane simultaneous node-label permutation",
        "permutation_unit": "neuron identifier applied simultaneously to sender and receiver labels",
        "permutation_seed_streams": [seed + index for index in range(len(COHORTS))],
        "discovery_role": (
            "select the positive direction and freeze the analysis specification; "
            "its one-sided permutation value is descriptive"
        ),
        "confirmation_cohorts": [row["cohort"] for row in cohorts[1:]],
        "cohorts": cohorts,
        "all_cohorts_descriptive_criteria_met": all_descriptive_criteria_met,
        **confirmation,
        "decision": (
            "microns_fixed_endpoint_survives_network_dependent_inference"
            if confirmation["confirmation_passed"]
            else "microns_fixed_endpoint_not_confirmed_by_network_dependent_inference"
        ),
        "interpretation": (
            "Discovery selects the positive direction and its one-sided permutation "
            "value is descriptive. Confirmation requires the fixed conjunction in both "
            "hold-outs. A positive decision supports only a local observational "
            "association after the fixed controls. Dyadic covariance assumes pairs "
            "without a shared unit are independent. The permutation test additionally "
            "assumes exchangeability of reduced-model residual arrays under node "
            "relabeling. Neither assumption establishes causality or independent "
            "biological replication."
        ),
        "method_references": [
            {
                "citation": "Aronow, Samii, and Assenova (2015)",
                "doi": "10.1093/pan/mpv018",
                "role": "directed dyadic cluster-robust covariance",
            },
            {
                "citation": "Dekker, Krackhardt, and Snijders (2007)",
                "doi": "10.1007/s11336-007-9016-1",
                "role": "Freedman-Lane MRQAP residual permutation",
            },
        ],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2))
    write_markdown(payload, markdown)
    return output


def write_markdown(payload: dict[str, Any], markdown: Path) -> None:
    lines = [
        "# MICRONS network-dependent inference",
        "",
        f"- Decision: `{payload['decision']}`",
        f"- Fixed endpoint: `{payload['primary_endpoint']}`",
        "",
        "| Cohort | Units | Connected | Coefficient | Dyadic SE | Dyadic p | FL p | Passed |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in payload["cohorts"]:
        permutation = row["freedman_lane_node_permutation"]
        lines.append(
            f"| `{row['cohort']}` | {row['n_units']} | "
            f"{row['n_connected_directed_pairs']} | {row['connected_coefficient']:.6g} | "
            f"{row['dyadic_cluster_standard_error']:.6g} | "
            f"{row['dyadic_cluster_two_sided_p_value']:.6g} | "
            f"{permutation['one_sided_p_value']:.6g} | "
            f"`{row['network_inference_passed']}` |"
        )
    lines.extend(["", "## Interpretation", "", payload["interpretation"], ""])
    markdown.parent.mkdir(parents=True, exist_ok=True)
    markdown.write_text("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--markdown", type=Path, default=DEFAULT_MARKDOWN)
    parser.add_argument("--data-root", type=Path, default=Path("."))
    parser.add_argument("--n-permutations", type=int, default=1_000)
    parser.add_argument("--seed", type=int, default=2_026_080_201)
    args = parser.parse_args()
    print(
        json.dumps(
            {
                "output": str(
                    run(
                        output=args.output,
                        markdown=args.markdown,
                        data_root=args.data_root,
                        n_permutations=args.n_permutations,
                        seed=args.seed,
                    ).resolve()
                )
            }
        )
    )


if __name__ == "__main__":
    main()
