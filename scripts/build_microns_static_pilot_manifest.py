"""Build a bounded MICrONS pilot manifest from small public static files.

The manifest is intentionally strict. Static MICrONS files can provide
functional properties, anatomical coordinates and EM/function co-registration,
but true synaptic edges require either CAVE access or a very large synapse
graph download. This script records that distinction instead of silently
substituting spatial proximity for structural connectivity.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd


DEFAULT_STATIC_ROOT = Path("data/microns/static_small")
DEFAULT_MANIFEST = Path("data/microns/pilot_manifest.json")
DEFAULT_MATCHED = Path("data/microns/static_small/matched_function_em_units.csv")
DEFAULT_DIAGNOSTIC = Path("results/microns_pilot_gate/static_pilot_diagnostic.json")


COREG_COLUMNS = [
    "id",
    "valid",
    "pt_position_x",
    "pt_position_y",
    "pt_position_z",
    "session",
    "scan_idx",
    "unit_id",
    "pt_supervoxel_id",
    "pt_root_id",
]


def _read_coreg(path: Path) -> pd.DataFrame:
    """Read the headerless MICrONS functional co-registration CSV."""

    return pd.read_csv(path, names=COREG_COLUMNS)


def _join_functional(static_root: Path) -> pd.DataFrame:
    """Join static co-registration rows to lightweight functional properties."""

    coreg = _read_coreg(static_root / "func_unit_em_match_release.csv")
    anatomy = pd.read_csv(static_root / "dt_anatomy_units.csv")
    performance = pd.read_csv(static_root / "dt_performance_units.csv")
    tuning = pd.read_csv(static_root / "dt_ori_dir_tuning_units.csv")
    key = ["session", "scan_idx", "unit_id"]
    matched = coreg.merge(anatomy, on=key, how="left")
    matched = matched.merge(performance, on=key, how="left")
    matched = matched.merge(tuning, on=key, how="left")
    return matched


def _count_existing(paths: list[Path]) -> dict[str, Any]:
    """Return reproducibility metadata for downloaded static files."""

    return {
        str(path): {
            "exists": path.exists(),
            "size_bytes": path.stat().st_size if path.exists() else None,
        }
        for path in paths
    }


def build_manifest(
    *,
    static_root: Path,
    manifest_path: Path,
    matched_output: Path,
    diagnostic_output: Path,
    cave_auth_error: str | None = None,
) -> dict[str, Any]:
    """Write a MICrONS pilot manifest and diagnostic from local static files."""

    required = [
        static_root / "func_unit_em_match_release.csv",
        static_root / "proofreading_status_public_release.csv",
        static_root / "dt_anatomy_units.csv",
        static_root / "dt_performance_units.csv",
        static_root / "dt_ori_dir_tuning_units.csv",
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing MICrONS static inputs: {missing}")

    matched = _join_functional(static_root)
    matched_output.parent.mkdir(parents=True, exist_ok=True)
    matched.to_csv(matched_output, index=False)

    has_spatial = matched[["pt_position_x", "pt_position_y", "pt_position_z"]].notna().all(axis=1)
    has_function = matched[["cc_abs", "cc_max", "gOSI", "pref_ori"]].notna().all(axis=1)
    has_root = matched["pt_root_id"].notna() & (matched["pt_root_id"] != 0)
    usable = matched[has_spatial & has_function & has_root]

    # True synaptic edges are not present in the small static files. We do not
    # use distance-derived pseudo-edges because that would change the scientific
    # question from structure-function to spatial-function.
    has_structural_edges = False
    structural_edge_source = None
    blocked_reason = (
        "Small static files provide EM/function matched units and functional "
        "properties, but no synaptic edge list. CAVE synapse_query needs a token, "
        "and the static minnie65 synapse graph is ~47.5 GB."
    )

    manifest = {
        "dataset": "MICrONS minnie65 public static small pilot",
        "materialization_version": "static_v117_plus_digital_twin_v2",
        "n_neurons": int(len(usable)),
        "n_coregistered_units": int(len(matched)),
        "has_spatial_coordinates": bool(len(usable) > 0),
        "has_functional_responses": bool(len(usable) > 0),
        "has_structural_edges": has_structural_edges,
        "estimated_download_gb": round(
            sum(path.stat().st_size for path in required) / (1024**3),
            6,
        ),
        "matched_units_csv": str(matched_output),
        "structural_edge_source": structural_edge_source,
        "blocked_reason": blocked_reason,
        "cave_auth_error": cave_auth_error,
        "sources": {
            "functional_coregistration": (
                "https://bossdb-open-data.s3.amazonaws.com/iarpa_microns/minnie/"
                "functional_coregistration/func_unit_em_match_release.csv"
            ),
            "proofreading_status": (
                "https://bossdb-open-data.s3.amazonaws.com/iarpa_microns/minnie/"
                "proofreading_status/proofreading_status_public_release.csv"
            ),
            "digital_twin_properties_v2": (
                "https://bossdb-open-data.s3.amazonaws.com/iarpa_microns/minnie/"
                "functional_data/digital_twin_properties/v2/README.md"
            ),
            "synapse_graph_large_static": (
                "https://bossdb-open-data.s3.amazonaws.com/iarpa_microns/minnie/"
                "minnie65/synapse_graph/synapses_pni_2.csv"
            ),
        },
    }
    diagnostic = {
        "manifest": manifest,
        "static_files": _count_existing(required),
        "columns": list(matched.columns),
        "n_with_spatial": int(has_spatial.sum()),
        "n_with_function": int(has_function.sum()),
        "n_with_root_id": int(has_root.sum()),
        "n_usable_without_edges": int(len(usable)),
        "interpretation": (
            "The static small subset is valid as a co-registration/functional "
            "properties smoke test, but it is not an approved structure-function "
            "pilot until real synaptic edges are available."
        ),
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    diagnostic_output.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2))
    diagnostic_output.write_text(json.dumps(diagnostic, indent=2))
    return diagnostic


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--static-root", type=Path, default=DEFAULT_STATIC_ROOT)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--matched-output", type=Path, default=DEFAULT_MATCHED)
    parser.add_argument("--diagnostic-output", type=Path, default=DEFAULT_DIAGNOSTIC)
    parser.add_argument("--cave-auth-error", default=None)
    args = parser.parse_args()
    payload = build_manifest(
        static_root=args.static_root,
        manifest_path=args.manifest,
        matched_output=args.matched_output,
        diagnostic_output=args.diagnostic_output,
        cave_auth_error=args.cave_auth_error,
    )
    print(
        json.dumps(
            {
                "manifest": str(args.manifest.resolve()),
                "n_usable_without_edges": payload["n_usable_without_edges"],
                "has_structural_edges": payload["manifest"]["has_structural_edges"],
            }
        )
    )


if __name__ == "__main__":
    main()
