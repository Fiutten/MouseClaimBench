"""Build an expanded MICrONS digital-twin/coregistration pilot through CAVE.

This script uses modern CAVE materialization tables that already join digital
twin functional properties with EM root IDs. It is the scalable route beyond
the small static v117 micro-pilot.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd


DEFAULT_UNITS = Path("data/microns/expanded/dt_coreg_units_v1507_sample1000.csv")
DEFAULT_EDGES = Path("data/microns/expanded/dt_coreg_edges_v1507_sample1000.csv")
DEFAULT_MANIFEST = Path("data/microns/expanded/pilot_manifest.json")

FUNCTION_COLUMNS = ("cc_abs", "cc_max", "cc_norm", "OSI", "DSI", "gOSI", "gDSI", "pref_ori", "pref_dir")


def _usable_units(frame: pd.DataFrame, limit_units: int, offset_units: int = 0) -> pd.DataFrame:
    """Return valid, unique-root units with functional and spatial properties."""

    required = ["pt_root_id", "pt_position", *FUNCTION_COLUMNS]
    usable = frame.dropna(subset=required).copy()
    usable["pt_root_id"] = usable["pt_root_id"].astype("int64")
    usable = usable.drop_duplicates(subset=["pt_root_id"], keep="first")
    return usable.iloc[offset_units : offset_units + limit_units].copy()


def _try_enrich_cell_types(client: Any, units: pd.DataFrame, version: int) -> pd.DataFrame:
    """Best-effort coarse/fine cell type enrichment from CAVE tables."""

    roots = set(units["pt_root_id"].astype("int64"))
    enriched = units.copy()
    for table, column in (
        ("baylor_log_reg_cell_type_coarse_v1", "coarse_cell_type"),
        ("baylor_gnn_cell_type_fine_model_v2", "fine_cell_type"),
    ):
        try:
            cell_types = client.materialize.query_table(
                table,
                limit=50000,
                materialization_version=version,
            )
        except Exception:  # noqa: BLE001 - enrichment is optional
            continue
        if "pt_root_id" not in cell_types or "cell_type" not in cell_types:
            continue
        subset = cell_types[cell_types["pt_root_id"].astype("int64").isin(roots)]
        subset = subset[["pt_root_id", "cell_type"]].drop_duplicates("pt_root_id")
        subset = subset.rename(columns={"cell_type": column})
        enriched = enriched.merge(subset, on="pt_root_id", how="left")
    return enriched


def query_expanded_pilot(
    *,
    units_output: Path,
    edges_output: Path,
    manifest_output: Path,
    table: str,
    materialization_version: int,
    limit_query_rows: int,
    limit_units: int,
    offset_units: int,
    limit_edges: int,
    datastack: str,
    enrich_cell_types: bool,
) -> dict[str, Any]:
    """Query expanded co-registered units and their induced synaptic subgraph."""

    from caveclient import CAVEclient  # noqa: PLC0415

    client = CAVEclient(datastack)
    raw = client.materialize.query_table(
        table,
        limit=limit_query_rows,
        materialization_version=materialization_version,
    )
    units = _usable_units(raw, limit_units, offset_units)
    if enrich_cell_types:
        units = _try_enrich_cell_types(client, units, materialization_version)
    roots = units["pt_root_id"].astype("int64").tolist()
    edges = client.materialize.synapse_query(
        pre_ids=roots,
        post_ids=roots,
        limit=limit_edges,
        materialization_version=materialization_version,
    )

    units_output.parent.mkdir(parents=True, exist_ok=True)
    edges_output.parent.mkdir(parents=True, exist_ok=True)
    manifest_output.parent.mkdir(parents=True, exist_ok=True)
    units.to_csv(units_output, index=False)
    edges.to_csv(edges_output, index=False)
    edge_pairs = edges[["pre_pt_root_id", "post_pt_root_id"]].drop_duplicates()
    manifest = {
        "dataset": "MICrONS expanded CAVE digital twin coregistration pilot",
        "datastack": datastack,
        "table": table,
        "materialization_version": materialization_version,
        "n_neurons": int(len(units)),
        "n_query_rows": int(len(raw)),
        "n_structural_edges": int(len(edges)),
        "n_unique_edge_pairs": int(len(edge_pairs)),
        "has_spatial_coordinates": "pt_position" in units.columns,
        "has_functional_responses": all(column in units.columns for column in FUNCTION_COLUMNS),
        "has_structural_edges": len(edges) > 0,
        "estimated_download_gb": round(
            (units_output.stat().st_size + edges_output.stat().st_size) / (1024**3),
            6,
        ),
        "units_csv": str(units_output),
        "structural_edge_source": str(edges_output),
        "cave_query": {
            "limit_query_rows": limit_query_rows,
            "limit_units": limit_units,
            "offset_units": offset_units,
            "limit_edges": limit_edges,
        },
        "cell_type_enrichment": {
            "requested": enrich_cell_types,
            "coarse_coverage": int(units.get("coarse_cell_type", pd.Series(dtype=object)).notna().sum()),
            "fine_coverage": int(units.get("fine_cell_type", pd.Series(dtype=object)).notna().sum()),
        },
        "citation_notice": (
            "CAVE table owner notice requests citation of Wang et al. 2025, "
            "Ding/Fahey/Papadopoulos et al. 2025, and MICrONS Consortium et al. 2025."
        ),
    }
    manifest_output.write_text(json.dumps(manifest, indent=2))
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--units-output", type=Path, default=DEFAULT_UNITS)
    parser.add_argument("--edges-output", type=Path, default=DEFAULT_EDGES)
    parser.add_argument("--manifest-output", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--table", default="digital_twin_properties_bcm_coreg_v4")
    parser.add_argument("--materialization-version", type=int, default=1507)
    parser.add_argument("--limit-query-rows", type=int, default=1200)
    parser.add_argument("--limit-units", type=int, default=1000)
    parser.add_argument("--offset-units", type=int, default=0)
    parser.add_argument("--limit-edges", type=int, default=100000)
    parser.add_argument("--datastack", default="minnie65_public")
    parser.add_argument("--no-cell-types", action="store_true")
    args = parser.parse_args()
    payload = query_expanded_pilot(
        units_output=args.units_output,
        edges_output=args.edges_output,
        manifest_output=args.manifest_output,
        table=args.table,
        materialization_version=args.materialization_version,
        limit_query_rows=args.limit_query_rows,
        limit_units=args.limit_units,
        offset_units=args.offset_units,
        limit_edges=args.limit_edges,
        datastack=args.datastack,
        enrich_cell_types=not args.no_cell_types,
    )
    print(json.dumps(payload))


if __name__ == "__main__":
    main()
