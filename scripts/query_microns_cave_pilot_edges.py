"""Query a tiny MICrONS structure-function edge pilot through CAVE.

This script is the approved route for MICrONS structure-function evidence. It
uses the small static EM/function co-registration table to choose candidate
neurons, then asks CAVE for true synaptic edges among those root IDs. If CAVE
authentication is missing, the script records a blocked diagnostic instead of
approving the pilot.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from build_microns_static_pilot_manifest import (
    DEFAULT_DIAGNOSTIC,
    DEFAULT_MANIFEST,
    DEFAULT_MATCHED,
    DEFAULT_STATIC_ROOT,
    build_manifest,
)


DEFAULT_EDGES = Path("data/microns/static_small/cave_synaptic_edges.csv")


def _load_candidate_roots(matched_units: Path, limit_neurons: int) -> list[int]:
    """Load unique MICrONS root IDs from the local matched unit table."""

    matched = pd.read_csv(matched_units)
    roots = (
        matched["pt_root_id"]
        .dropna()
        .astype("int64")
        .drop_duplicates()
        .head(limit_neurons)
        .tolist()
    )
    if not roots:
        raise ValueError(f"No candidate pt_root_id values found in {matched_units}")
    return [int(root) for root in roots]


def query_edges(
    *,
    matched_units: Path,
    output_edges: Path,
    manifest: Path,
    limit_neurons: int,
    limit_edges: int,
    datastack: str,
    materialization_version: int | None,
) -> dict[str, Any]:
    """Query CAVE for synaptic edges and update the local pilot manifest."""

    from caveclient import CAVEclient  # noqa: PLC0415

    roots = _load_candidate_roots(matched_units, limit_neurons)
    try:
        client = CAVEclient(datastack)
        edges = client.materialize.synapse_query(
            pre_ids=roots,
            post_ids=roots,
            limit=limit_edges,
            materialization_version=materialization_version,
        )
    except Exception as exc:  # noqa: BLE001
        diagnostic = build_manifest(
            static_root=DEFAULT_STATIC_ROOT,
            manifest_path=manifest,
            matched_output=matched_units,
            diagnostic_output=DEFAULT_DIAGNOSTIC,
            cave_auth_error=f"{type(exc).__name__}: {str(exc)[:1000]}",
        )
        return {
            "approved": False,
            "blocked": True,
            "error_type": type(exc).__name__,
            "error": str(exc)[:1000],
            "n_candidate_roots": len(roots),
            "diagnostic": diagnostic,
        }

    output_edges.parent.mkdir(parents=True, exist_ok=True)
    edges.to_csv(output_edges, index=False)
    manifest_payload = json.loads(manifest.read_text())
    manifest_payload.pop("cave_auth_error", None)
    manifest_payload.pop("blocked_reason", None)
    manifest_payload.update(
        {
            "has_structural_edges": len(edges) > 0,
            "n_structural_edges": int(len(edges)),
            "structural_edge_source": str(output_edges),
            "cave_datastack": datastack,
            "cave_query": {
                "limit_neurons": limit_neurons,
                "limit_edges": limit_edges,
                "materialization_version": materialization_version,
                "method": "client.materialize.synapse_query(pre_ids=roots, post_ids=roots)",
            },
        }
    )
    manifest.write_text(json.dumps(manifest_payload, indent=2))
    return {
        "approved": len(edges) > 0,
        "blocked": False,
        "n_candidate_roots": len(roots),
        "n_edges": int(len(edges)),
        "edges": str(output_edges),
        "manifest": str(manifest),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matched-units", type=Path, default=DEFAULT_MATCHED)
    parser.add_argument("--output-edges", type=Path, default=DEFAULT_EDGES)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--limit-neurons", type=int, default=172)
    parser.add_argument("--limit-edges", type=int, default=10000)
    parser.add_argument("--datastack", default="minnie65_public")
    parser.add_argument("--materialization-version", type=int, default=117)
    args = parser.parse_args()
    payload = query_edges(
        matched_units=args.matched_units,
        output_edges=args.output_edges,
        manifest=args.manifest,
        limit_neurons=args.limit_neurons,
        limit_edges=args.limit_edges,
        datastack=args.datastack,
        materialization_version=args.materialization_version,
    )
    print(json.dumps(payload))


if __name__ == "__main__":
    main()
