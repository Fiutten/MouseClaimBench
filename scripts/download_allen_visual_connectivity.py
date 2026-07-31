"""Download and aggregate six-area visual connectivity from the official Allen API."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any
from urllib.parse import quote
from urllib.request import urlopen

import numpy as np

API = "https://api.brain-map.org/api/v2/data/query.json?criteria="
REGIONS = (
    {"id": 385, "acronym": "VISp", "name": "Primary visual area"},
    {"id": 409, "acronym": "VISl", "name": "Lateral visual area"},
    {"id": 402, "acronym": "VISal", "name": "Anterolateral visual area"},
    {"id": 417, "acronym": "VISrl", "name": "Rostrolateral visual area"},
    {"id": 533, "acronym": "VISpm", "name": "posteromedial visual area"},
    {"id": 394, "acronym": "VISam", "name": "Anteromedial visual area"},
)


def query(criteria: str) -> list[dict[str, Any]]:
    """Run one Allen RMA query and require a successful complete response."""

    url = API + quote(criteria, safe=":,$[]'")
    with urlopen(url, timeout=120) as response:  # noqa: S310 - fixed official API host
        payload = json.load(response)
    if not payload["success"]:
        raise RuntimeError(f"Allen API query failed: {payload['msg']}")
    return list(payload["msg"])


def build_payload() -> dict[str, Any]:
    index = query("model::ApiConnectivity,rma::options[num_rows$eqall]")
    region_ids = {item["id"] for item in REGIONS}
    experiments = [
        {
            "data_set_id": int(item["data_set_id"]),
            "source_structure_id": int(item["structure_id"]),
            "injection_volume": float(item["injection_volume"]),
            "transgenic_line_id": item.get("transgenic_line_id"),
        }
        for item in index
        if int(item["product_id"]) == 5 and int(item["structure_id"]) in region_ids
    ]
    unionizes: list[dict[str, Any]] = []
    target_ids = ",".join(str(item["id"]) for item in REGIONS)
    for start in range(0, len(experiments), 40):
        experiment_ids = ",".join(
            str(item["data_set_id"]) for item in experiments[start : start + 40]
        )
        unionizes.extend(
            query(
                "model::ProjectionStructureUnionize,"
                f"rma::criteria[section_data_set_id$in{experiment_ids}]"
                "[is_injection$eqfalse][hemisphere_id$eq3]"
                f"[structure_id$in{target_ids}],"
                "rma::options[num_rows$eqall]"
                "[only$eq'section_data_set_id,structure_id,hemisphere_id,"
                "normalized_projection_volume']"
            )
        )
    source_by_experiment = {
        item["data_set_id"]: item["source_structure_id"] for item in experiments
    }
    positions = {item["id"]: index for index, item in enumerate(REGIONS)}
    samples: dict[tuple[int, int], list[float]] = {}
    for row in unionizes:
        source = source_by_experiment[int(row["section_data_set_id"])]
        target = int(row["structure_id"])
        samples.setdefault((target, source), []).append(float(row["normalized_projection_volume"]))
    weights = np.zeros((len(REGIONS), len(REGIONS)), dtype=float)
    for (target, source), values in samples.items():
        if target != source:
            weights[positions[target], positions[source]] = float(np.median(values))
    acronym_by_id = {item["id"]: item["acronym"] for item in REGIONS}
    counts = {
        acronym: sum(
            item["source_structure_id"] == region_id for item in experiments
        )
        for region_id, acronym in acronym_by_id.items()
    }
    return {
        "source": "Allen Mouse Brain Connectivity Atlas API",
        "downloaded_at_utc": datetime.now(timezone.utc).isoformat(),
        "product_id": 5,
        "hemisphere_id": 3,
        "aggregation": "median normalized_projection_volume across experiments by source",
        "diagonal": "set to zero to exclude injection-site contamination",
        "regions": list(REGIONS),
        "experiment_counts_by_source": counts,
        "experiments": experiments,
        "weights_target_by_source": weights.tolist(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "output",
        type=Path,
        default=Path("mousebrainbench/data/reference/allen_visual_connectivity.json"),
        nargs="?",
    )
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    payload = build_payload()
    args.output.write_text(json.dumps(payload, indent=2))
    print(json.dumps({"output": str(args.output.resolve()), "experiments": len(payload["experiments"])}))


if __name__ == "__main__":
    main()
