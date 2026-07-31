"""Gate a MICrONS pilot without opening a large-data integration front.

MICrONS is scientifically valuable but too large to treat casually. This gate
only approves a pilot when a small local manifest already exposes the minimum
structure-function fields needed for a falsifiable analysis.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from mousebrainbench import __version__
from mousebrainbench.artifacts import code_revision


DEFAULT_MANIFEST = Path("data/microns/pilot_manifest.json")
DEFAULT_OUTPUT = Path("results/microns_pilot_gate/summary.json")
DEFAULT_MARKDOWN = Path("results/microns_pilot_gate/summary.md")

REQUIRED_FIELDS = (
    "n_neurons",
    "has_spatial_coordinates",
    "has_functional_responses",
    "has_structural_edges",
    "estimated_download_gb",
)


def _load_manifest(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text())


def evaluate_manifest(path: Path = DEFAULT_MANIFEST) -> dict[str, Any]:
    """Evaluate whether a MICrONS pilot is currently actionable."""

    manifest = _load_manifest(path)
    if manifest is None:
        return {
            "version": __version__,
            "git_revision": code_revision(),
            "manifest": str(path),
            "manifest_available": False,
            "missing_fields": list(REQUIRED_FIELDS),
            "decision": "defer_microns_until_small_manifest_available",
            "approved": False,
            "reason": (
                "No local small manifest is available. Downloading or integrating "
                "MICrONS wholesale would violate the bounded-pilot requirement."
            ),
            "approved_question": None,
        }

    missing = [field for field in REQUIRED_FIELDS if field not in manifest]
    estimated_download_gb = float(manifest.get("estimated_download_gb", float("inf")))
    n_structural_edges = int(manifest.get("n_structural_edges", 0))
    q1_approved = (
        not missing
        and bool(manifest.get("has_spatial_coordinates"))
        and bool(manifest.get("has_functional_responses"))
        and bool(manifest.get("has_structural_edges"))
        and int(manifest.get("n_neurons", 0)) >= 500
        and n_structural_edges >= 500
        and estimated_download_gb <= 5.0
    )
    micro_approved = (
        not missing
        and bool(manifest.get("has_spatial_coordinates"))
        and bool(manifest.get("has_functional_responses"))
        and bool(manifest.get("has_structural_edges"))
        and int(manifest.get("n_neurons", 0)) >= 100
        and n_structural_edges >= 20
        and estimated_download_gb <= 1.0
    )
    return {
        "version": __version__,
        "git_revision": code_revision(),
        "manifest": str(path),
        "manifest_available": True,
        "manifest_payload": manifest,
        "missing_fields": missing,
        "decision": (
            "approve_bounded_microns_structure_function_pilot"
            if q1_approved
            else "approve_microns_structure_function_micro_pilot"
            if micro_approved
            else "defer_microns_pilot_manifest_insufficient"
        ),
        "approved": q1_approved,
        "micro_pilot_approved": micro_approved or q1_approved,
        "q1_pilot_approved": q1_approved,
        "reason": (
            "Manifest supports a Q1-scale bounded structure-function test."
            if q1_approved
            else "Manifest supports a small structure-function micro-pilot, not Q1 scale."
            if micro_approved
            else "Manifest does not yet satisfy the minimum bounded-pilot criteria."
        ),
        "approved_question": (
            "Does local structural wiring or spatial embedding explain functional "
            "similarity beyond matched nulls in a bounded MICrONS subset?"
            if q1_approved or micro_approved
            else None
        ),
    }


def write_outputs(payload: dict[str, Any], output: Path, markdown: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2))
    lines = [
        "# MICrONS Pilot Gate",
        "",
        f"- Decision: `{payload['decision']}`",
        f"- Approved: `{payload['approved']}`",
        f"- Micro-pilot approved: `{payload.get('micro_pilot_approved', False)}`",
        f"- Q1 pilot approved: `{payload.get('q1_pilot_approved', False)}`",
        f"- Manifest available: `{payload['manifest_available']}`",
        f"- Reason: {payload['reason']}",
        "",
        "## Bounded pilot rule",
        "",
        (
            "MICrONS only enters this project if a small manifest exposes spatial "
            "coordinates, functional responses, structural edges, at least 500 "
            "neurons, and an estimated download of at most 5 GB."
        ),
        "",
    ]
    if payload.get("approved_question"):
        lines.extend(["## Approved question", "", str(payload["approved_question"]), ""])
    markdown.parent.mkdir(parents=True, exist_ok=True)
    markdown.write_text("\n".join(lines))


def run(
    output: Path = DEFAULT_OUTPUT,
    markdown: Path = DEFAULT_MARKDOWN,
    manifest: Path = DEFAULT_MANIFEST,
) -> Path:
    payload = evaluate_manifest(manifest)
    write_outputs(payload, output, markdown)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--markdown", type=Path, default=DEFAULT_MARKDOWN)
    args = parser.parse_args()
    print(json.dumps({"output": str(run(args.output, args.markdown, args.manifest).resolve())}))


if __name__ == "__main__":
    main()
