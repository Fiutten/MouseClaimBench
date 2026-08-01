"""Validate the preregisterable expert study and build unlabeled item packets."""

from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path
from typing import Any

import yaml

from mousebrainbench import __version__
from mousebrainbench.artifacts import code_revision


DEFAULT_PROTOCOL = Path("configs/human_evaluation_protocol.yaml")
DEFAULT_CANDIDATES = Path("results/llm_claim_extraction_audit/summary.json")
DEFAULT_OUTPUT = Path("results/human_evaluation_protocol/study_package.json")
DEFAULT_TEMPLATE = Path("results/human_evaluation_protocol/annotation_template.csv")

REQUIRED_REFERENCE_LABELS = {
    "supportable",
    "unsupported",
    "insufficient_evidence",
    "out_of_scope",
}


def _load_yaml(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text())


def _validate_protocol(protocol: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    governance = protocol.get("governance", {})
    if protocol.get("status") != "not_executed":
        errors.append("protocol status must remain 'not_executed' before data collection")
    if governance.get("no_results_available") is not True:
        errors.append("governance.no_results_available must be true")
    if governance.get("ethics_approval_required_before_recruitment") is not True:
        errors.append("ethics approval must be required before recruitment")
    if governance.get("preregistration_required") is not True:
        errors.append("preregistration must be required")
    primary = protocol.get("outcomes", {}).get("primary", {})
    if primary.get("analysis") != "mixed_effects_logistic_regression":
        errors.append("primary analysis must account for repeated participant and item decisions")
    random_effects = set(primary.get("random_effects", []))
    if not {"participant", "item"}.issubset(random_effects):
        errors.append("primary analysis must include participant and item random effects")
    labels = set(
        protocol.get("reference_standard", {}).get("permitted_reference_labels", [])
    )
    if labels != REQUIRED_REFERENCE_LABELS:
        errors.append("reference label inventory does not match the declared four-state standard")
    if protocol.get("reporting", {}).get("prohibit_claim_of_human_benefit_before_execution") is not True:
        errors.append("claims of human benefit must be prohibited before execution")
    return errors


def _sample_candidates(
    candidates: list[dict[str, Any]],
    *,
    maximum_per_type: int,
    target_count: int,
    seed: int,
) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for candidate in candidates:
        grouped.setdefault(str(candidate.get("claim_type", "unclassified")), []).append(candidate)
    rng = random.Random(seed)
    selected: list[dict[str, Any]] = []
    for claim_type in sorted(grouped):
        rows = list(grouped[claim_type])
        rng.shuffle(rows)
        selected.extend(rows[:maximum_per_type])
    if len(selected) > target_count:
        rng.shuffle(selected)
        selected = selected[:target_count]
    return selected


def _write_annotation_template(path: Path, items: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = (
        "item_id",
        "claim_type",
        "claim_text",
        "source",
        "annotator_id",
        "reference_label",
        "confidence_1_to_5",
        "rationale",
        "adjudication_status",
    )
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for item in items:
            writer.writerow(
                {
                    "item_id": item["item_id"],
                    "claim_type": item["claim_type"],
                    "claim_text": item["claim_text"],
                    "source": item["source"],
                    "annotator_id": "",
                    "reference_label": "",
                    "confidence_1_to_5": "",
                    "rationale": "",
                    "adjudication_status": "",
                }
            )


def run(
    protocol_path: Path = DEFAULT_PROTOCOL,
    candidates_path: Path = DEFAULT_CANDIDATES,
    output: Path = DEFAULT_OUTPUT,
    template: Path = DEFAULT_TEMPLATE,
) -> Path:
    """Validate protocol invariants and create an explicitly unlabeled package."""

    protocol = _load_yaml(protocol_path)
    errors = _validate_protocol(protocol)
    candidate_payload = json.loads(candidates_path.read_text())
    item_config = protocol["items"]
    selected = _sample_candidates(
        candidate_payload.get("candidates", []),
        maximum_per_type=int(item_config["maximum_per_claim_type"]),
        target_count=int(item_config["target_count"]),
        seed=int(item_config["random_seed"]),
    )
    items = [
        {
            "item_id": f"MCB-{index:03d}",
            "claim_type": row.get("claim_type"),
            "claim_text": row.get("text"),
            "source": row.get("source"),
            "reference_label": None,
            "human_annotations": [],
        }
        for index, row in enumerate(selected, start=1)
    ]
    _write_annotation_template(template, items)
    payload = {
        "version": __version__,
        "git_revision": code_revision(),
        "analysis": "human_expert_evaluation_protocol_preparation",
        "study_id": protocol.get("study_id"),
        "study_status": "not_executed",
        "protocol_valid": not errors,
        "protocol_errors": errors,
        "source_candidates": len(candidate_payload.get("candidates", [])),
        "selected_unlabeled_items": len(items),
        "items": items,
        "annotation_template": str(template),
        "results_available": False,
        "decision": (
            "human_study_protocol_ready_for_ethics_and_preregistration"
            if not errors and items
            else "human_study_protocol_requires_revision"
        ),
        "prohibited_interpretation": (
            "This preparation artifact is not evidence that MouseClaimBench improves human decisions."
        ),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2))
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--candidates", type=Path, default=DEFAULT_CANDIDATES)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--template", type=Path, default=DEFAULT_TEMPLATE)
    args = parser.parse_args()
    print(
        json.dumps(
            {
                "output": str(
                    run(args.protocol, args.candidates, args.output, args.template).resolve()
                )
            }
        )
    )


if __name__ == "__main__":
    main()
