"""Generate a complete, blank external-review packet from the frozen profile."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import yaml

DEFAULT_PROTOCOL = Path("configs/validation/knowledge_profile_external_review_v1.yaml")
DEFAULT_OUTPUT = Path("docs/knowledge_profile_external_review_v1")


def build_items(profile: dict[str, Any], basis: dict[str, Any]) -> list[dict[str, str]]:
    """Build relation, inference-rule, and coverage review statements."""

    meanings = {str(row["id"]): str(row["meaning"]) for row in profile["claims"]}
    items: list[dict[str, str]] = []
    for relation in basis["relations"]:
        claim = str(relation["claim"])
        block = str(relation["evidence_block"])
        items.append(
            {
                "item_id": f"relation__{claim}__{block}",
                "item_type": "evidence_to_claim_relation",
                "claim": claim,
                "evidence_block": block,
                "statement": f"For '{claim}', the block '{block}' has role: {relation['role']}.",
                "claim_meaning": meanings[claim],
                "rationale": str(relation["rationale"]),
                "scope": str(relation["scope"]),
                "exceptions": str(relation["exceptions"]),
                "alternatives_rejected": str(relation["alternatives_rejected"]),
                "source_ids": "|".join(str(value) for value in relation["source_ids"]),
            }
        )
    for rule in profile["inference_rules"]:
        items.append(
            {
                "item_id": f"rule__{rule['id']}",
                "item_type": "non_compensatory_inference_rule",
                "claim": "all_profile_claims",
                "evidence_block": str(rule["evidence_status"]),
                "statement": (
                    f"If {rule['quantifier']} required blocks have status "
                    f"'{rule['evidence_status']}', conclude '{rule['conclusion']}'."
                ),
                "claim_meaning": "Non-compensatory decision semantics",
                "rationale": str(rule["rationale"]),
                "scope": "All claims containing at least one required evidence block",
                "exceptions": "Reviewers must identify any claim-specific exception",
                "alternatives_rejected": "Compensation by unrelated passed evidence",
                "source_ids": "profile_rule",
            }
        )
    items.extend(
        (
            {
                "item_id": "coverage__claim_set",
                "item_type": "profile_comprehensiveness",
                "claim": "profile",
                "evidence_block": "claim_set",
                "statement": "The declared claim set covers the relevant claim classes for the stated context.",
                "claim_meaning": "Profile-level comprehensiveness",
                "rationale": "Missing claim classes can make a mapping appear valid while incomplete.",
                "scope": str(profile["domain"]),
                "exceptions": "The profile is not a universal scientific ontology.",
                "alternatives_rejected": "Inferring comprehensiveness from relation-level agreement alone.",
                "source_ids": "COSMIN_CONTENT_VALIDITY",
            },
            {
                "item_id": "coverage__evidence_block_set",
                "item_type": "profile_comprehensiveness",
                "claim": "profile",
                "evidence_block": "evidence_block_set",
                "statement": "The evidence-block set is sufficient for the profile's declared context of use.",
                "claim_meaning": "Profile-level comprehensiveness",
                "rationale": "A claim mapping can omit a necessary evidence class.",
                "scope": str(profile["domain"]),
                "exceptions": "Domain-specific protocols may add stricter blocks.",
                "alternatives_rejected": "Treating the current block list as exhaustive without review.",
                "source_ids": "COSMIN_CONTENT_VALIDITY",
            },
        )
    )
    return items


def _write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def run(protocol_path: Path = DEFAULT_PROTOCOL, output: Path = DEFAULT_OUTPUT) -> Path:
    protocol = yaml.safe_load(protocol_path.read_text())
    profile = yaml.safe_load(Path(protocol["profile"]).read_text())
    basis = yaml.safe_load(Path(protocol["basis"]).read_text())
    items = build_items(profile, basis)
    output.mkdir(parents=True, exist_ok=True)
    item_fields = list(items[0])
    _write_csv(output / "review_items.csv", items, item_fields)
    _write_csv(
        output / "ratings.csv",
        [],
        [
            "rater_id",
            "item_id",
            "relevance",
            "comprehensibility",
            "scientific_safety",
            "item_decision",
            "critical_veto",
            "comment",
        ],
    )
    _write_csv(
        output / "raters.csv",
        [],
        [
            "rater_id",
            "independent_of_authors",
            "conflict_disclosed",
            "years_relevant_experience",
            "causal_inference_or_causal_discovery",
            "computational_neuroscience_or_brain_modelling",
            "model_validation_reproducibility_or_digital_twins",
            "completed",
        ],
    )
    manifest = {
        "study_id": protocol["study_id"],
        "version": protocol["version"],
        "items": len(items),
        "relations": len(basis["relations"]),
        "inference_rules": len(profile["inference_rules"]),
        "profile_comprehensiveness_items": 2,
        "item_order_rule": protocol["rating"]["item_order"],
        "items_sha256": hashlib.sha256(
            json.dumps(items, sort_keys=True).encode()
        ).hexdigest(),
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    print(run(args.protocol, args.output).resolve())


if __name__ == "__main__":
    main()
