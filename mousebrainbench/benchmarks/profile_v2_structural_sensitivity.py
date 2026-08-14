"""Measure profile-v2 decision sensitivity to structural policy changes.

The benchmark holds every evidence package fixed and changes only the policy.
It evaluates the baseline profile, each of 60 leave-one-relation-out profiles,
and each of 160 one-block conservative extensions. The result measures policy
dependence. It neither selects a preferred profile nor supplies content validity.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import yaml

from mousebrainbench import __version__
from mousebrainbench.artifacts import code_revision
from mousebrainbench.benchmarks.profile_v2_artifact_application import (
    STRICT_TWIN,
    build_cases,
)
from mousebrainbench.knowledge import (
    ClaimAuthorizationProfile,
    ClaimAuthorizationSystem,
    load_authorization_profile_v2,
)
from mousebrainbench.validation.evidence_contract import EvidenceBlock, EvidenceStatus

DEFAULT_PROTOCOL = Path("configs/benchmarks/profile_v2_structural_sensitivity.yaml")
DEFAULT_OUTPUT = Path("results/profile_v2_structural_sensitivity/summary.json")
DEFAULT_MARKDOWN = Path("results/profile_v2_structural_sensitivity/summary.md")
DEFAULT_ROWS = Path("results/profile_v2_structural_sensitivity/variants.csv")


@dataclass(frozen=True)
class SensitivityCase:
    case_id: str
    source_role: str
    claim: str
    blocks: dict[str, EvidenceBlock]


@dataclass(frozen=True)
class ProfileVariant:
    variant_id: str
    operation: str
    claim: str | None
    evidence_block: str | None
    profile: ClaimAuthorizationProfile


def _blocks_from_decision(decision: dict[str, Any]) -> dict[str, EvidenceBlock]:
    return {
        row["name"]: EvidenceBlock.from_mapping(
            name=row["name"],
            status=EvidenceStatus(row["declared_status"]),
            source=row["source"],
            rule=row["rule"],
            rationale=row["rationale"],
            observations=row.get("observations", {}),
        )
        for row in decision["facts"]
    }


def build_sensitivity_cases(protocol: dict[str, Any]) -> tuple[SensitivityCase, ...]:
    artifact_protocol = yaml.safe_load(Path(protocol["artifact_protocol"]).read_text())
    cases: list[SensitivityCase] = []
    for case in build_cases(artifact_protocol):
        cases.extend(
            (
                SensitivityCase(
                    case_id=f"{case.name}__target",
                    source_role="retrospective_artifact",
                    claim=case.target_claim,
                    blocks=case.blocks,
                ),
                SensitivityCase(
                    case_id=f"{case.name}__strict_twin",
                    source_role="retrospective_artifact",
                    claim=STRICT_TWIN,
                    blocks=case.blocks,
                ),
            )
        )

    dandi = json.loads(Path(protocol["dandi_result"]).read_text())
    for row in dandi["applications"]:
        authorization = row["authorization"]
        blocks = _blocks_from_decision(authorization)
        resource = str(row["resource"]).replace(":", "_").lower()
        cases.extend(
            (
                SensitivityCase(
                    case_id=f"{resource}__target",
                    source_role="pre_access_frozen_dandi",
                    claim=authorization["claim"],
                    blocks=blocks,
                ),
                SensitivityCase(
                    case_id=f"{resource}__strict_twin",
                    source_role="pre_access_frozen_dandi",
                    claim=STRICT_TWIN,
                    blocks=blocks,
                ),
            )
        )
    expected = int(protocol["case_design"]["expected_decisions_per_profile"])
    if len(cases) != expected:
        raise RuntimeError(f"sensitivity case builder produced {len(cases)} rather than {expected}")
    return tuple(cases)


def _variant_profile(
    profile: ClaimAuthorizationProfile,
    claim: str,
    blocks: tuple[str, ...],
    variant_id: str,
) -> ClaimAuthorizationProfile:
    requirements = tuple(
        replace(requirement, required_blocks=blocks) if requirement.claim == claim else requirement
        for requirement in profile.requirements
    )
    return replace(
        profile,
        requirements=requirements,
        source_hash=f"{profile.source_hash}::{variant_id}",
    )


def build_variants() -> tuple[ProfileVariant, ...]:
    profile = load_authorization_profile_v2()
    block_names = tuple(item.name for item in profile.evidence_blocks)
    variants = [ProfileVariant("baseline", "baseline", None, None, profile)]
    for requirement in profile.requirements:
        for block in requirement.required_blocks:
            variant_id = f"remove__{requirement.claim}__{block}"
            variants.append(
                ProfileVariant(
                    variant_id,
                    "remove_required_relation",
                    requirement.claim,
                    block,
                    _variant_profile(
                        profile,
                        requirement.claim,
                        tuple(name for name in requirement.required_blocks if name != block),
                        variant_id,
                    ),
                )
            )
        for block in block_names:
            if block in requirement.required_blocks:
                continue
            variant_id = f"add__{requirement.claim}__{block}"
            variants.append(
                ProfileVariant(
                    variant_id,
                    "add_unrequired_relation",
                    requirement.claim,
                    block,
                    _variant_profile(
                        profile,
                        requirement.claim,
                        (*requirement.required_blocks, block),
                        variant_id,
                    ),
                )
            )
    return tuple(variants)


def _deficit_names(decision: Any) -> frozenset[str]:
    return frozenset(fact.name for fact in decision.deficits)


def _jaccard(left: frozenset[str], right: frozenset[str]) -> float:
    union = left | right
    return 1.0 if not union else len(left & right) / len(union)


def evaluate(protocol: dict[str, Any]) -> dict[str, Any]:
    """Evaluate all fixed cases under exhaustive one-relation perturbations."""

    cases = build_sensitivity_cases(protocol)
    variants = build_variants()
    baseline_profile = variants[0].profile
    baseline = {
        case.case_id: ClaimAuthorizationSystem(baseline_profile, case.blocks).infer(case.claim)
        for case in cases
    }
    rows: list[dict[str, Any]] = []
    for variant in variants[1:]:
        flips = expansions = contractions = deficit_changes = 0
        similarities = []
        changed_cases = []
        for case in cases:
            reference = baseline[case.case_id]
            observed = ClaimAuthorizationSystem(variant.profile, case.blocks).infer(case.claim)
            flipped = observed.authorized != reference.authorized
            flips += int(flipped)
            expansions += int(flipped and observed.authorized)
            contractions += int(flipped and reference.authorized)
            reference_deficits = _deficit_names(reference)
            observed_deficits = _deficit_names(observed)
            deficit_changes += int(reference_deficits != observed_deficits)
            similarities.append(_jaccard(reference_deficits, observed_deficits))
            if flipped:
                changed_cases.append(case.case_id)
        rows.append(
            {
                "variant_id": variant.variant_id,
                "operation": variant.operation,
                "claim": variant.claim,
                "evidence_block": variant.evidence_block,
                "decision_flips": flips,
                "authorization_expansions": expansions,
                "authorization_contractions": contractions,
                "deficit_set_changes": deficit_changes,
                "mean_deficit_jaccard": sum(similarities) / len(similarities),
                "changed_cases": changed_cases,
            }
        )

    removal = [row for row in rows if row["operation"] == "remove_required_relation"]
    addition = [row for row in rows if row["operation"] == "add_unrequired_relation"]
    baseline_authorized = sum(decision.authorized for decision in baseline.values())
    return {
        "fixed_cases": len(cases),
        "baseline_authorizations": baseline_authorized,
        "profile_variants": len(variants),
        "perturbed_profiles": len(rows),
        "relation_removal_variants": len(removal),
        "relation_addition_variants": len(addition),
        "profile_case_evaluations": len(variants) * len(cases),
        "removal_variants_with_decision_flips": sum(row["decision_flips"] > 0 for row in removal),
        "addition_variants_with_decision_flips": sum(row["decision_flips"] > 0 for row in addition),
        "authorization_expansions": sum(row["authorization_expansions"] for row in rows),
        "authorization_contractions": sum(row["authorization_contractions"] for row in rows),
        "variants_with_deficit_changes": sum(row["deficit_set_changes"] > 0 for row in rows),
        "decision_rows": rows,
        "baseline_rows": [
            {
                "case_id": case.case_id,
                "source_role": case.source_role,
                "claim": case.claim,
                "authorized": baseline[case.case_id].authorized,
                "deficits": sorted(_deficit_names(baseline[case.case_id])),
            }
            for case in cases
        ],
        "claim_boundary": protocol["claim_boundary"],
        "completed": len(removal) == 60 and len(addition) == 160,
    }


def _write_rows(payload: dict[str, Any], path: Path) -> None:
    fieldnames = (
        "variant_id",
        "operation",
        "claim",
        "evidence_block",
        "decision_flips",
        "authorization_expansions",
        "authorization_contractions",
        "deficit_set_changes",
        "mean_deficit_jaccard",
        "changed_cases",
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in payload["decision_rows"]:
            writer.writerow({**row, "changed_cases": "|".join(row["changed_cases"])})


def _write_markdown(payload: dict[str, Any], path: Path) -> None:
    changed = [row for row in payload["decision_rows"] if row["decision_flips"]]
    lines = [
        "# Profile v2 structural sensitivity",
        "",
        f"- Decision: `{payload['decision']}`",
        f"- Fixed artifact decisions per profile: `{payload['fixed_cases']}`",
        f"- Baseline authorizations: `{payload['baseline_authorizations']}`",
        f"- Profiles evaluated: `{payload['profile_variants']}`",
        f"- Profile-case evaluations: `{payload['profile_case_evaluations']}`",
        f"- Removal variants with a decision flip: `{payload['removal_variants_with_decision_flips']}`",
        f"- Addition variants with a decision flip: `{payload['addition_variants_with_decision_flips']}`",
        f"- Authorization expansions: `{payload['authorization_expansions']}`",
        f"- Authorization contractions: `{payload['authorization_contractions']}`",
        "",
        "| Policy perturbation | Claim | Evidence block | Flips | Changed cases |",
        "|---|---|---|---:|---|",
    ]
    lines.extend(
        f"| `{row['operation']}` | `{row['claim']}` | `{row['evidence_block']}` | "
        f"{row['decision_flips']} | {', '.join(row['changed_cases'])} |"
        for row in changed
    )
    lines.extend(("", payload["claim_boundary"], ""))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines))


def run(
    *,
    protocol_path: Path = DEFAULT_PROTOCOL,
    output: Path = DEFAULT_OUTPUT,
    markdown: Path = DEFAULT_MARKDOWN,
    rows: Path = DEFAULT_ROWS,
) -> Path:
    protocol = yaml.safe_load(protocol_path.read_text())
    assessment = evaluate(protocol)
    payload = {
        "version": __version__,
        "git_revision": code_revision(),
        "analysis": "profile_v2_structural_policy_sensitivity",
        "protocol": str(protocol_path),
        "protocol_sha256": hashlib.sha256(protocol_path.read_bytes()).hexdigest(),
        **assessment,
        "decision": (
            "structural_policy_sensitivity_completed"
            if assessment["completed"]
            else "structural_policy_sensitivity_incomplete"
        ),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n")
    _write_rows(payload, rows)
    _write_markdown(payload, markdown)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--markdown", type=Path, default=DEFAULT_MARKDOWN)
    parser.add_argument("--rows", type=Path, default=DEFAULT_ROWS)
    args = parser.parse_args()
    print(
        run(
            protocol_path=args.protocol,
            output=args.output,
            markdown=args.markdown,
            rows=args.rows,
        ).resolve()
    )


if __name__ == "__main__":
    main()
