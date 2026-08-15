"""Stress all declared integrity-attack compositions and explicit trust limits."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from dataclasses import replace
from itertools import chain, combinations
from pathlib import Path
from typing import Any

import yaml

from mousebrainbench import __version__
from mousebrainbench.artifacts import code_revision
from mousebrainbench.benchmarks.profile_v2_contract_mutation import _complete_blocks
from mousebrainbench.benchmarks.profile_v2_provenance_attacks import (
    ORIGINAL_ATTACK_FAMILIES,
    _apply_attack,
    _base_manifest,
    _digest,
    _replace_artifact,
    expected_deficits_for_attacks,
)
from mousebrainbench.knowledge import load_authorization_profile_v2
from mousebrainbench.knowledge.integrity import DomainIntegrityAuthorizationSystem

DEFAULT_PROTOCOL = Path("configs/benchmarks/profile_v2_compositional_integrity_stress.yaml")
DEFAULT_OUTPUT = Path("results/profile_v2_compositional_integrity_stress/summary.json")
DEFAULT_MARKDOWN = Path("results/profile_v2_compositional_integrity_stress/summary.md")


def _all_subsets(values: tuple[str, ...]) -> tuple[tuple[str, ...], ...]:
    return tuple(chain.from_iterable(combinations(values, size) for size in range(len(values) + 1)))


def _coherent_content_replacement(manifest):
    first = manifest.artifacts[0]
    replacement = _digest(f"coherent-replacement::{first.artifact_id}")
    return _replace_artifact(
        manifest,
        0,
        replace(first, declared_sha256=replacement, observed_sha256=replacement),
    )


def _coherent_independence_forgery(manifest):
    first, second = manifest.artifacts[:2]
    updated = _replace_artifact(
        manifest,
        0,
        replace(
            first,
            cohorts=(f"declared-{first.artifact_id}",),
            study_id=f"declared-study-{first.artifact_id}",
            data_generation_id=f"declared-generation-{first.artifact_id}",
        ),
    )
    updated = _replace_artifact(
        updated,
        1,
        replace(
            second,
            cohorts=(f"declared-{second.artifact_id}",),
            study_id=f"declared-study-{second.artifact_id}",
            data_generation_id=f"declared-generation-{second.artifact_id}",
        ),
    )
    pair = (first.artifact_id, second.artifact_id)
    return replace(
        updated,
        independent_artifact_pairs=(*updated.independent_artifact_pairs, pair),
        disjoint_cohort_pairs=(*updated.disjoint_cohort_pairs, pair),
    )


def evaluate(protocol: dict[str, Any]) -> dict[str, Any]:
    """Evaluate every attack subset and two deliberately unobservable forgeries."""

    profile = load_authorization_profile_v2()
    attacks = ORIGINAL_ATTACK_FAMILIES
    expected_families = int(protocol["in_model_design"]["attack_families"])
    if len(attacks) != expected_families:
        raise ValueError(
            "the compositional protocol must use the eight historical attack families"
        )
    subsets = _all_subsets(attacks)
    in_model_packages = attacked_packages = false_authorizations = exact_traces = 0
    by_order: Counter[int] = Counter()
    for requirement in profile.requirements:
        blocks = _complete_blocks(requirement.claim)
        for selected in subsets:
            manifest = _base_manifest(requirement.claim)
            for attack in selected:
                manifest = _apply_attack(manifest, attack)
            decision = DomainIntegrityAuthorizationSystem(profile, blocks, manifest).infer(
                requirement.claim
            )
            expected = expected_deficits_for_attacks(tuple(selected))
            observed = tuple(row.code for row in decision.integrity_deficits)
            in_model_packages += 1
            attacked_packages += int(bool(selected))
            false_authorizations += int(bool(selected) and decision.authorized)
            exact_traces += int(observed == expected)
            by_order[len(selected)] += 1

    boundary_rows = []
    boundary_builders = {
        "coherent_content_and_hash_replacement": _coherent_content_replacement,
        "coherent_independence_metadata_forgery": _coherent_independence_forgery,
    }
    for requirement in profile.requirements:
        blocks = _complete_blocks(requirement.claim)
        for name, builder in boundary_builders.items():
            manifest = builder(_base_manifest(requirement.claim))
            decision = DomainIntegrityAuthorizationSystem(profile, blocks, manifest).infer(
                requirement.claim
            )
            boundary_rows.append(
                {
                    "claim": requirement.claim,
                    "negative_control": name,
                    "authorized": decision.authorized,
                    "integrity_deficits": [row.code.value for row in decision.integrity_deficits],
                }
            )

    expected_in_model = int(protocol["in_model_design"]["expected_packages"])
    expected_boundary = int(protocol["trust_boundary_negative_controls"]["expected_cases"])
    endpoints = {
        "all_in_model_packages_generated": in_model_packages == expected_in_model,
        "no_false_authorization_for_declared_invariants": false_authorizations == 0,
        "exact_trace_for_every_declared_composition": exact_traces == in_model_packages,
        "all_trust_boundary_controls_generated": len(boundary_rows) == expected_boundary,
        "coherent_forgery_controls_escape_without_external_anchor": all(
            row["authorized"] and not row["integrity_deficits"] for row in boundary_rows
        ),
    }
    return {
        "in_model_packages": in_model_packages,
        "in_model_attacked_packages": attacked_packages,
        "in_model_false_authorizations": false_authorizations,
        "in_model_exact_traces": exact_traces,
        "in_model_exact_trace_rate": exact_traces / in_model_packages,
        "compositions_by_attack_order": {
            str(order): count for order, count in sorted(by_order.items())
        },
        "trust_boundary_negative_controls": len(boundary_rows),
        "trust_boundary_authorizations": sum(row["authorized"] for row in boundary_rows),
        "trust_boundary_rows": boundary_rows,
        "endpoints": endpoints,
        "all_endpoints_passed": all(endpoints.values()),
        "claim_boundary": protocol["claim_boundary"],
    }


def _write_markdown(payload: dict[str, Any], path: Path) -> None:
    lines = [
        "# Profile v2 compositional integrity stress",
        "",
        f"- Decision: `{payload['decision']}`",
        f"- In-model packages: `{payload['in_model_packages']}`",
        f"- In-model attacked packages: `{payload['in_model_attacked_packages']}`",
        f"- In-model false authorizations: `{payload['in_model_false_authorizations']}`",
        f"- Exact attack traces: `{payload['in_model_exact_traces']}`",
        f"- Trust-boundary negative controls: `{payload['trust_boundary_negative_controls']}`",
        f"- Expected trust-boundary authorizations: `{payload['trust_boundary_authorizations']}`",
        "",
        "| Attack order | Packages across ten claims |",
        "|---:|---:|",
    ]
    lines.extend(
        f"| {order} | {count} |" for order, count in payload["compositions_by_attack_order"].items()
    )
    lines.extend(("", payload["claim_boundary"], ""))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines))


def run(
    *,
    protocol_path: Path = DEFAULT_PROTOCOL,
    output: Path = DEFAULT_OUTPUT,
    markdown: Path = DEFAULT_MARKDOWN,
) -> Path:
    protocol = yaml.safe_load(protocol_path.read_text())
    assessment = evaluate(protocol)
    payload = {
        "version": __version__,
        "git_revision": code_revision(),
        "analysis": "profile_v2_compositional_integrity_stress",
        "protocol": str(protocol_path),
        "protocol_sha256": hashlib.sha256(protocol_path.read_bytes()).hexdigest(),
        **assessment,
        "decision": (
            "declared_compositions_confirmed_with_explicit_trust_boundary"
            if assessment["all_endpoints_passed"]
            else "compositional_integrity_stress_failed"
        ),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n")
    _write_markdown(payload, markdown)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--markdown", type=Path, default=DEFAULT_MARKDOWN)
    args = parser.parse_args()
    print(run(protocol_path=args.protocol, output=args.output, markdown=args.markdown).resolve())


if __name__ == "__main__":
    main()
