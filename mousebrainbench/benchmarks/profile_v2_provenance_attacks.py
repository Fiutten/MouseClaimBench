"""Evaluate profile-v2 authorization under provenance and dependence attacks."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass, replace
from itertools import combinations
from pathlib import Path
from typing import Any

import yaml

from mousebrainbench import __version__
from mousebrainbench.artifacts import code_revision
from mousebrainbench.benchmarks.profile_v2_contract_mutation import _complete_blocks
from mousebrainbench.knowledge import ClaimAuthorizationSystem, load_authorization_profile_v2
from mousebrainbench.knowledge.integrity import (
    ArtifactRecord,
    EvidenceAttestation,
    EvidencePackageManifest,
    IntegrityAwareAuthorizationSystem,
    IntegrityDeficitCode,
)
from mousebrainbench.validation.evidence_contract import EvidenceStatus

DEFAULT_PROTOCOL = Path("configs/benchmarks/profile_v2_provenance_attacks.yaml")
DEFAULT_OUTPUT = Path("results/profile_v2_provenance_attacks/summary.json")
DEFAULT_MARKDOWN = Path("results/profile_v2_provenance_attacks/summary.md")

ATTACK_TO_DEFICIT = {
    "artifact_hash_tampering": IntegrityDeficitCode.ARTIFACT_HASH_MISMATCH,
    "profile_version_substitution": IntegrityDeficitCode.PROFILE_IDENTITY_MISMATCH,
    "dangling_provenance_reference": IntegrityDeficitCode.UNKNOWN_PROVENANCE_REFERENCE,
    "circular_provenance": IntegrityDeficitCode.PROVENANCE_CYCLE,
    "duplicate_independent_artifact": IntegrityDeficitCode.DUPLICATE_INDEPENDENT_ARTIFACT,
    "overlapping_independent_cohorts": IntegrityDeficitCode.OVERLAPPING_INDEPENDENT_COHORTS,
    "contradictory_attestation": IntegrityDeficitCode.CONTRADICTORY_ATTESTATION,
    "missing_block_lineage": IntegrityDeficitCode.MISSING_BLOCK_LINEAGE,
}


@dataclass(frozen=True)
class AttackCase:
    case_id: str
    claim: str
    attacks: tuple[str, ...]
    manifest: EvidencePackageManifest
    expected_deficits: tuple[IntegrityDeficitCode, ...]


def _digest(value: str) -> str:
    return f"sha256:{hashlib.sha256(value.encode()).hexdigest()}"


def _base_manifest(claim: str) -> EvidencePackageManifest:
    profile = load_authorization_profile_v2()
    requirement = profile.requirement(claim)
    if requirement is None:
        raise ValueError(f"unknown claim: {claim}")
    artifacts = [
        ArtifactRecord(
            artifact_id=f"{claim}-{name}",
            declared_sha256=_digest(f"{claim}-{name}"),
            observed_sha256=_digest(f"{claim}-{name}"),
            cohorts=(f"mouse-{index}",),
            study_id=f"study-{index}",
            data_generation_id=f"generation-{index}",
        )
        for index, name in enumerate(requirement.required_blocks)
    ]
    if len(artifacts) < 2:
        artifacts.append(
            ArtifactRecord(
                artifact_id=f"{claim}-support",
                declared_sha256=_digest(f"{claim}-support"),
                observed_sha256=_digest(f"{claim}-support"),
                cohorts=("mouse-support",),
                study_id="study-support",
                data_generation_id="generation-support",
            )
        )
    return EvidencePackageManifest(
        package_id=f"{claim}-pristine",
        profile_id=profile.profile_id,
        profile_version=profile.version,
        profile_hash=profile.source_hash,
        artifacts=tuple(artifacts),
        block_artifacts=tuple(
            (name, (f"{claim}-{name}",)) for name in requirement.required_blocks
        ),
        attestations=tuple(
            EvidenceAttestation(name, EvidenceStatus.PASSED, f"{claim}-{name}")
            for name in requirement.required_blocks
        ),
    )


def _replace_artifact(
    manifest: EvidencePackageManifest,
    index: int,
    artifact: ArtifactRecord,
) -> EvidencePackageManifest:
    artifacts = list(manifest.artifacts)
    artifacts[index] = artifact
    return replace(manifest, artifacts=tuple(artifacts))


def _apply_attack(
    manifest: EvidencePackageManifest,
    attack: str,
) -> EvidencePackageManifest:
    first, second = manifest.artifacts[:2]
    if attack == "artifact_hash_tampering":
        return _replace_artifact(
            manifest, 0, replace(first, observed_sha256=_digest("tampered-content"))
        )
    if attack == "profile_version_substitution":
        return replace(manifest, profile_version="0.0.0-substituted")
    if attack == "dangling_provenance_reference":
        return _replace_artifact(
            manifest, 0, replace(first, derived_from=(*first.derived_from, "missing-artifact"))
        )
    if attack == "circular_provenance":
        updated = _replace_artifact(
            manifest,
            0,
            replace(first, derived_from=(*first.derived_from, second.artifact_id)),
        )
        current_second = updated.artifacts[1]
        return _replace_artifact(
            updated,
            1,
            replace(
                current_second,
                derived_from=(*current_second.derived_from, first.artifact_id),
            ),
        )
    if attack == "duplicate_independent_artifact":
        updated_second = replace(
            second,
            declared_sha256=first.declared_sha256,
            observed_sha256=first.declared_sha256,
            study_id=first.study_id,
            data_generation_id=first.data_generation_id,
        )
        updated = _replace_artifact(manifest, 1, updated_second)
        pair = (first.artifact_id, second.artifact_id)
        return replace(
            updated,
            independent_artifact_pairs=(*updated.independent_artifact_pairs, pair),
        )
    if attack == "overlapping_independent_cohorts":
        updated_first = replace(first, cohorts=(*first.cohorts, "mouse-overlap"))
        updated_second = replace(second, cohorts=(*second.cohorts, "mouse-overlap"))
        updated = _replace_artifact(manifest, 0, updated_first)
        updated = _replace_artifact(updated, 1, updated_second)
        pair = (first.artifact_id, second.artifact_id)
        return replace(
            updated,
            disjoint_cohort_pairs=(*updated.disjoint_cohort_pairs, pair),
        )
    if attack == "contradictory_attestation":
        block_name = manifest.attestations[0].block_name
        conflict = EvidenceAttestation(
            block_name, EvidenceStatus.FAILED, first.artifact_id
        )
        return replace(manifest, attestations=(*manifest.attestations, conflict))
    if attack == "missing_block_lineage":
        return replace(manifest, block_artifacts=manifest.block_artifacts[1:])
    raise ValueError(f"unknown attack: {attack}")


def generate_cases(protocol: dict[str, Any]) -> tuple[AttackCase, ...]:
    profile = load_authorization_profile_v2()
    attacks = tuple(protocol["attack_families"])
    cases: list[AttackCase] = []
    attack_sets = ((), *((name,) for name in attacks), *combinations(attacks, 2))
    for requirement in profile.requirements:
        for selected in attack_sets:
            manifest = _base_manifest(requirement.claim)
            for attack in selected:
                manifest = _apply_attack(manifest, attack)
            identifier = "pristine" if not selected else "__".join(selected)
            cases.append(
                AttackCase(
                    case_id=f"{requirement.claim}__{identifier}",
                    claim=requirement.claim,
                    attacks=tuple(selected),
                    manifest=replace(
                        manifest,
                        package_id=f"{requirement.claim}__{identifier}",
                    ),
                    expected_deficits=tuple(
                        sorted(
                            (ATTACK_TO_DEFICIT[name] for name in selected),
                            key=lambda value: value.value,
                        )
                    ),
                )
            )
    expected = int(protocol["design"]["expected_cases"])
    if len(cases) != expected:
        raise RuntimeError(f"attack generator produced {len(cases)} rather than {expected}")
    return tuple(cases)


def evaluate(protocol: dict[str, Any]) -> dict[str, Any]:
    profile = load_authorization_profile_v2()
    cases = generate_cases(protocol)
    full_false_authorizations = 0
    full_false_rejections = 0
    exact_traces = 0
    core_false_authorizations = 0
    hash_only_false_authorizations = 0
    family_detection = {name: {"cases": 0, "detected": 0} for name in ATTACK_TO_DEFICIT}
    for case in cases:
        blocks = _complete_blocks(case.claim)
        full = IntegrityAwareAuthorizationSystem(
            profile, blocks, case.manifest
        ).infer(case.claim)
        observed = tuple(row.code for row in full.integrity_deficits)
        attacked = bool(case.attacks)
        full_false_authorizations += int(attacked and full.authorized)
        full_false_rejections += int(not attacked and not full.authorized)
        exact_traces += int(observed == case.expected_deficits)
        core = ClaimAuthorizationSystem(profile, blocks).infer(case.claim)
        core_false_authorizations += int(attacked and core.authorized)
        has_hash_failure = (
            IntegrityDeficitCode.ARTIFACT_HASH_MISMATCH in observed
        )
        hash_only_authorized = core.authorized and not has_hash_failure
        hash_only_false_authorizations += int(attacked and hash_only_authorized)
        for attack in case.attacks:
            family_detection[attack]["cases"] += 1
            family_detection[attack]["detected"] += int(
                ATTACK_TO_DEFICIT[attack] in observed
            )
    attacked_cases = sum(bool(case.attacks) for case in cases)
    pristine_cases = len(cases) - attacked_cases
    endpoints = {
        "full_integrity_gate_false_authorizations_equal_0": (
            full_false_authorizations == 0
        ),
        "full_integrity_gate_false_rejections_equal_0": full_false_rejections == 0,
        "exact_attack_trace_rate_equal_1": exact_traces == len(cases),
        "core_profile_false_authorizes_at_least_one_attack": (
            core_false_authorizations > 0
        ),
        "hash_only_baseline_false_authorizes_at_least_one_attack": (
            hash_only_false_authorizations > 0
        ),
    }
    return {
        "cases": len(cases),
        "pristine_cases": pristine_cases,
        "attacked_cases": attacked_cases,
        "full_integrity_gate": {
            "false_authorizations": full_false_authorizations,
            "false_rejections": full_false_rejections,
            "exact_attack_traces": exact_traces,
            "exact_attack_trace_rate": exact_traces / len(cases),
        },
        "comparators": {
            "profile_without_integrity_false_authorizations": core_false_authorizations,
            "hash_only_false_authorizations": hash_only_false_authorizations,
        },
        "family_detection": family_detection,
        "endpoints": endpoints,
        "all_endpoints_passed": all(endpoints.values()),
        "claim_boundary": protocol["claim_boundary"],
    }


def _write_markdown(payload: dict[str, Any], path: Path) -> None:
    gate = payload["full_integrity_gate"]
    lines = [
        "# Profile v2 provenance attacks",
        "",
        f"- Decision: `{payload['decision']}`",
        f"- Cases: `{payload['cases']}`",
        f"- Attacked cases: `{payload['attacked_cases']}`",
        f"- Full-gate false authorizations: `{gate['false_authorizations']}`",
        f"- Exact attack-trace rate: `{gate['exact_attack_trace_rate']:.4f}`",
        "",
        "| Attack | Cases | Detected |",
        "|---|---:|---:|",
    ]
    lines.extend(
        f"| `{name}` | {row['cases']} | {row['detected']} |"
        for name, row in payload["family_detection"].items()
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
        "analysis": "profile_v2_provenance_attack_benchmark",
        "protocol": str(protocol_path),
        "protocol_sha256": hashlib.sha256(protocol_path.read_bytes()).hexdigest(),
        **assessment,
        "decision": (
            "provenance_attack_gate_confirmed"
            if assessment["all_endpoints_passed"]
            else "provenance_attack_gate_failed"
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
