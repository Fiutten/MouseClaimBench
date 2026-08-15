"""Regression benchmark for integrity references and attestation consistency."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import yaml

from mousebrainbench import __version__
from mousebrainbench.artifacts import code_revision
from mousebrainbench.benchmarks.profile_v2_contract_mutation import (
    _block,
    _complete_blocks,
)
from mousebrainbench.benchmarks.profile_v2_provenance_attacks import _base_manifest
from mousebrainbench.knowledge import FinalAuthorizationSystem, load_authorization_profile_v2
from mousebrainbench.knowledge.integrity import (
    EvidenceAttestation,
    EvidencePackageManifest,
    IntegrityDeficitCode,
)
from mousebrainbench.validation.evidence_contract import EvidenceBlock, EvidenceStatus

DEFAULT_PROTOCOL = Path("configs/benchmarks/profile_v2_integrity_regression.yaml")
DEFAULT_OUTPUT = Path("results/profile_v2_integrity_regression/summary.json")
DEFAULT_MARKDOWN = Path("results/profile_v2_integrity_regression/summary.md")

# These are targeted software-regression families, not additional historical
# attack families. Their separation prevents the 13-type deficit taxonomy from
# being conflated with either integrity benchmark.
EXTENDED_REGRESSION_FAMILIES = (
    "dangling_attestation_artifact",
    "dangling_independence_left",
    "dangling_independence_right",
    "dangling_cohort_left",
    "dangling_cohort_right",
    "unknown_attested_block",
    "attestation_failed_block_passed",
    "attestation_passed_block_failed",
    "missing_block_attestation",
)


@dataclass(frozen=True)
class IntegrityRegressionCase:
    case_id: str
    claim: str
    attack: str | None
    blocks: dict[str, EvidenceBlock]
    manifest: EvidencePackageManifest
    expected: tuple[IntegrityDeficitCode, ...]


def _mutate(
    attack: str,
    blocks: dict[str, EvidenceBlock],
    manifest: EvidencePackageManifest,
) -> tuple[dict[str, EvidenceBlock], EvidencePackageManifest, tuple[IntegrityDeficitCode, ...]]:
    first_artifact = manifest.artifacts[0].artifact_id
    first_attestation = manifest.attestations[0]
    if attack == "dangling_attestation_artifact":
        attestations = (
            replace(first_attestation, artifact_id="missing-attestation-artifact"),
            *manifest.attestations[1:],
        )
        return blocks, replace(manifest, attestations=attestations), (
            IntegrityDeficitCode.UNKNOWN_PROVENANCE_REFERENCE,
        )
    if attack.startswith("dangling_independence_"):
        pair = (
            ("missing-independent-artifact", first_artifact)
            if attack.endswith("left")
            else (first_artifact, "missing-independent-artifact")
        )
        return blocks, replace(manifest, independent_artifact_pairs=(pair,)), (
            IntegrityDeficitCode.UNKNOWN_PROVENANCE_REFERENCE,
        )
    if attack.startswith("dangling_cohort_"):
        pair = (
            ("missing-cohort-artifact", first_artifact)
            if attack.endswith("left")
            else (first_artifact, "missing-cohort-artifact")
        )
        return blocks, replace(manifest, disjoint_cohort_pairs=(pair,)), (
            IntegrityDeficitCode.UNKNOWN_PROVENANCE_REFERENCE,
        )
    if attack == "unknown_attested_block":
        unknown = EvidenceAttestation(
            "nonexistent-block", EvidenceStatus.PASSED, first_artifact
        )
        return blocks, replace(
            manifest, attestations=(*manifest.attestations, unknown)
        ), (IntegrityDeficitCode.UNKNOWN_BLOCK_REFERENCE,)
    if attack == "attestation_failed_block_passed":
        attestations = (
            replace(first_attestation, status=EvidenceStatus.FAILED),
            *manifest.attestations[1:],
        )
        return blocks, replace(manifest, attestations=attestations), (
            IntegrityDeficitCode.ATTESTATION_BLOCK_STATUS_MISMATCH,
        )
    if attack == "attestation_passed_block_failed":
        updated = dict(blocks)
        updated[first_attestation.block_name] = _block(
            first_attestation.block_name, EvidenceStatus.FAILED
        )
        return updated, manifest, (
            IntegrityDeficitCode.ATTESTATION_BLOCK_STATUS_MISMATCH,
        )
    if attack == "missing_block_attestation":
        return blocks, replace(manifest, attestations=manifest.attestations[1:]), (
            IntegrityDeficitCode.MISSING_BLOCK_ATTESTATION,
        )
    raise ValueError(f"unknown integrity regression attack: {attack}")


def generate_cases() -> tuple[IntegrityRegressionCase, ...]:
    profile = load_authorization_profile_v2()
    cases: list[IntegrityRegressionCase] = []
    for requirement in profile.requirements:
        claim = requirement.claim
        pristine_blocks = _complete_blocks(claim)
        pristine_manifest = _base_manifest(claim)
        cases.append(
            IntegrityRegressionCase(
                f"{claim}__pristine",
                claim,
                None,
                pristine_blocks,
                pristine_manifest,
                (),
            )
        )
        for attack in EXTENDED_REGRESSION_FAMILIES:
            blocks, manifest, expected = _mutate(
                attack, _complete_blocks(claim), _base_manifest(claim)
            )
            cases.append(
                IntegrityRegressionCase(
                    f"{claim}__{attack}", claim, attack, blocks, manifest, expected
                )
            )
    return tuple(cases)


def evaluate(protocol: dict[str, Any]) -> dict[str, Any]:
    profile = load_authorization_profile_v2()
    cases = generate_cases()
    expected_cases = int(protocol["design"]["expected_cases"])
    if len(cases) != expected_cases:
        raise RuntimeError(f"generated {len(cases)} cases, expected {expected_cases}")
    exact_traces = false_authorizations = false_rejections = 0
    family_detection = {name: 0 for name in EXTENDED_REGRESSION_FAMILIES}
    for case in cases:
        decision = FinalAuthorizationSystem(
            profile, case.blocks, case.manifest
        ).infer(case.claim)
        observed = tuple(row.code for row in decision.integrity_deficits)
        exact_traces += int(observed == case.expected)
        false_authorizations += int(case.attack is not None and decision.authorized)
        false_rejections += int(case.attack is None and not decision.authorized)
        if case.attack is not None:
            family_detection[case.attack] += int(observed == case.expected)
    endpoints = {
        "exact_integrity_traces_equal_1": exact_traces == len(cases),
        "false_authorizations_equal_0": false_authorizations == 0,
        "false_rejections_equal_0": false_rejections == 0,
        "every_regression_family_detected_for_all_claims": all(
            count == len(profile.requirements) for count in family_detection.values()
        ),
    }
    return {
        "cases": len(cases),
        "pristine_cases": len(profile.requirements),
        "attacked_cases": len(cases) - len(profile.requirements),
        "regression_families": list(EXTENDED_REGRESSION_FAMILIES),
        "exact_integrity_traces": exact_traces,
        "exact_integrity_trace_rate": exact_traces / len(cases),
        "false_authorizations": false_authorizations,
        "false_rejections": false_rejections,
        "family_detection": family_detection,
        "endpoints": endpoints,
        "all_endpoints_passed": all(endpoints.values()),
        "interpretation": protocol["interpretation"],
    }


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
        "analysis": "profile_v2_extended_integrity_regression",
        "protocol": str(protocol_path),
        "protocol_sha256": hashlib.sha256(protocol_path.read_bytes()).hexdigest(),
        **assessment,
        "decision": (
            "extended_integrity_regression_confirmed"
            if assessment["all_endpoints_passed"]
            else "extended_integrity_regression_failed"
        ),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n")
    lines = [
        "# Extended profile-v2 integrity regression",
        "",
        f"- Decision: `{payload['decision']}`",
        f"- Cases: `{payload['cases']}`",
        f"- Attacked cases: `{payload['attacked_cases']}`",
        f"- Exact trace rate: `{payload['exact_integrity_trace_rate']:.4f}`",
        "",
        payload["interpretation"],
        "",
    ]
    markdown.parent.mkdir(parents=True, exist_ok=True)
    markdown.write_text("\n".join(lines))
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
