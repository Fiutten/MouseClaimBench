"""Check formal profile-v2 properties over deterministic evidence packages."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from mousebrainbench import __version__
from mousebrainbench.artifacts import code_revision
from mousebrainbench.benchmarks.profile_v2_contract_mutation import (
    _block,
    _complete_blocks,
)
from mousebrainbench.knowledge import (
    ClaimAuthorizationSystem,
    ProfileAuthorizationStatus,
    load_authorization_profile_v2,
)
from mousebrainbench.validation.evidence_contract import EvidenceBlock, EvidenceStatus

DEFAULT_PROTOCOL = Path("configs/benchmarks/profile_v2_formal_properties.yaml")
DEFAULT_OUTPUT = Path("results/profile_v2_formal_properties/summary.json")
DEFAULT_MARKDOWN = Path("results/profile_v2_formal_properties/summary.md")


def _deficits(decision) -> tuple[tuple[str, EvidenceStatus], ...]:
    return tuple(
        sorted(
            ((fact.name, fact.effective_status) for fact in decision.deficits),
            key=lambda item: item[0],
        )
    )


def _random_package(
    claim: str,
    rng: np.random.Generator,
) -> dict[str, EvidenceBlock]:
    blocks = _complete_blocks(claim)
    statuses = tuple(EvidenceStatus)
    for name in tuple(blocks):
        draw = int(rng.integers(0, len(statuses) + 2))
        if draw < len(statuses):
            blocks[name] = _block(name, statuses[draw])
        elif draw == len(statuses):
            del blocks[name]
        else:
            observations = dict(blocks[name].observations)
            first_field = next(iter(observations))
            del observations[first_field]
            blocks[name] = _block(name, observations=observations)
    return blocks


def _irrelevant_block() -> EvidenceBlock:
    return EvidenceBlock.from_mapping(
        name="unprofiled_irrelevant_evidence",
        status=EvidenceStatus.FAILED,
        source="controlled formal-property case",
        rule="the block is not required by any profile claim",
        rationale="irrelevant evidence must not alter a declared claim",
        observations={"payload": "deliberately irrelevant"},
    )


def evaluate(protocol: dict[str, Any]) -> dict[str, Any]:
    """Evaluate semantic equivalences and metamorphic properties."""

    profile = load_authorization_profile_v2()
    random_config = protocol["deterministic_random_packages"]
    rng = np.random.default_rng(int(random_config["seed"]))
    per_claim = int(random_config["packages_per_claim"])
    counters = {
        name: {"checks": 0, "violations": 0}
        for name in protocol["properties"]
    }

    def record(name: str, passed: bool) -> None:
        counters[name]["checks"] += 1
        counters[name]["violations"] += int(not passed)

    for requirement in profile.requirements:
        claim = requirement.claim
        for _ in range(per_claim):
            blocks = _random_package(claim, rng)
            decision = ClaimAuthorizationSystem(profile, blocks).infer(claim)
            all_passed = bool(decision.facts) and all(
                fact.effective_status is EvidenceStatus.PASSED
                for fact in decision.facts
            )
            expected_deficits = tuple(
                sorted(
                    (
                        (fact.name, fact.effective_status)
                        for fact in decision.facts
                        if fact.effective_status is not EvidenceStatus.PASSED
                    ),
                    key=lambda item: item[0],
                )
            )
            record(
                "authorization_soundness_relative_to_profile",
                not decision.authorized or all_passed,
            )
            record(
                "authorization_completeness_relative_to_profile",
                not all_passed or decision.authorized,
            )
            record("complete_deficit_identity", _deficits(decision) == expected_deficits)

            reversed_blocks = dict(reversed(tuple(blocks.items())))
            reversed_decision = ClaimAuthorizationSystem(profile, reversed_blocks).infer(
                claim
            )
            record(
                "invariance_to_input_order",
                reversed_decision.status is decision.status
                and _deficits(reversed_decision) == _deficits(decision),
            )

            extended = dict(blocks)
            extended["unprofiled_irrelevant_evidence"] = _irrelevant_block()
            extended_decision = ClaimAuthorizationSystem(profile, extended).infer(claim)
            record(
                "invariance_to_irrelevant_evidence",
                extended_decision.status is decision.status
                and _deficits(extended_decision) == _deficits(decision),
            )

            degradable = next(
                (
                    fact.name
                    for fact in decision.facts
                    if fact.effective_status is EvidenceStatus.PASSED
                ),
                None,
            )
            if degradable is not None:
                degraded = dict(blocks)
                degraded[degradable] = _block(degradable, EvidenceStatus.FAILED)
                degraded_decision = ClaimAuthorizationSystem(profile, degraded).infer(
                    claim
                )
                record(
                    "monotonicity_under_evidence_degradation",
                    not degraded_decision.authorized
                    and set(_deficits(decision)) <= set(_deficits(degraded_decision)),
                )

    outside = ClaimAuthorizationSystem(profile, {}).infer(
        "undeclared_complete_mouse_consciousness"
    )
    record(
        "outside_profile_closure",
        outside.status is ProfileAuthorizationStatus.OUTSIDE_PROFILE
        and not outside.authorized
        and outside.required_blocks == (),
    )
    total_checks = sum(row["checks"] for row in counters.values())
    total_violations = sum(row["violations"] for row in counters.values())
    return {
        "profile_id": profile.profile_id,
        "profile_version": profile.version,
        "profile_hash": profile.source_hash,
        "random_packages": per_claim * len(profile.requirements),
        "properties": counters,
        "total_checks": total_checks,
        "total_violations": total_violations,
        "all_properties_hold": total_violations == 0,
        "claim_boundary": protocol["claim_boundary"],
    }


def _write_markdown(payload: dict[str, Any], path: Path) -> None:
    lines = [
        "# Profile v2 formal properties",
        "",
        f"- Decision: `{payload['decision']}`",
        f"- Random packages: `{payload['random_packages']}`",
        f"- Property checks: `{payload['total_checks']}`",
        f"- Violations: `{payload['total_violations']}`",
        "",
        "| Property | Checks | Violations |",
        "|---|---:|---:|",
    ]
    lines.extend(
        f"| `{name}` | {row['checks']} | {row['violations']} |"
        for name, row in payload["properties"].items()
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
        "analysis": "profile_v2_formal_property_audit",
        "protocol": str(protocol_path),
        "protocol_sha256": hashlib.sha256(protocol_path.read_bytes()).hexdigest(),
        **assessment,
        "decision": (
            "formal_properties_confirmed"
            if assessment["all_properties_hold"]
            else "formal_property_violation_detected"
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
