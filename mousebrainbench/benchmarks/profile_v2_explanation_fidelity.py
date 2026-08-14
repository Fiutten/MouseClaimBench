"""Verify counterfactual fidelity of complete profile-v2 deficit traces."""

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
from mousebrainbench.benchmarks.profile_v2_formal_properties import _random_package
from mousebrainbench.knowledge import ClaimAuthorizationSystem, load_authorization_profile_v2
from mousebrainbench.validation.evidence_contract import EvidenceBlock, EvidenceStatus

DEFAULT_PROTOCOL = Path("configs/benchmarks/profile_v2_explanation_fidelity.yaml")
DEFAULT_OUTPUT = Path("results/profile_v2_explanation_fidelity/summary.json")
DEFAULT_MARKDOWN = Path("results/profile_v2_explanation_fidelity/summary.md")


def _repair(
    blocks: dict[str, EvidenceBlock],
    deficit_names: tuple[str, ...],
) -> dict[str, EvidenceBlock]:
    repaired = dict(blocks)
    for name in deficit_names:
        repaired[name] = _block(name, EvidenceStatus.PASSED)
    return repaired


def evaluate(protocol: dict[str, Any]) -> dict[str, Any]:
    """Evaluate explanation sufficiency, minimality, and witness completeness."""

    profile = load_authorization_profile_v2()
    random_config = protocol["deterministic_packages"]
    rng = np.random.default_rng(int(random_config["seed"]))
    per_claim = int(random_config["packages_per_claim"])
    totals = {
        "random_packages": 0,
        "non_authorized_packages": 0,
        "repair_sufficiency_checks": 0,
        "repair_sufficiency_violations": 0,
        "repair_minimality_checks": 0,
        "repair_minimality_violations": 0,
        "witness_checks": 0,
        "witness_violations": 0,
        "pristine_degradation_checks": 0,
        "pristine_degradation_violations": 0,
        "multi_deficit_packages": 0,
        "reported_deficits_in_multi_deficit_packages": 0,
        "deficits_hidden_by_single_reason_trace": 0,
    }
    per_claim_rows = []

    for requirement in profile.requirements:
        claim_totals = {
            "claim": requirement.claim,
            "packages": per_claim,
            "non_authorized": 0,
            "multi_deficit": 0,
            "reported_deficits": 0,
            "single_reason_hidden_deficits": 0,
        }
        for _ in range(per_claim):
            blocks = _random_package(requirement.claim, rng)
            decision = ClaimAuthorizationSystem(profile, blocks).infer(requirement.claim)
            totals["random_packages"] += 1
            deficit_names = tuple(fact.name for fact in decision.deficits)
            for fact in decision.deficits:
                totals["witness_checks"] += 1
                witness_present = bool(
                    fact.name.strip()
                    and fact.source.strip()
                    and fact.rule.strip()
                    and fact.rationale.strip()
                )
                totals["witness_violations"] += int(not witness_present)

            if not decision.authorized:
                totals["non_authorized_packages"] += 1
                claim_totals["non_authorized"] += 1
                repaired_blocks = _repair(blocks, deficit_names)
                repaired = ClaimAuthorizationSystem(profile, repaired_blocks).infer(
                    requirement.claim
                )
                totals["repair_sufficiency_checks"] += 1
                totals["repair_sufficiency_violations"] += int(
                    not repaired.authorized or bool(repaired.deficits)
                )

                for retained_deficit in deficit_names:
                    partial = dict(repaired_blocks)
                    if retained_deficit in blocks:
                        partial[retained_deficit] = blocks[retained_deficit]
                    else:
                        partial.pop(retained_deficit, None)
                    partial_decision = ClaimAuthorizationSystem(profile, partial).infer(
                        requirement.claim
                    )
                    observed = tuple(fact.name for fact in partial_decision.deficits)
                    totals["repair_minimality_checks"] += 1
                    totals["repair_minimality_violations"] += int(
                        partial_decision.authorized or observed != (retained_deficit,)
                    )

            if len(deficit_names) > 1:
                hidden = len(deficit_names) - 1
                totals["multi_deficit_packages"] += 1
                totals["reported_deficits_in_multi_deficit_packages"] += len(deficit_names)
                totals["deficits_hidden_by_single_reason_trace"] += hidden
                claim_totals["multi_deficit"] += 1
                claim_totals["reported_deficits"] += len(deficit_names)
                claim_totals["single_reason_hidden_deficits"] += hidden

        pristine = _complete_blocks(requirement.claim)
        for block_name in requirement.required_blocks:
            degraded = dict(pristine)
            degraded[block_name] = _block(block_name, EvidenceStatus.FAILED)
            decision = ClaimAuthorizationSystem(profile, degraded).infer(requirement.claim)
            observed = tuple(fact.name for fact in decision.deficits)
            totals["pristine_degradation_checks"] += 1
            totals["pristine_degradation_violations"] += int(
                decision.authorized or observed != (block_name,)
            )
        per_claim_rows.append(claim_totals)

    violation_keys = [name for name in totals if name.endswith("_violations")]
    all_hold = all(totals[name] == 0 for name in violation_keys)
    return {
        **totals,
        "per_claim": per_claim_rows,
        "complete_trace_information_gain": (
            totals["reported_deficits_in_multi_deficit_packages"] / totals["multi_deficit_packages"]
            if totals["multi_deficit_packages"]
            else 1.0
        ),
        "all_explanation_properties_hold": all_hold,
        "claim_boundary": protocol["claim_boundary"],
    }


def _write_markdown(payload: dict[str, Any], path: Path) -> None:
    lines = [
        "# Profile v2 counterfactual explanation fidelity",
        "",
        f"- Decision: `{payload['decision']}`",
        f"- Random packages: `{payload['random_packages']}`",
        f"- Non-authorized packages repaired: `{payload['repair_sufficiency_checks']}`",
        f"- Individual necessity checks: `{payload['repair_minimality_checks']}`",
        f"- Deficit witness checks: `{payload['witness_checks']}`",
        f"- Pristine single-degradation checks: `{payload['pristine_degradation_checks']}`",
        f"- Multi-deficit packages: `{payload['multi_deficit_packages']}`",
        f"- Deficits hidden by a single-reason trace: `{payload['deficits_hidden_by_single_reason_trace']}`",
        "",
        "| Claim | Packages | Non-authorized | Multi-deficit | Hidden by one reason |",
        "|---|---:|---:|---:|---:|",
    ]
    lines.extend(
        f"| `{row['claim']}` | {row['packages']} | {row['non_authorized']} | "
        f"{row['multi_deficit']} | {row['single_reason_hidden_deficits']} |"
        for row in payload["per_claim"]
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
        "analysis": "profile_v2_counterfactual_explanation_fidelity",
        "protocol": str(protocol_path),
        "protocol_sha256": hashlib.sha256(protocol_path.read_bytes()).hexdigest(),
        **assessment,
        "decision": (
            "counterfactual_explanation_fidelity_confirmed"
            if assessment["all_explanation_properties_hold"]
            else "counterfactual_explanation_violation_detected"
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
