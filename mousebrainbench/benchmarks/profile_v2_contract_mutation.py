"""Run the frozen contract-mutation benchmark for the hardened v2 profile."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from dataclasses import dataclass
from itertools import combinations, product
from pathlib import Path
from typing import Any

import yaml

from mousebrainbench import __version__
from mousebrainbench.artifacts import code_revision
from mousebrainbench.knowledge import (
    ClaimAuthorizationSystem,
    authorize_with_clingo_v2,
    load_authorization_profile_v2,
)
from mousebrainbench.validation.evidence_contract import EvidenceBlock, EvidenceStatus

DEFAULT_PROTOCOL = Path("configs/benchmarks/profile_v2_contract_mutation.yaml")
DEFAULT_OUTPUT = Path("results/profile_v2_contract_mutation/summary.json")
DEFAULT_MARKDOWN = Path("results/profile_v2_contract_mutation/summary.md")
DEFAULT_CASES = Path("results/profile_v2_contract_mutation/cases.csv")

PRIORITY = (
    EvidenceStatus.FAILED,
    EvidenceStatus.REQUIRES_REVIEW,
    EvidenceStatus.UNKNOWN,
    EvidenceStatus.NOT_APPLICABLE,
)


@dataclass(frozen=True)
class MutationCase:
    """One controlled evidence package and its specification-defined outcome."""

    case_id: str
    family: str
    claim: str
    blocks: dict[str, EvidenceBlock]
    expected_authorized: bool
    expected_deficits: tuple[tuple[str, EvidenceStatus], ...]


def _complete_observations(block_name: str) -> dict[str, Any]:
    specification = load_authorization_profile_v2().block_specification(block_name)
    return {
        field: 0 if field in {"overlap", "synchronization_error"} else f"declared-{field}"
        for field in specification.required_observations_when_passed
    }


def _block(
    name: str,
    status: EvidenceStatus = EvidenceStatus.PASSED,
    *,
    observations: dict[str, Any] | None = None,
) -> EvidenceBlock:
    return EvidenceBlock.from_mapping(
        name=name,
        status=status,
        source="frozen-contract-mutation-case",
        rule="controlled profile-v2 predicate",
        rationale="the mutation truth is fixed by the executable contract",
        observations=(
            _complete_observations(name) if observations is None else observations
        ),
    )


def _complete_blocks(claim: str) -> dict[str, EvidenceBlock]:
    requirement = load_authorization_profile_v2().requirement(claim)
    if requirement is None:
        raise ValueError(f"unknown profile-v2 claim: {claim}")
    return {name: _block(name) for name in requirement.required_blocks}


def generate_cases() -> tuple[MutationCase, ...]:
    """Enumerate the mutation space fixed by the protocol."""

    profile = load_authorization_profile_v2()
    defects = tuple(status for status in EvidenceStatus if status is not EvidenceStatus.PASSED)
    cases: list[MutationCase] = []
    for requirement in profile.requirements:
        claim = requirement.claim
        cases.append(
            MutationCase(
                case_id=f"{claim}__pristine",
                family="pristine_complete",
                claim=claim,
                blocks=_complete_blocks(claim),
                expected_authorized=True,
                expected_deficits=(),
            )
        )
        for block_name in requirement.required_blocks:
            for status in defects:
                blocks = _complete_blocks(claim)
                blocks[block_name] = _block(block_name, status)
                cases.append(
                    MutationCase(
                        case_id=f"{claim}__status__{block_name}__{status.value}",
                        family="single_status_defect",
                        claim=claim,
                        blocks=blocks,
                        expected_authorized=False,
                        expected_deficits=((block_name, status),),
                    )
                )
            blocks = _complete_blocks(claim)
            del blocks[block_name]
            cases.append(
                MutationCase(
                    case_id=f"{claim}__omitted__{block_name}",
                    family="omitted_required_block",
                    claim=claim,
                    blocks=blocks,
                    expected_authorized=False,
                    expected_deficits=((block_name, EvidenceStatus.UNKNOWN),),
                )
            )
            specification = profile.block_specification(block_name)
            for field in specification.required_observations_when_passed:
                blocks = _complete_blocks(claim)
                observations = dict(blocks[block_name].observations)
                del observations[field]
                blocks[block_name] = _block(block_name, observations=observations)
                cases.append(
                    MutationCase(
                        case_id=f"{claim}__metadata__{block_name}__{field}",
                        family="missing_required_observation",
                        claim=claim,
                        blocks=blocks,
                        expected_authorized=False,
                        expected_deficits=((block_name, EvidenceStatus.REQUIRES_REVIEW),),
                    )
                )
        for left, right in combinations(requirement.required_blocks, 2):
            for left_status, right_status in product(defects, repeat=2):
                blocks = _complete_blocks(claim)
                blocks[left] = _block(left, left_status)
                blocks[right] = _block(right, right_status)
                cases.append(
                    MutationCase(
                        case_id=(
                            f"{claim}__mixed__{left}__{left_status.value}__"
                            f"{right}__{right_status.value}"
                        ),
                        family="pairwise_mixed_status_defects",
                        claim=claim,
                        blocks=blocks,
                        expected_authorized=False,
                        expected_deficits=tuple(
                            sorted(
                                ((left, left_status), (right, right_status)),
                                key=lambda item: item[0],
                            )
                        ),
                    )
                )
    identifiers = [case.case_id for case in cases]
    if len(identifiers) != len(set(identifiers)):
        raise RuntimeError("contract-mutation case identifiers are not unique")
    return tuple(cases)


def _raw_statuses(case: MutationCase) -> dict[str, EvidenceStatus]:
    requirement = load_authorization_profile_v2().requirement(case.claim)
    if requirement is None:
        raise RuntimeError("mutation case uses an undeclared claim")
    return {
        name: (
            case.blocks[name].status
            if name in case.blocks
            else EvidenceStatus.UNKNOWN
        )
        for name in requirement.required_blocks
    }


def _comparator_decisions(case: MutationCase) -> dict[str, bool]:
    statuses = _raw_statuses(case)
    passed = sum(status is EvidenceStatus.PASSED for status in statuses.values())
    all_passed = passed == len(statuses)
    prediction_shortcut = (
        statuses.get("prediction") is EvidenceStatus.PASSED
        if "prediction" in statuses
        else all_passed
    )
    return {
        "raw_status_all_passed_ignoring_schema": all_passed,
        "equal_weight_75_percent_ignoring_schema": passed / len(statuses) >= 0.75,
        "prediction_shortcut_ignoring_schema": prediction_shortcut,
    }


def _priority_trace(case: MutationCase) -> tuple[tuple[str, EvidenceStatus], ...]:
    statuses = _raw_statuses(case)
    for target in PRIORITY:
        witnesses = tuple(
            sorted((name, status) for name, status in statuses.items() if status is target)
        )
        if witnesses:
            return witnesses
    return ()


def _deficit_set(decision) -> tuple[tuple[str, EvidenceStatus], ...]:
    return tuple(
        sorted(
            ((fact.name, fact.effective_status) for fact in decision.deficits),
            key=lambda item: item[0],
        )
    )


def _asp_selection(
    cases: tuple[MutationCase, ...], maximum: int | None = None
) -> set[str]:
    """Select all ASP cases or a deterministic per-family diagnostic subset."""

    if maximum is None:
        return {case.case_id for case in cases}
    selected: set[str] = set()
    families = sorted({case.family for case in cases})
    for family in families:
        identifiers = sorted(case.case_id for case in cases if case.family == family)
        selected.update(identifiers[:maximum])
    return selected


def evaluate(
    cases: tuple[MutationCase, ...],
    protocol: dict[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Evaluate v2, transparent shortcuts, deficit recall, and ASP conformance."""

    profile = load_authorization_profile_v2()
    asp_protocol = protocol["asp_conformance"]
    selection = asp_protocol["selection"]
    maximum_asp = (
        None
        if selection == "all_generated_cases"
        else int(asp_protocol["maximum_cases_per_family"])
    )
    asp_selection = _asp_selection(cases, maximum_asp)
    rows: list[dict[str, Any]] = []
    comparator_false_authorizations: Counter[str] = Counter()
    v2_false_authorizations = 0
    v2_false_rejections = 0
    v2_exact_deficits = 0
    priority_exact_deficits = 0
    asp_exact = 0
    for case in cases:
        decision = ClaimAuthorizationSystem(profile, case.blocks).infer(case.claim)
        predicted = decision.authorized
        actual_deficits = _deficit_set(decision)
        exact_deficits = actual_deficits == case.expected_deficits
        v2_false_authorizations += int(predicted and not case.expected_authorized)
        v2_false_rejections += int(not predicted and case.expected_authorized)
        v2_exact_deficits += int(exact_deficits)
        priority_exact = _priority_trace(case) == case.expected_deficits
        priority_exact_deficits += int(priority_exact)
        comparators = _comparator_decisions(case)
        for name, authorized in comparators.items():
            comparator_false_authorizations[name] += int(
                authorized and not case.expected_authorized
            )
        asp_checked = case.case_id in asp_selection
        asp_match = None
        if asp_checked:
            asp = authorize_with_clingo_v2(profile, case.claim, case.blocks)
            asp_match = (
                asp.status is decision.status
                and asp.deficits == actual_deficits
            )
            asp_exact += int(asp_match)
        rows.append(
            {
                "case_id": case.case_id,
                "family": case.family,
                "claim": case.claim,
                "expected_authorized": case.expected_authorized,
                "v2_authorized": predicted,
                "expected_deficits": len(case.expected_deficits),
                "v2_exact_deficits": exact_deficits,
                "priority_exact_deficits": priority_exact,
                "asp_checked": asp_checked,
                "asp_exact": asp_match,
                **comparators,
            }
        )
    mutation_count = sum(not case.expected_authorized for case in cases)
    pristine_count = len(cases) - mutation_count
    family_counts = Counter(case.family for case in cases)
    asp_cases = len(asp_selection)
    endpoints = {
        "zero_false_authorizations_by_profile_v2": v2_false_authorizations == 0,
        "all_pristine_cases_authorized": v2_false_rejections == 0,
        "exact_deficit_set_recall_equal_1": v2_exact_deficits == len(cases),
        "python_asp_status_and_deficit_equivalence_equal_1": asp_exact == asp_cases,
        "at_least_one_transparent_comparator_has_false_authorizations": any(
            value > 0 for value in comparator_false_authorizations.values()
        ),
    }
    summary = {
        "profile_id": profile.profile_id,
        "profile_version": profile.version,
        "profile_hash": profile.source_hash,
        "cases": len(cases),
        "pristine_cases": pristine_count,
        "mutation_cases": mutation_count,
        "family_counts": dict(sorted(family_counts.items())),
        "profile_v2": {
            "false_authorizations": v2_false_authorizations,
            "false_rejections": v2_false_rejections,
            "exact_deficit_sets": v2_exact_deficits,
            "exact_deficit_rate": v2_exact_deficits / len(cases),
        },
        "prioritized_single_reason_trace": {
            "exact_deficit_sets": priority_exact_deficits,
            "exact_deficit_rate": priority_exact_deficits / len(cases),
        },
        "comparator_false_authorizations": dict(
            sorted(comparator_false_authorizations.items())
        ),
        "asp_conformance": {
            "cases": asp_cases,
            "exact_status_and_deficit_matches": asp_exact,
            "rate": asp_exact / asp_cases,
        },
        "primary_endpoints": endpoints,
        "all_primary_endpoints_passed": all(endpoints.values()),
        "interpretation": protocol["interpretation"],
    }
    return summary, rows


def _write_cases(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(rows[0]),
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


def _write_markdown(payload: dict[str, Any], path: Path) -> None:
    priority_rate = payload["prioritized_single_reason_trace"]["exact_deficit_rate"]
    lines = [
        "# Profile v2 contract-mutation benchmark",
        "",
        f"- Decision: `{payload['decision']}`",
        f"- Cases: `{payload['cases']}`",
        f"- Mutation cases: `{payload['mutation_cases']}`",
        f"- v2 false authorizations: `{payload['profile_v2']['false_authorizations']}`",
        f"- v2 exact deficit rate: `{payload['profile_v2']['exact_deficit_rate']:.4f}`",
        f"- Prioritized-trace exact deficit rate: `{priority_rate:.4f}`",
        f"- Python-ASP equivalence: `{payload['asp_conformance']['rate']:.4f}`",
        "",
        "## Comparator false authorizations",
        "",
    ]
    lines.extend(
        f"- `{name}`: `{value}`"
        for name, value in payload["comparator_false_authorizations"].items()
    )
    lines.extend(("", payload["interpretation"], ""))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines))


def run(
    *,
    protocol_path: Path = DEFAULT_PROTOCOL,
    output: Path = DEFAULT_OUTPUT,
    markdown: Path = DEFAULT_MARKDOWN,
    cases_output: Path = DEFAULT_CASES,
) -> Path:
    """Run and persist the frozen deterministic mutation benchmark."""

    protocol = yaml.safe_load(protocol_path.read_text())
    cases = generate_cases()
    summary, rows = evaluate(cases, protocol)
    payload = {
        "version": __version__,
        "git_revision": code_revision(),
        "analysis": "profile_v2_contract_mutation",
        "protocol": str(protocol_path),
        "protocol_sha256": hashlib.sha256(protocol_path.read_bytes()).hexdigest(),
        **summary,
        "decision": (
            "profile_v2_contract_mutation_passed"
            if summary["all_primary_endpoints_passed"]
            else "profile_v2_contract_mutation_failed"
        ),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n")
    _write_cases(rows, cases_output)
    _write_markdown(payload, markdown)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--markdown", type=Path, default=DEFAULT_MARKDOWN)
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES)
    args = parser.parse_args()
    print(
        run(
            protocol_path=args.protocol,
            output=args.output,
            markdown=args.markdown,
            cases_output=args.cases,
        ).resolve()
    )


if __name__ == "__main__":
    main()
