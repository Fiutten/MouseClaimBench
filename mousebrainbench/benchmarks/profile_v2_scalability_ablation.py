"""Measure profile-v2 integrity ablations and implementation scalability."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import statistics
import time
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from mousebrainbench import __version__
from mousebrainbench.artifacts import code_revision
from mousebrainbench.benchmarks.profile_v2_contract_mutation import _complete_blocks
from mousebrainbench.benchmarks.profile_v2_provenance_attacks import (
    _base_manifest,
    _digest,
    generate_cases,
)
from mousebrainbench.knowledge import ClaimAuthorizationSystem, load_authorization_profile_v2
from mousebrainbench.knowledge.integrity import (
    ArtifactRecord,
    IntegrityAwareAuthorizationSystem,
    IntegrityDeficitCode,
    validate_evidence_manifest,
)

DEFAULT_PROTOCOL = Path("configs/benchmarks/profile_v2_scalability_ablation.yaml")
DEFAULT_OUTPUT = Path("results/profile_v2_scalability_ablation/summary.json")
DEFAULT_MARKDOWN = Path("results/profile_v2_scalability_ablation/summary.md")


def _timed(repetitions: int, function: Callable[[], int]) -> tuple[list[float], int]:
    durations: list[float] = []
    decisions = 0
    for _ in range(repetitions):
        start = time.perf_counter_ns()
        decisions = function()
        durations.append((time.perf_counter_ns() - start) / 1e9)
    return durations, decisions


def _expanded_manifest(claim: str, artifact_count: int):
    base = _base_manifest(claim)
    if artifact_count < len(base.artifacts):
        raise ValueError("artifact scaling cannot remove claim-supporting artifacts")
    extras = tuple(
        ArtifactRecord(
            artifact_id=f"scale-extra-{index}",
            declared_sha256=_digest(f"scale-extra-{index}"),
            observed_sha256=_digest(f"scale-extra-{index}"),
            cohorts=(f"scale-unit-{index}",),
            study_id=f"scale-study-{index}",
            data_generation_id=f"scale-generation-{index}",
        )
        for index in range(artifact_count - len(base.artifacts))
    )
    return replace(base, package_id=f"scale-{artifact_count}", artifacts=(*base.artifacts, *extras))


def evaluate_ablation(attack_protocol: dict[str, Any]) -> dict[str, Any]:
    """Quantify false authorization when integrity controls are removed."""

    profile = load_authorization_profile_v2()
    cases = generate_cases(attack_protocol)
    all_controls = frozenset(IntegrityDeficitCode)
    control_sets = {
        "profile_only": frozenset(),
        "hash_only": frozenset({IntegrityDeficitCode.ARTIFACT_HASH_MISMATCH}),
        "full_integrity": all_controls,
    }
    control_sets.update(
        {
            f"without_{omitted.value}": all_controls - {omitted}
            for omitted in IntegrityDeficitCode
        }
    )
    counters = {
        name: {"false_authorizations": 0, "false_rejections": 0}
        for name in control_sets
    }
    attacked_cases = 0
    pristine_cases = 0
    for case in cases:
        blocks = _complete_blocks(case.claim)
        core = ClaimAuthorizationSystem(profile, blocks).infer(case.claim)
        observed = {
            row.code
            for row in validate_evidence_manifest(profile, blocks, case.manifest)
        }
        attacked = bool(case.attacks)
        attacked_cases += int(attacked)
        pristine_cases += int(not attacked)
        for name, enabled in control_sets.items():
            authorized = core.authorized and not (observed & enabled)
            counters[name]["false_authorizations"] += int(attacked and authorized)
            counters[name]["false_rejections"] += int(not attacked and not authorized)
    return {
        "cases": len(cases),
        "attacked_cases": attacked_cases,
        "pristine_cases": pristine_cases,
        "systems": counters,
    }


def evaluate_scaling(protocol: dict[str, Any]) -> dict[str, Any]:
    """Measure end-to-end authorization without asserting portable speed."""

    profile = load_authorization_profile_v2()
    claims = tuple(item.claim for item in profile.requirements)
    systems = tuple(
        IntegrityAwareAuthorizationSystem(
            profile, _complete_blocks(claim), _base_manifest(claim)
        )
        for claim in claims
    )
    warmup = int(protocol["batch_scaling"]["warmup_packages"])
    for index in range(warmup):
        if not systems[index % len(systems)].infer(claims[index % len(claims)]).authorized:
            raise RuntimeError("pristine warm-up package was not authorized")
    batch_rows = []
    repetitions = int(protocol["batch_scaling"]["repetitions"])
    for package_count in protocol["batch_scaling"]["package_counts"]:
        count = int(package_count)

        def authorize_batch(package_total: int = count) -> int:
            authorized = 0
            for index in range(package_total):
                position = index % len(systems)
                authorized += int(systems[position].infer(claims[position]).authorized)
            return authorized

        durations, authorized = _timed(repetitions, authorize_batch)
        median_seconds = statistics.median(durations)
        batch_rows.append(
            {
                "packages": count,
                "repetitions": repetitions,
                "authorized_per_repetition": authorized,
                "median_seconds": median_seconds,
                "median_packages_per_second": count / median_seconds,
                "durations_seconds": durations,
            }
        )
    artifact_rows = []
    claim = str(protocol["artifact_scaling"]["claim"])
    blocks = _complete_blocks(claim)
    artifact_repetitions = int(protocol["artifact_scaling"]["repetitions"])
    for artifact_count in protocol["artifact_scaling"]["artifact_counts"]:
        count = int(artifact_count)
        system = IntegrityAwareAuthorizationSystem(
            profile, blocks, _expanded_manifest(claim, count)
        )

        def authorize_package(current_system=system) -> int:
            return int(current_system.infer(claim).authorized)

        durations, authorized = _timed(artifact_repetitions, authorize_package)
        artifact_rows.append(
            {
                "artifacts": count,
                "repetitions": artifact_repetitions,
                "authorized_per_repetition": authorized,
                "median_seconds": statistics.median(durations),
                "durations_seconds": durations,
            }
        )
    x = np.log([row["artifacts"] for row in artifact_rows])
    y = np.log([max(row["median_seconds"], 1e-12) for row in artifact_rows])
    slope = float(np.polyfit(x, y, 1)[0]) if len(artifact_rows) > 1 else float("nan")
    return {
        "batch_scaling": batch_rows,
        "artifact_scaling": artifact_rows,
        "descriptive_log_log_artifact_slope": slope,
        "all_pristine_decisions_authorized": all(
            row["authorized_per_repetition"] == row["packages"] for row in batch_rows
        )
        and all(row["authorized_per_repetition"] == 1 for row in artifact_rows),
        "timing_boundary": "descriptive wall-clock measurements on one host",
    }


def evaluate(protocol: dict[str, Any]) -> dict[str, Any]:
    attack_path = Path(protocol["ablation"]["attack_protocol"])
    ablation = evaluate_ablation(yaml.safe_load(attack_path.read_text()))
    scaling = evaluate_scaling(protocol)
    full = ablation["systems"]["full_integrity"]
    omissions = {
        name: row
        for name, row in ablation["systems"].items()
        if name.startswith("without_")
    }
    endpoints = {
        "all_pristine_scaling_decisions_authorized": scaling[
            "all_pristine_decisions_authorized"
        ],
        "full_gate_false_authorizations_equal_0": (
            full["false_authorizations"] == 0
        ),
        "full_gate_false_rejections_equal_0": full["false_rejections"] == 0,
        "every_leave_one_out_exposes_false_authorization": all(
            row["false_authorizations"] > 0 for row in omissions.values()
        ),
    }
    return {
        "ablation": ablation,
        "scalability": scaling,
        "endpoints": endpoints,
        "all_endpoints_passed": all(endpoints.values()),
        "claim_boundary": protocol["claim_boundary"],
    }


def _write_markdown(payload: dict[str, Any], path: Path) -> None:
    systems = payload["ablation"]["systems"]
    lines = [
        "# Profile v2 scalability and ablation",
        "",
        f"- Decision: `{payload['decision']}`",
        f"- Host: `{payload['runtime']['platform']}`",
        "",
        "| Integrity system | False authorizations | False rejections |",
        "|---|---:|---:|",
    ]
    lines.extend(
        f"| `{name}` | {row['false_authorizations']} | {row['false_rejections']} |"
        for name, row in systems.items()
    )
    lines.extend(("", "| Packages | Median s | Packages/s |", "|---:|---:|---:|"))
    lines.extend(
        f"| {row['packages']} | {row['median_seconds']:.6f} | "
        f"{row['median_packages_per_second']:.1f} |"
        for row in payload["scalability"]["batch_scaling"]
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
        "analysis": "profile_v2_scalability_ablation",
        "protocol": str(protocol_path),
        "protocol_sha256": hashlib.sha256(protocol_path.read_bytes()).hexdigest(),
        "runtime": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "machine": platform.machine(),
            "processor": platform.processor(),
        },
        **assessment,
        "decision": (
            "scalability_ablation_confirmed"
            if assessment["all_endpoints_passed"]
            else "scalability_ablation_failed"
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
