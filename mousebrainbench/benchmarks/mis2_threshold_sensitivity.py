"""Threshold and data-regime sensitivity for MIS 2.0.

The calibration benchmark asks whether designed cases pass or fail at nominal
thresholds. This module goes one step further: it sweeps noise, sample size, and
threshold profiles to map where the non-compensatory MIS gate is safe,
conservative, or dangerous.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from mousebrainbench import __version__
from mousebrainbench.artifacts import code_revision
from mousebrainbench.benchmarks.mis2_synthetic_calibration import SyntheticScenario
from mousebrainbench.benchmarks.synthetic_identifiability import _synthetic_case


DEFAULT_OUTPUT = Path("results/mis2_threshold_sensitivity/summary.json")
DEFAULT_MARKDOWN = Path("results/mis2_threshold_sensitivity/summary.md")


@dataclass(frozen=True)
class ThresholdProfile:
    """Multiplicative threshold perturbation applied per MIS evidence block."""

    name: str
    reproducibility: float
    topology_specificity: float
    directed_identifiability: float


BASE_SCENARIOS = (
    SyntheticScenario(
        name="directed_truth",
        expected_mechanistic=True,
        directed_latency=True,
        region_specific_amplitude=True,
    ),
    SyntheticScenario(
        name="common_drive",
        expected_mechanistic=False,
        directed_latency=False,
        region_specific_amplitude=False,
    ),
    SyntheticScenario(
        name="topology_without_direction",
        expected_mechanistic=False,
        directed_latency=False,
        region_specific_amplitude=True,
    ),
    SyntheticScenario(
        name="direction_without_topology",
        expected_mechanistic=False,
        directed_latency=True,
        region_specific_amplitude=False,
        topology_specific_prediction=False,
    ),
    SyntheticScenario(
        name="prediction_without_true_topology",
        expected_mechanistic=False,
        directed_latency=False,
        region_specific_amplitude=True,
        topology_specific_prediction=False,
    ),
)

THRESHOLD_PROFILES = (
    ThresholdProfile("lenient_all", 0.80, 0.80, 0.80),
    ThresholdProfile("nominal", 1.00, 1.00, 1.00),
    ThresholdProfile("strict_all", 1.20, 1.20, 1.20),
    ThresholdProfile("strict_topology", 1.00, 1.30, 1.00),
    ThresholdProfile("strict_direction", 1.00, 1.00, 1.30),
)

NOISE_LEVELS = (0.08, 0.24, 0.45, 0.60)
SESSION_COUNTS = (6, 12, 24)
DEFAULT_SEEDS = tuple(range(200, 206))


def _criterion_passed(value: float, threshold: float, direction: str) -> bool:
    if direction == "gt":
        return value > threshold
    if direction == "gte":
        return value >= threshold
    if direction == "lt":
        return value < threshold
    if direction == "lte":
        return value <= threshold
    raise ValueError(f"Unknown criterion direction: {direction}")


def _adjusted_mis_passed(case: dict[str, Any], profile: ThresholdProfile) -> tuple[bool, dict[str, bool]]:
    """Re-score a nominal synthetic case under one threshold profile."""

    multipliers = {
        "reproducibility": profile.reproducibility,
        "topology_specificity": profile.topology_specificity,
        "directed_identifiability": profile.directed_identifiability,
    }
    block_status: dict[str, bool] = {}
    for block in case["mis"]["blocks"]:
        block_name = str(block["name"])
        multiplier = multipliers[block_name]
        criterion_status = []
        for criterion in block["criteria"]:
            criterion_status.append(
                _criterion_passed(
                    float(criterion["value"]),
                    float(criterion["threshold"]) * multiplier,
                    str(criterion["direction"]),
                )
            )
        block_status[block_name] = all(criterion_status)
    return all(block_status.values()), block_status


def _decision_class(passed: bool, expected: bool) -> str:
    if passed and expected:
        return "true_positive"
    if passed and not expected:
        return "false_positive"
    if not passed and expected:
        return "false_negative"
    return "true_negative"


def _phase_label(false_positive_rate: float, false_negative_rate: float) -> str:
    if false_positive_rate == 0 and false_negative_rate <= 0.25:
        return "safe"
    if false_positive_rate == 0:
        return "conservative"
    if false_negative_rate <= 0.25:
        return "dangerous"
    return "unstable"


def _raw_cases(
    *,
    noise: float,
    n_sessions: int,
    seeds: tuple[int, ...],
) -> list[dict[str, Any]]:
    rows = []
    for scenario in BASE_SCENARIOS:
        for seed in seeds:
            case = _synthetic_case(
                name=scenario.name,
                directed_latency=scenario.directed_latency,
                region_specific_amplitude=scenario.region_specific_amplitude,
                topology_specific_prediction=scenario.topology_specific_prediction,
                observation_noise=noise,
                split_noise=noise / 2.0,
                prediction_noise=noise / 4.0,
                n_sessions=n_sessions,
                n_null_per_session=80,
                n_permutation_predictions=60,
                seed=seed,
            )
            rows.append(
                {
                    "scenario": scenario.name,
                    "expected_mechanistic": scenario.expected_mechanistic,
                    "seed": seed,
                    "case": case,
                }
            )
    return rows


def _profile_summary(
    *,
    raw_rows: list[dict[str, Any]],
    profile: ThresholdProfile,
    noise: float,
    n_sessions: int,
) -> dict[str, Any]:
    decisions = []
    for row in raw_rows:
        passed, blocks = _adjusted_mis_passed(row["case"], profile)
        decisions.append(
            {
                "scenario": row["scenario"],
                "seed": row["seed"],
                "expected_mechanistic": row["expected_mechanistic"],
                "mis_passed": passed,
                "decision_class": _decision_class(passed, row["expected_mechanistic"]),
                "blocks": blocks,
            }
        )
    total_positive = sum(row["expected_mechanistic"] for row in decisions)
    total_negative = len(decisions) - total_positive
    false_positive = sum(row["decision_class"] == "false_positive" for row in decisions)
    false_negative = sum(row["decision_class"] == "false_negative" for row in decisions)
    false_positive_rate = false_positive / total_negative if total_negative else 0.0
    false_negative_rate = false_negative / total_positive if total_positive else 0.0
    return {
        "profile": profile.name,
        "noise": noise,
        "n_sessions": n_sessions,
        "n_runs": len(decisions),
        "false_positive_count": false_positive,
        "false_negative_count": false_negative,
        "false_positive_rate": false_positive_rate,
        "false_negative_rate": false_negative_rate,
        "phase": _phase_label(false_positive_rate, false_negative_rate),
        "decisions": decisions,
    }


def run_sensitivity(
    *,
    output: Path = DEFAULT_OUTPUT,
    markdown: Path = DEFAULT_MARKDOWN,
    seeds: tuple[int, ...] = DEFAULT_SEEDS,
) -> Path:
    """Run the full MIS 2.0 threshold sensitivity sweep."""

    rows = []
    for noise in NOISE_LEVELS:
        for n_sessions in SESSION_COUNTS:
            raw_rows = _raw_cases(noise=noise, n_sessions=n_sessions, seeds=seeds)
            for profile in THRESHOLD_PROFILES:
                rows.append(
                    _profile_summary(
                        raw_rows=raw_rows,
                        profile=profile,
                        noise=noise,
                        n_sessions=n_sessions,
                    )
                )

    phase_counts = {phase: sum(row["phase"] == phase for row in rows) for phase in _PHASE_ORDER}
    nominal_rows = [row for row in rows if row["profile"] == "nominal"]
    payload = {
        "version": __version__,
        "git_revision": code_revision(),
        "analysis": "mis2_threshold_sensitivity",
        "scope": (
            "Synthetic threshold, noise, and sample-size sensitivity for MIS 2.0. "
            "This evaluates decision stability under known truth and does not "
            "constitute biological evidence."
        ),
        "seeds": list(seeds),
        "noise_levels": list(NOISE_LEVELS),
        "session_counts": list(SESSION_COUNTS),
        "threshold_profiles": [profile.__dict__ for profile in THRESHOLD_PROFILES],
        "phase_counts": phase_counts,
        "nominal_phase_counts": {
            phase: sum(row["phase"] == phase for row in nominal_rows) for phase in _PHASE_ORDER
        },
        "rows": rows,
        "decision": (
            "mis2_sensitivity_supports_conservative_gate"
            if phase_counts["dangerous"] == 0 and phase_counts["unstable"] == 0
            else "mis2_sensitivity_detects_unsafe_threshold_region"
        ),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2))
    write_markdown(payload, markdown)
    return output


_PHASE_ORDER = ("safe", "conservative", "dangerous", "unstable")


def write_markdown(payload: dict[str, Any], markdown: Path) -> None:
    """Write the sensitivity summary as a compact report."""

    lines = [
        "# MIS 2.0 Threshold Sensitivity",
        "",
        f"- Decision: `{payload['decision']}`",
        f"- Operating cells: `{len(payload['rows'])}`",
        f"- Seeds per scenario: `{len(payload['seeds'])}`",
        "",
        "## Phase Counts",
        "",
        "| Phase | Count | Interpretation |",
        "|---|---:|---|",
    ]
    interpretations = {
        "safe": "FPR is zero and FNR remains low.",
        "conservative": "FPR is zero, but FNR is high under weak data regimes.",
        "dangerous": "FPR is non-zero while sensitivity appears acceptable.",
        "unstable": "FPR and FNR are both problematic.",
    }
    for phase in _PHASE_ORDER:
        lines.append(f"| `{phase}` | `{payload['phase_counts'][phase]}` | {interpretations[phase]} |")

    lines.extend(
        [
            "",
            "## Nominal Profile",
            "",
            "| Noise | Sessions | FPR | FNR | Phase |",
            "|---:|---:|---:|---:|---|",
        ]
    )
    for row in payload["rows"]:
        if row["profile"] != "nominal":
            continue
        lines.append(
            f"| `{row['noise']:.2f}` | `{row['n_sessions']}` | "
            f"`{row['false_positive_rate']:.3f}` | `{row['false_negative_rate']:.3f}` | "
            f"`{row['phase']}` |"
        )

    lines.extend(
        [
            "",
            "## Worst Conservative Cells",
            "",
            "| Profile | Noise | Sessions | FPR | FNR | Phase |",
            "|---|---:|---:|---:|---:|---|",
        ]
    )
    conservative = sorted(
        [row for row in payload["rows"] if row["phase"] == "conservative"],
        key=lambda row: (-row["false_negative_rate"], row["profile"], row["noise"], row["n_sessions"]),
    )
    for row in conservative[:10]:
        lines.append(
            f"| `{row['profile']}` | `{row['noise']:.2f}` | `{row['n_sessions']}` | "
            f"`{row['false_positive_rate']:.3f}` | `{row['false_negative_rate']:.3f}` | "
            f"`{row['phase']}` |"
        )

    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "The useful operating region is defined by zero false positives in the designed "
            "non-mechanistic cases. Conservative cells are not failures of the claim gate. "
            "They show where low SNR, few sessions, or strict thresholds prevent a true "
            "mechanistic signal from passing. Dangerous or unstable cells would require "
            "threshold redesign before using MIS 2.0 as a stronger methodological claim.",
            "",
        ]
    )
    markdown.parent.mkdir(parents=True, exist_ok=True)
    markdown.write_text("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--markdown", type=Path, default=DEFAULT_MARKDOWN)
    parser.add_argument("--seeds", type=int, default=len(DEFAULT_SEEDS))
    args = parser.parse_args()
    seeds = tuple(range(200, 200 + args.seeds))
    path = run_sensitivity(output=args.output, markdown=args.markdown, seeds=seeds)
    print(json.dumps({"output": str(path.resolve())}))


if __name__ == "__main__":
    main()
