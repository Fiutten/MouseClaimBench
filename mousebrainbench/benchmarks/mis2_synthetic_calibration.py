"""Expanded synthetic calibration benchmark for MIS 2.0 development.

This benchmark does not replace the submitted manuscript artifacts. It extends
the truth-known synthetic checks so we can estimate how the current
non-compensatory gate behaves across repeated seeds, noise levels, and failure
modes before claiming a stronger MIS 2.0 methodology.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from mousebrainbench import __version__
from mousebrainbench.artifacts import code_revision
from mousebrainbench.benchmarks.synthetic_identifiability import _synthetic_case


DEFAULT_OUTPUT = Path("results/mis2_synthetic_calibration/summary.json")
DEFAULT_MARKDOWN = Path("results/mis2_synthetic_calibration/summary.md")


@dataclass(frozen=True)
class SyntheticScenario:
    """Truth-known stress case for evaluating MIS decisions."""

    name: str
    expected_mechanistic: bool
    directed_latency: bool
    region_specific_amplitude: bool
    topology_specific_prediction: bool = True
    observation_noise: float = 0.08
    split_noise: float = 0.04
    prediction_noise: float = 0.02
    n_sessions: int = 24


SCENARIOS = (
    SyntheticScenario(
        name="clean_directed_truth",
        expected_mechanistic=True,
        directed_latency=True,
        region_specific_amplitude=True,
    ),
    SyntheticScenario(
        name="noisy_directed_truth",
        expected_mechanistic=True,
        directed_latency=True,
        region_specific_amplitude=True,
        observation_noise=0.16,
        split_noise=0.08,
        prediction_noise=0.04,
    ),
    SyntheticScenario(
        name="low_sample_directed_truth",
        expected_mechanistic=True,
        directed_latency=True,
        region_specific_amplitude=True,
        n_sessions=8,
    ),
    SyntheticScenario(
        name="low_snr_directed_truth",
        expected_mechanistic=True,
        directed_latency=True,
        region_specific_amplitude=True,
        observation_noise=0.60,
        split_noise=0.30,
        prediction_noise=0.15,
        n_sessions=8,
    ),
    SyntheticScenario(
        name="common_drive_high_reproducibility",
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
    SyntheticScenario(
        name="noisy_common_drive",
        expected_mechanistic=False,
        directed_latency=False,
        region_specific_amplitude=False,
        observation_noise=0.16,
        split_noise=0.08,
    ),
)


def _block_status(case: dict[str, Any]) -> dict[str, bool]:
    return {str(block["name"]): bool(block["passed"]) for block in case["mis"]["blocks"]}


def _scenario_result(scenario: SyntheticScenario, seed: int) -> dict[str, Any]:
    case = _synthetic_case(
        name=scenario.name,
        directed_latency=scenario.directed_latency,
        region_specific_amplitude=scenario.region_specific_amplitude,
        topology_specific_prediction=scenario.topology_specific_prediction,
        observation_noise=scenario.observation_noise,
        split_noise=scenario.split_noise,
        prediction_noise=scenario.prediction_noise,
        n_sessions=scenario.n_sessions,
        seed=seed,
    )
    passed = bool(case["mis"]["passed"])
    expected = scenario.expected_mechanistic
    if passed and expected:
        decision_class = "true_positive"
    elif passed and not expected:
        decision_class = "false_positive"
    elif not passed and expected:
        decision_class = "false_negative"
    else:
        decision_class = "true_negative"
    return {
        "scenario": scenario.name,
        "seed": seed,
        "expected_mechanistic": expected,
        "mis_passed": passed,
        "decision_class": decision_class,
        "blocks": _block_status(case),
        "score": case["mis"]["score"],
        "summary": case["summary"],
    }


def _summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    total_positive = sum(row["expected_mechanistic"] for row in rows)
    total_negative = len(rows) - total_positive
    false_positive = sum(row["decision_class"] == "false_positive" for row in rows)
    false_negative = sum(row["decision_class"] == "false_negative" for row in rows)
    by_scenario = []
    for scenario in SCENARIOS:
        subset = [row for row in rows if row["scenario"] == scenario.name]
        passed = sum(row["mis_passed"] for row in subset)
        block_pass_rates = {
            block: sum(row["blocks"][block] for row in subset) / len(subset)
            for block in ("reproducibility", "topology_specificity", "directed_identifiability")
        }
        by_scenario.append(
            {
                "scenario": scenario.name,
                "expected_mechanistic": scenario.expected_mechanistic,
                "n_runs": len(subset),
                "mis_pass_rate": passed / len(subset),
                "decision_classes": {
                    name: sum(row["decision_class"] == name for row in subset)
                    for name in ("true_positive", "false_positive", "true_negative", "false_negative")
                },
                "block_pass_rates": block_pass_rates,
            }
        )
    return {
        "n_runs": len(rows),
        "n_positive_truth_runs": total_positive,
        "n_negative_truth_runs": total_negative,
        "false_positive_count": false_positive,
        "false_negative_count": false_negative,
        "false_positive_rate": false_positive / total_negative if total_negative else None,
        "false_negative_rate": false_negative / total_positive if total_positive else None,
        "by_scenario": by_scenario,
    }


def run_calibration(
    *,
    output: Path = DEFAULT_OUTPUT,
    markdown: Path = DEFAULT_MARKDOWN,
    seeds: tuple[int, ...] = tuple(range(100, 112)),
) -> Path:
    """Run the MIS 2.0 synthetic calibration suite."""

    rows = [_scenario_result(scenario, seed) for scenario in SCENARIOS for seed in seeds]
    summary = _summarize(rows)
    payload = {
        "version": __version__,
        "git_revision": code_revision(),
        "analysis": "mis2_synthetic_calibration",
        "scope": (
            "Truth-known calibration for MIS 2.0 development. This is a synthetic "
            "decision-stability audit, not empirical evidence about mouse brain data."
        ),
        "seeds": list(seeds),
        "summary": summary,
        "rows": rows,
        "decision": (
            "mis2_nominal_synthetic_suite_passed"
            if summary["false_positive_count"] == 0
            else "mis2_requires_gate_revision"
        ),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2))
    write_markdown(payload, markdown)
    return output


def write_markdown(payload: dict[str, Any], markdown: Path) -> None:
    """Write a compact human-readable calibration report."""

    summary = payload["summary"]
    lines = [
        "# MIS 2.0 Synthetic Calibration",
        "",
        f"- Decision: `{payload['decision']}`",
        f"- Runs: `{summary['n_runs']}`",
        f"- False-positive rate: `{summary['false_positive_rate']:.4f}`",
        f"- False-negative rate: `{summary['false_negative_rate']:.4f}`",
        "",
        "## Scenario Summary",
        "",
        "| Scenario | Truth | MIS pass rate | FP | FN | Repro | Topology | Direction |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary["by_scenario"]:
        classes = row["decision_classes"]
        blocks = row["block_pass_rates"]
        lines.append(
            f"| `{row['scenario']}` | `{row['expected_mechanistic']}` | "
            f"`{row['mis_pass_rate']:.3f}` | `{classes['false_positive']}` | "
            f"`{classes['false_negative']}` | `{blocks['reproducibility']:.3f}` | "
            f"`{blocks['topology_specificity']:.3f}` | "
            f"`{blocks['directed_identifiability']:.3f}` |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "This calibration is intentionally synthetic. A zero false-positive rate in these "
            "cases means the current non-compensatory MIS gate rejects the designed "
            "non-mechanistic failure modes. False negatives identify conservative low-SNR "
            "or low-sample regions of the gate and should be interpreted as sensitivity, "
            "not biological failure.",
            "",
        ]
    )
    markdown.parent.mkdir(parents=True, exist_ok=True)
    markdown.write_text("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--markdown", type=Path, default=DEFAULT_MARKDOWN)
    parser.add_argument("--seeds", type=int, default=12)
    args = parser.parse_args()
    seeds = tuple(range(100, 100 + args.seeds))
    path = run_calibration(output=args.output, markdown=args.markdown, seeds=seeds)
    print(json.dumps({"output": str(path.resolve())}))


if __name__ == "__main__":
    main()
