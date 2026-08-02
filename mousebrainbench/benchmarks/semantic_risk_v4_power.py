"""Prospective exact operating characteristics for the v4 external contract."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from mousebrainbench import __version__
from mousebrainbench.artifacts import code_revision
from mousebrainbench.validation.nondegenerate_risk_control import (
    one_sided_binomial_bound,
)

DEFAULT_PROTOCOL = Path("configs/benchmarks/semantic_risk_control_v4.yaml")
DEFAULT_OUTPUT = Path("results/semantic_risk_v4_power/summary.json")
DEFAULT_MARKDOWN = Path("results/semantic_risk_v4_power/summary.md")


def minimum_units_for_zero_failures(target: float, confidence: float) -> int:
    """Return the first sample size whose zero-failure upper bound meets target."""

    for units in range(1, 1_000_001):
        if one_sided_binomial_bound(0, units, confidence=confidence, side="upper") <= target:
            return units
    raise RuntimeError("required sample size exceeds one million units")


def maximum_compatible_failures(units: int, target: float, confidence: float) -> int:
    compatible = [
        failures
        for failures in range(units + 1)
        if one_sided_binomial_bound(
            failures, units, confidence=confidence, side="upper"
        )
        <= target
    ]
    return max(compatible, default=-1)


def minimum_successes_for_lower_bound(
    units: int, minimum: float, confidence: float
) -> int | None:
    return next(
        (
            successes
            for successes in range(units + 1)
            if one_sided_binomial_bound(
                successes, units, confidence=confidence, side="lower"
            )
            >= minimum
        ),
        None,
    )


def run(
    *,
    protocol: Path = DEFAULT_PROTOCOL,
    output: Path = DEFAULT_OUTPUT,
    markdown: Path = DEFAULT_MARKDOWN,
) -> Path:
    payload = yaml.safe_load(protocol.read_text())
    contract = payload["risk_and_activation_contract"]
    target = float(contract["target_experiment_failure_probability"])
    confidence = float(contract["confidence_level"])
    minimum_coverage = float(contract["minimum_authorized_experiment_coverage"])
    minimum_recovery = float(contract["minimum_positive_recovery"])
    sample_sizes = (10, 20, 29, 40, 60, 100, 200, 500)
    rows = []
    for units in sample_sizes:
        rows.append(
            {
                "units": units,
                "maximum_compatible_failures": maximum_compatible_failures(
                    units, target, confidence
                ),
                "minimum_authorized_units": minimum_successes_for_lower_bound(
                    units, minimum_coverage, confidence
                ),
                "minimum_recovered_positive_units": minimum_successes_for_lower_bound(
                    units, minimum_recovery, confidence
                ),
            }
        )
    result = {
        "version": __version__,
        "git_revision": code_revision(),
        "analysis": "semantic_risk_v4_prospective_power",
        "protocol": str(protocol),
        "target_risk": target,
        "confidence": confidence,
        "minimum_coverage": minimum_coverage,
        "minimum_positive_recovery": minimum_recovery,
        "minimum_units_for_zero_failures": minimum_units_for_zero_failures(
            target, confidence
        ),
        "operating_characteristics": rows,
        "interpretation": (
            "These are exact design constraints, not observed performance and not a "
            "post-hoc power calculation."
        ),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    lines = [
        "# Semantic risk v4 prospective operating characteristics",
        "",
        f"- Minimum zero-failure units: `{result['minimum_units_for_zero_failures']}`",
        "- Analysis role: `prospective design constraint`",
        "",
        "| Units | Max failures | Min authorized | Min recovered positives |",
        "|---:|---:|---:|---:|",
    ]
    lines.extend(
        f"| {row['units']} | {row['maximum_compatible_failures']} | "
        f"{row['minimum_authorized_units']} | {row['minimum_recovered_positive_units']} |"
        for row in rows
    )
    lines.append("")
    markdown.parent.mkdir(parents=True, exist_ok=True)
    markdown.write_text("\n".join(lines))
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--markdown", type=Path, default=DEFAULT_MARKDOWN)
    args = parser.parse_args()
    print(json.dumps({"output": str(run(protocol=args.protocol, output=args.output, markdown=args.markdown).resolve())}))


if __name__ == "__main__":
    main()
