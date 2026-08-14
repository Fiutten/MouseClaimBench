"""Expose post-outcome decision boundaries for the frozen DANDI thresholds.

This analysis changes one author-defined operational threshold at a time while
holding every other frozen condition fixed. It is descriptive sensitivity, not
criterion validation, threshold calibration, or permission to replace either
pre-access protocol after observing its outcome.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import yaml

from mousebrainbench import __version__
from mousebrainbench.artifacts import code_revision

DEFAULT_PROTOCOL = Path("configs/benchmarks/dandi_threshold_sensitivity.yaml")
DEFAULT_OUTPUT = Path("results/dandi_threshold_sensitivity/summary.json")
DEFAULT_MARKDOWN = Path("results/dandi_threshold_sensitivity/summary.md")


def _application(payload: dict[str, Any], resource: str) -> dict[str, Any]:
    for application in payload["applications"]:
        if application["resource"] == resource:
            return application
    raise KeyError(f"DANDI result does not contain {resource}")


def _minimum_grid(
    observed: float,
    thresholds: list[float],
    *,
    inclusive: bool,
) -> list[dict[str, Any]]:
    return [
        {
            "threshold": threshold,
            "operator": ">=" if inclusive else ">",
            "authorized_with_other_frozen_conditions_held": (
                observed >= threshold if inclusive else observed > threshold
            ),
        }
        for threshold in thresholds
    ]


def evaluate(protocol: dict[str, Any]) -> dict[str, Any]:
    """Evaluate one-at-a-time threshold perturbations from frozen results."""

    result = json.loads(Path(protocol["source_result"]).read_text())
    contrast_protocol = yaml.safe_load(
        Path(protocol["source_protocols"]["contrast"]).read_text()
    )
    availability_protocol = yaml.safe_load(
        Path(protocol["source_protocols"]["availability"]).read_text()
    )
    contrast = _application(result, "DANDI:000039")
    availability = _application(result, "DANDI:001176")
    aggregate = contrast["aggregate"]
    relative_mse_improvement = (
        aggregate["baseline_sse"] - aggregate["model_sse"]
    ) / aggregate["baseline_sse"]
    observed = {
        "contrast_minimum_subjects": float(contrast["usable_subjects"]),
        "median_subject_correlation": float(
            aggregate["median_subject_correlation"]
        ),
        "bootstrap_lower_bound": float(aggregate["bootstrap_lower_95"]),
        "positive_subject_fraction": float(
            aggregate["positive_subject_fraction"]
        ),
        "minimum_relative_mse_improvement": float(relative_mse_improvement),
        "availability_minimum_subjects": float(
            availability["selected_subjects"]
        ),
    }
    inclusive = {
        "contrast_minimum_subjects": True,
        "median_subject_correlation": True,
        "bootstrap_lower_bound": False,
        "positive_subject_fraction": True,
        "minimum_relative_mse_improvement": False,
        "availability_minimum_subjects": True,
    }
    grids = {
        name: _minimum_grid(
            observed[name], list(thresholds), inclusive=inclusive[name]
        )
        for name, thresholds in protocol["one_at_a_time_grids"].items()
    }
    frozen_thresholds = {
        "contrast_minimum_subjects": float(
            contrast_protocol["acceptance"]["minimum_subjects"]
        ),
        "median_subject_correlation": float(
            contrast_protocol["acceptance"][
                "median_subject_correlation_minimum"
            ]
        ),
        "bootstrap_lower_bound": float(
            contrast_protocol["acceptance"][
                "subject_bootstrap_lower_95_minimum"
            ]
        ),
        "positive_subject_fraction": float(
            contrast_protocol["acceptance"][
                "minimum_fraction_positive_subjects"
            ]
        ),
        "minimum_relative_mse_improvement": 0.0,
        "availability_minimum_subjects": float(
            availability_protocol["population"]["minimum_subjects"]
        ),
    }
    frozen_grid_membership = {
        name: threshold in protocol["one_at_a_time_grids"][name]
        for name, threshold in frozen_thresholds.items()
    }
    contrast_frozen_pass = all(contrast["conditions"].values())
    availability_frozen_pass = (
        availability["selected_subjects"]
        >= availability_protocol["population"]["minimum_subjects"]
    )
    conditions = {
        "frozen_contrast_decision_reproduced": contrast_frozen_pass,
        "frozen_availability_decision_reproduced": not availability_frozen_pass,
        "every_grid_contains_frozen_threshold": all(
            frozen_grid_membership.values()
        ),
        "no_threshold_is_replaced": True,
    }
    return {
        "source_result_git_revision": result["git_revision"],
        "observed": observed,
        "frozen_thresholds": frozen_thresholds,
        "threshold_operators": {
            name: ">=" if value else ">" for name, value in inclusive.items()
        },
        "operational_rationales": protocol["operational_rationales"],
        "one_at_a_time_results": grids,
        "decision_boundaries": {
            "contrast_maximum_subject_requirement_that_passes": int(
                contrast["usable_subjects"]
            ),
            "contrast_maximum_median_correlation_threshold_that_passes": observed[
                "median_subject_correlation"
            ],
            "contrast_maximum_bootstrap_lower_threshold_that_passes": observed[
                "bootstrap_lower_bound"
            ],
            "contrast_maximum_positive_fraction_threshold_that_passes": observed[
                "positive_subject_fraction"
            ],
            "contrast_maximum_relative_mse_improvement_that_passes": observed[
                "minimum_relative_mse_improvement"
            ],
            "availability_maximum_subject_requirement_that_passes": int(
                availability["selected_subjects"]
            ),
        },
        "conditions": conditions,
        "completed": all(conditions.values()),
        "scope": protocol["scope"],
    }


def _write_markdown(payload: dict[str, Any], path: Path) -> None:
    lines = [
        "# DANDI threshold sensitivity",
        "",
        f"- Decision: `{payload['decision']}`",
        "- Analysis type: post-outcome, one-at-a-time descriptive sensitivity",
        "- Criterion validity claimed: `false`",
        "- Threshold calibration claimed: `false`",
        "",
        "| Criterion | Observed value | Frozen threshold |",
        "|---|---:|---:|",
    ]
    lines.extend(
        f"| `{name}` | {value:.6f} | {payload['frozen_thresholds'][name]:.6f} |"
        for name, value in payload["observed"].items()
    )
    lines.extend(("", payload["scope"]["interpretation"], ""))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines))


def run(
    *,
    protocol_path: Path = DEFAULT_PROTOCOL,
    output: Path = DEFAULT_OUTPUT,
    markdown: Path = DEFAULT_MARKDOWN,
) -> Path:
    """Run and persist the descriptive threshold-sensitivity analysis."""

    protocol = yaml.safe_load(protocol_path.read_text())
    assessment = evaluate(protocol)
    payload = {
        "version": __version__,
        "git_revision": code_revision(),
        "analysis": "dandi_threshold_sensitivity",
        "protocol": str(protocol_path),
        "protocol_sha256": hashlib.sha256(protocol_path.read_bytes()).hexdigest(),
        **assessment,
        "decision": (
            "dandi_threshold_sensitivity_complete"
            if assessment["completed"]
            else "dandi_threshold_sensitivity_incomplete"
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
    print(
        run(
            protocol_path=args.protocol,
            output=args.output,
            markdown=args.markdown,
        ).resolve()
    )


if __name__ == "__main__":
    main()
