"""Threshold sensitivity analysis for the ClaimBench v2 gate."""

from __future__ import annotations

import argparse
import json
from itertools import product
from pathlib import Path
from typing import Any

from mousebrainbench import __version__
from mousebrainbench.artifacts import code_revision
from mousebrainbench.benchmarks.claim_adversarial_v2 import build_cases
from mousebrainbench.validation.claim_evaluation import (
    ClaimGateEvaluator,
    ClaimGateThresholds,
    aggregate_claim_confusion,
    claim_confusion_matrix,
)


DEFAULT_OUTPUT = Path("results/claim_threshold_sensitivity_v2/summary.json")
DEFAULT_MARKDOWN = Path("results/claim_threshold_sensitivity_v2/summary.md")


def _grid() -> tuple[ClaimGateThresholds, ...]:
    predictive = (0.25, 0.30, 0.35)
    reproducible = (0.65, 0.70, 0.75)
    topology = (0.04, 0.05, 0.06)
    directed = (0.45, 0.50, 0.55)
    structure_function = (0.008, 0.010, 0.012)
    return tuple(
        ClaimGateThresholds(
            predictive_score=p,
            reproducibility_score=r,
            topology_effect=t,
            directed_fraction=d,
            matched_structure_function_effect=sf,
        )
        for p, r, t, d, sf in product(
            predictive,
            reproducible,
            topology,
            directed,
            structure_function,
        )
    )


def _row_for_thresholds(thresholds: ClaimGateThresholds) -> dict[str, Any]:
    cases = build_cases()
    evaluator = ClaimGateEvaluator(
        thresholds=thresholds,
        name=(
            "claim_gate"
            f"_p{thresholds.predictive_score:.2f}"
            f"_r{thresholds.reproducibility_score:.2f}"
            f"_t{thresholds.topology_effect:.3f}"
            f"_d{thresholds.directed_fraction:.2f}"
            f"_sf{thresholds.matched_structure_function_effect:.3f}"
        ),
    )
    truth = {case.name: set(case.true_claims) for case in cases}
    decisions = {
        evaluator.name: {
            case.name: set(evaluator.evaluate(case.evidence).allowed_claims) for case in cases
        }
    }
    aggregate = aggregate_claim_confusion(
        claim_confusion_matrix(truth_by_case=truth, decisions_by_evaluator=decisions)
    )[0]
    return {
        "thresholds": {
            "predictive_score": thresholds.predictive_score,
            "reproducibility_score": thresholds.reproducibility_score,
            "topology_effect": thresholds.topology_effect,
            "directed_fraction": thresholds.directed_fraction,
            "matched_structure_function_effect": thresholds.matched_structure_function_effect,
        },
        **aggregate,
    }


def run(output: Path = DEFAULT_OUTPUT, markdown: Path = DEFAULT_MARKDOWN) -> Path:
    """Run sensitivity grid over the v2 adversarial suite."""

    rows = [_row_for_thresholds(thresholds) for thresholds in _grid()]
    safe_rows = [
        row
        for row in rows
        if int(row["fp"]) == 0 and float(row["conservativeness_index"]) <= 0.25
    ]
    dangerous_rows = [row for row in rows if int(row["fp"]) > 0]
    payload = {
        "version": __version__,
        "git_revision": code_revision(),
        "analysis": "claim_threshold_sensitivity_v2",
        "num_threshold_cells": len(rows),
        "safe_cells": len(safe_rows),
        "dangerous_cells": len(dangerous_rows),
        "max_safe_conservativeness_index": max(
            (float(row["conservativeness_index"]) for row in safe_rows),
            default=None,
        ),
        "min_dangerous_overclaiming_risk_index": min(
            (float(row["overclaiming_risk_index"]) for row in dangerous_rows),
            default=None,
        ),
        "rows": rows,
        "decision": (
            "claim_thresholds_have_nontrivial_safe_region_with_reportable_limits"
            if len(safe_rows) >= 20
            else "claim_thresholds_require_careful_reporting"
        ),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2))
    write_markdown(payload, markdown)
    return output


def write_markdown(payload: dict[str, Any], markdown: Path) -> None:
    """Write threshold sensitivity report."""

    lines = [
        "# Claim Threshold Sensitivity v2",
        "",
        f"- Decision: `{payload['decision']}`",
        f"- Threshold cells: `{payload['num_threshold_cells']}`",
        f"- Safe cells: `{payload['safe_cells']}`",
        f"- Dangerous cells: `{payload['dangerous_cells']}`",
        "",
        "Safe means FP=0 and CI<=0.25 over the ClaimBench v2 adversarial suite.",
        "",
    ]
    markdown.parent.mkdir(parents=True, exist_ok=True)
    markdown.write_text("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--markdown", type=Path, default=DEFAULT_MARKDOWN)
    args = parser.parse_args()
    print(json.dumps({"output": str(run(args.output, args.markdown).resolve())}))


if __name__ == "__main__":
    main()
