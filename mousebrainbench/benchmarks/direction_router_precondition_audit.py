"""Post-hoc audit of an association precondition for direction routing.

The frozen v3 router is not changed retrospectively. This audit suppresses its
archived attempts only where the declared structural-equation regime has no
association. The result diagnoses a design repair and is not fresh confirmation.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

from mousebrainbench import __version__
from mousebrainbench.artifacts import code_revision
from mousebrainbench.benchmarks.semantic_risk_confirmation import EXPECTED_DIRECTION

DEFAULT_CASES = Path("results/semantic_risk_confirmation/cases.npz")
DEFAULT_OUTPUT = Path("results/direction_router_precondition/summary.json")
DEFAULT_MARKDOWN = Path("results/direction_router_precondition/summary.md")

NO_ASSOCIATION_REGIMES = frozenset({"independent_heavy_tailed"})


def direction_metrics(
    regimes: np.ndarray,
    attempted: np.ndarray,
    predicted: np.ndarray,
) -> dict[str, Any]:
    """Compute routing metrics against declared synthetic directions."""

    expected = np.asarray([EXPECTED_DIRECTION[str(value)] or "none" for value in regimes])
    attempts = np.asarray(attempted, dtype=bool)
    correct = attempts & (predicted == expected)
    identifiable = expected != "none"
    return {
        "cases": len(regimes),
        "attempts": int(attempts.sum()),
        "coverage": float(attempts.mean()),
        "attempted_accuracy": (
            float(correct.sum() / attempts.sum()) if attempts.any() else 0.0
        ),
        "identifiable_regime_attempted_accuracy": (
            float((correct & identifiable).sum() / (attempts & identifiable).sum())
            if (attempts & identifiable).any()
            else 0.0
        ),
        "spurious_attempts_without_reference_direction": int(
            (attempts & ~identifiable).sum()
        ),
    }


def run(
    *,
    cases_path: Path = DEFAULT_CASES,
    output: Path = DEFAULT_OUTPUT,
    markdown: Path = DEFAULT_MARKDOWN,
) -> Path:
    """Compare frozen attempts with the explicit association precondition."""

    archive = np.load(cases_path, allow_pickle=False)
    regimes = archive["regimes"]
    attempted = archive["route_attempted"].astype(bool)
    predicted = archive["route_directions"]
    association_established = ~np.isin(regimes, tuple(NO_ASSOCIATION_REGIMES))
    revised_attempted = attempted & association_established
    original = direction_metrics(regimes, attempted, predicted)
    revised = direction_metrics(regimes, revised_attempted, predicted)
    payload = {
        "version": __version__,
        "git_revision": code_revision(),
        "analysis": "direction_router_association_precondition_audit_v1",
        "analysis_role": "post_confirmation_exploratory_repair",
        "frozen_primary_router_unchanged": True,
        "no_association_regimes": sorted(NO_ASSOCIATION_REGIMES),
        "precondition": (
            "Observational direction routing requires independently declared "
            "evidence that association is established."
        ),
        "original": original,
        "with_association_precondition": revised,
        "removed_attempts": int((attempted & ~revised_attempted).sum()),
        "decision": (
            "precondition_removes_archived_spurious_attempts"
            if revised["spurious_attempts_without_reference_direction"] == 0
            and revised["attempts"] < original["attempts"]
            else "precondition_does_not_resolve_spurious_attempts"
        ),
        "limits": [
            "The no-association regime was identified after the primary v3 result.",
            "Association status is supplied by synthetic ground truth, not estimated from data.",
            "The repair requires a new frozen protocol and fresh cases before confirmatory use.",
        ],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n")
    _write_markdown(payload, markdown)
    return output


def _write_markdown(payload: Mapping[str, Any], path: Path) -> None:
    original = payload["original"]
    revised = payload["with_association_precondition"]
    lines = [
        "# Direction-router association precondition audit",
        "",
        f"- Decision: `{payload['decision']}`",
        f"- Removed attempts: `{payload['removed_attempts']}`",
        "",
        "| Router | Coverage | Attempted accuracy | Spurious attempts |",
        "|---|---:|---:|---:|",
        (
            f"| Frozen v3 | {original['coverage']:.4f} | "
            f"{original['attempted_accuracy']:.4f} | "
            f"{original['spurious_attempts_without_reference_direction']} |"
        ),
        (
            f"| Association precondition | {revised['coverage']:.4f} | "
            f"{revised['attempted_accuracy']:.4f} | "
            f"{revised['spurious_attempts_without_reference_direction']} |"
        ),
        "",
        "## Limits",
        "",
    ]
    lines.extend(f"- {item}" for item in payload["limits"])
    lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--markdown", type=Path, default=DEFAULT_MARKDOWN)
    args = parser.parse_args()
    result = run(cases_path=args.cases, output=args.output, markdown=args.markdown)
    print(json.dumps({"output": str(result.resolve())}))


if __name__ == "__main__":
    main()
