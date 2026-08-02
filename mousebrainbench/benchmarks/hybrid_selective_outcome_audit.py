"""Audit the immutable hybrid v2 outcome without refitting or regeneration."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from mousebrainbench import __version__
from mousebrainbench.artifacts import code_revision
from mousebrainbench.benchmarks.hybrid_selective_confirmation import DIRECTION_TRUTHS


DEFAULT_SUMMARY = Path("results/hybrid_selective_confirmation/summary.json")
DEFAULT_CASES = Path("results/hybrid_selective_confirmation/cases.npz")
DEFAULT_OUTPUT = Path("results/hybrid_selective_outcome_audit/audit.json")
DEFAULT_MARKDOWN = Path("results/hybrid_selective_outcome_audit/audit.md")

VARIABLE_CLAIMS = (
    "predictive",
    "internally_reproduced",
    "topology_specific",
    "directed",
    "mechanistic",
    "causal",
)


def _sha256(path: Path) -> str:
    return f"sha256:{hashlib.sha256(path.read_bytes()).hexdigest()}"


def _ece(probabilities: np.ndarray, labels: np.ndarray, bins: int = 10) -> float:
    edges = np.linspace(0.0, 1.0, bins + 1)
    assignments = np.clip(np.digitize(probabilities, edges[1:-1]), 0, bins - 1)
    total = len(probabilities)
    value = 0.0
    for index in range(bins):
        mask = assignments == index
        if not mask.any():
            continue
        value += float(mask.mean()) * abs(
            float(probabilities[mask].mean()) - float(labels[mask].mean())
        )
    return value if total else 0.0


def _selective_metrics(decisions: np.ndarray, labels: np.ndarray) -> dict[str, Any]:
    decided = decisions != 0
    supports = decisions == 1
    errors = ((decisions == 1) & ~labels) | ((decisions == -1) & labels)
    return {
        "coverage": float(decided.mean()),
        "selective_error": (
            float(errors[decided].mean()) if decided.any() else 1.0
        ),
        "false_authorizations": int((supports & ~labels).sum()),
        "supports": int(supports.sum()),
        "false_authorization_fraction": (
            float((supports & ~labels).sum() / supports.sum())
            if supports.any()
            else 0.0
        ),
    }


def _direction_by_regime(
    regimes: np.ndarray,
    statuses: np.ndarray,
    predictions: np.ndarray,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for regime, expected in DIRECTION_TRUTHS.items():
        mask = regimes == regime
        attempted = mask & (statuses != "requires_review")
        correct_direction = (
            "forward" if expected == "x_to_y" else "reverse" if expected == "y_to_x" else None
        )
        correct = attempted & (predictions == correct_direction) if correct_direction else np.zeros(len(regimes), dtype=bool)
        rows.append(
            {
                "regime": regime,
                "structural_direction": expected or "none",
                "cases": int(mask.sum()),
                "attempts": int(attempted.sum()),
                "coverage": float(attempted.sum() / mask.sum()),
                "attempted_accuracy": (
                    float(correct.sum() / attempted.sum())
                    if attempted.any() and correct_direction
                    else 0.0
                ),
                "forward": int((mask & (predictions == "forward")).sum()),
                "reverse": int((mask & (predictions == "reverse")).sum()),
                "review": int((mask & (statuses == "requires_review")).sum()),
            }
        )
    return rows


def run(
    *,
    summary_path: Path = DEFAULT_SUMMARY,
    cases_path: Path = DEFAULT_CASES,
    output: Path = DEFAULT_OUTPUT,
    markdown: Path = DEFAULT_MARKDOWN,
) -> Path:
    """Produce a descriptive audit from the already-frozen confirmation files."""

    summary = json.loads(summary_path.read_text())
    if summary["cases_sha256"] != _sha256(cases_path):
        raise ValueError("confirmation case archive differs from frozen summary")
    if summary["confirmatory_model_refitting_performed"]:
        raise ValueError("confirmation summary reports forbidden refitting")
    archive = np.load(cases_path)
    policy_names = archive["policy_names"].astype(str).tolist()
    decisions = {
        name: archive["policy_decisions"][index]
        for index, name in enumerate(policy_names)
    }
    labels = archive["labels"].astype(bool)
    claim_names = json.loads(
        Path("results/hybrid_selective_policy/model.json").read_text()
    )["claim_names"]
    claim_indices = {claim: index for index, claim in enumerate(claim_names)}
    variable_indices = np.asarray([claim_indices[claim] for claim in VARIABLE_CLAIMS])

    per_variable_claim = []
    for claim in VARIABLE_CLAIMS:
        index = claim_indices[claim]
        per_variable_claim.append(
            {
                "claim": claim,
                "policies": {
                    name: _selective_metrics(values[:, index], labels[:, index])
                    for name, values in decisions.items()
                },
            }
        )
    variable_claim_aggregate = {
        name: _selective_metrics(values[:, variable_indices], labels[:, variable_indices])
        for name, values in decisions.items()
    }

    calibrated = archive["full_calibrated_probabilities"][:, variable_indices].ravel()
    raw = archive["full_raw_probabilities"][:, variable_indices].ravel()
    variable_labels = labels[:, variable_indices].ravel()
    calibration = {
        "scope": list(VARIABLE_CLAIMS),
        "raw": {
            "brier_score": float(np.mean(np.square(raw - variable_labels))),
            "ece_10_equal_width_bins": _ece(raw, variable_labels),
        },
        "isotonic": {
            "brier_score": float(np.mean(np.square(calibrated - variable_labels))),
            "ece_10_equal_width_bins": _ece(calibrated, variable_labels),
        },
        "interpretation": "descriptive postconfirmation audit; no threshold was changed",
    }
    direction_rows = _direction_by_regime(
        archive["regimes"].astype(str),
        archive["anm_status"].astype(str),
        archive["anm_direction"].astype(str),
    )
    no_direction = [row for row in direction_rows if row["structural_direction"] == "none"]
    no_direction_cases = sum(row["cases"] for row in no_direction)
    no_direction_attempts = sum(row["attempts"] for row in no_direction)
    direction_failure = {
        "by_regime": direction_rows,
        "spurious_attempt_rate_in_no_direction_regimes": float(
            no_direction_attempts / no_direction_cases
        ),
        "principal_failures": [
            "direct_post_nonlinear is outside the additive-noise assumption and is predominantly reversed",
            "measurement_error_direct remains close to chance among attempted directions",
            "confounded and collider regimes receive many assumption-invalid forced directions",
        ],
    }

    constrained = variable_claim_aggregate["constrained_selective_hybrid"]
    unconstrained = variable_claim_aggregate["unconstrained_selective_logistic"]
    payload = {
        "version": __version__,
        "git_revision": code_revision(),
        "analysis": "postconfirmation_hybrid_v2_outcome_audit",
        "source_summary_hash": _sha256(summary_path),
        "source_cases_hash": _sha256(cases_path),
        "source_primary_decision": summary["decision"],
        "source_primary_passed": summary["primary_endpoint"]["passed"],
        "confirmation_reexecuted": False,
        "model_refitted": False,
        "thresholds_changed": False,
        "variable_claim_aggregate": variable_claim_aggregate,
        "per_variable_claim": per_variable_claim,
        "calibration": calibration,
        "direction_failure": direction_failure,
        "engineering_signal": {
            "all_claim_hybrid_false_authorization_fraction": next(
                row["false_authorization_fraction"]
                for row in summary["aggregate_by_policy"]
                if row["policy"] == "constrained_selective_hybrid"
            ),
            "all_claim_unconstrained_false_authorization_fraction": next(
                row["false_authorization_fraction"]
                for row in summary["aggregate_by_policy"]
                if row["policy"] == "unconstrained_selective_logistic"
            ),
            "variable_claim_hybrid_selective_error": constrained["selective_error"],
            "variable_claim_unconstrained_selective_error": unconstrained[
                "selective_error"
            ],
            "interpretation": (
                "hard vetoes enforce zero semantic violations and reduce false support, "
                "but do not establish lower predictive error"
            ),
        },
        "publication_assessment": {
            "strong_new_q1_claim_supported": False,
            "negative_result_is_reproducible": True,
            "reason": (
                "The frozen primary endpoint failed materially on ANM direction, while "
                "aggregate error is diluted by constant-label claims and calibration adds "
                "no clear confirmatory advantage."
            ),
            "defensible_claim": (
                "A constrained selective layer can enforce semantic support boundaries "
                "with non-trivial coverage, but the tested ANM gate is not a reliable "
                "general directional-evidence component across assumption violations."
            ),
            "prohibited_next_action": (
                "Do not tune on v2 and present a rerun as independent confirmation."
            ),
        },
        "decision": "negative_confirmatory_result_with_partial_engineering_signal",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2))
    write_markdown(payload, markdown)
    return output


def write_markdown(payload: Mapping[str, Any], markdown: Path) -> None:
    assessment = payload["publication_assessment"]
    lines = [
        "# Hybrid selective v2 outcome audit",
        "",
        f"- Decision: `{payload['decision']}`",
        f"- Frozen primary passed: `{payload['source_primary_passed']}`",
        f"- Strong new Q1 claim supported: `{assessment['strong_new_q1_claim_supported']}`",
        f"- Confirmation reexecuted: `{payload['confirmation_reexecuted']}`",
        f"- Model refitted: `{payload['model_refitted']}`",
        "",
        "## Variable-claim performance",
        "",
        "| Policy | Coverage | Selective error | False authorization fraction |",
        "|---|---:|---:|---:|",
    ]
    for policy, metrics in payload["variable_claim_aggregate"].items():
        lines.append(
            f"| `{policy}` | {metrics['coverage']:.4f} | "
            f"{metrics['selective_error']:.4f} | "
            f"{metrics['false_authorization_fraction']:.4f} |"
        )
    lines.extend(
        [
            "",
            "## Direction by regime",
            "",
            "| Regime | Truth | Coverage | Attempted accuracy | Forward | Reverse | Review |",
            "|---|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in payload["direction_failure"]["by_regime"]:
        lines.append(
            f"| `{row['regime']}` | `{row['structural_direction']}` | "
            f"{row['coverage']:.4f} | {row['attempted_accuracy']:.4f} | "
            f"{row['forward']} | {row['reverse']} | {row['review']} |"
        )
    lines.extend(
        [
            "",
            "## Scientific assessment",
            "",
            assessment["reason"],
            "",
            f"Defensible claim: {assessment['defensible_claim']}",
            "",
            f"Blocked action: {assessment['prohibited_next_action']}",
            "",
        ]
    )
    markdown.parent.mkdir(parents=True, exist_ok=True)
    markdown.write_text("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--markdown", type=Path, default=DEFAULT_MARKDOWN)
    args = parser.parse_args()
    print(
        json.dumps(
            {
                "output": str(
                    run(
                        summary_path=args.summary,
                        cases_path=args.cases,
                        output=args.output,
                        markdown=args.markdown,
                    ).resolve()
                )
            }
        )
    )


if __name__ == "__main__":
    main()
