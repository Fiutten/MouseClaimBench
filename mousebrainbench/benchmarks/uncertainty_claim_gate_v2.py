"""Local numerical-stability audit over constructed ClaimBench v2 cases.

The three states emitted here describe local perturbation stability only. They
are distinct from the five workflow dispositions defined by the domain-aware
v3 evidence contract.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path
from typing import Any

from mousebrainbench import __version__
from mousebrainbench.artifacts import code_revision
from mousebrainbench.benchmarks.claim_adversarial_v2 import build_cases
from mousebrainbench.validation.claim_evaluation import CLAIM_TYPES, ClaimEvidence, ClaimGateEvaluator


DEFAULT_OUTPUT = Path("results/uncertainty_claim_gate_v2/summary.json")
DEFAULT_MARKDOWN = Path("results/uncertainty_claim_gate_v2/summary.md")


def _evidence_samples(evidence: ClaimEvidence, sigma: float = 0.03) -> tuple[ClaimEvidence, ...]:
    """Generate deterministic local uncertainty samples around normalized evidence."""

    samples = [evidence]
    numeric_fields = (
        "predictive_score",
        "reproducibility_score",
        "topology_effect",
        "directed_fraction",
        "matched_structure_function_effect",
    )
    for field in numeric_fields:
        value = float(getattr(evidence, field))
        for delta in (-sigma, sigma):
            updated = max(0.0, min(1.0, value + delta))
            samples.append(replace(evidence, **{field: updated}))
    return tuple(samples)


def _claim_probability(evidence: ClaimEvidence, claim: str) -> float:
    evaluator = ClaimGateEvaluator()
    samples = _evidence_samples(evidence)
    hits = sum(claim in evaluator.evaluate(sample).allowed_claims for sample in samples)
    return hits / len(samples)


def _status(probability: float) -> str:
    if probability >= 0.95:
        return "supported"
    if probability <= 0.05:
        return "blocked"
    return "uncertain"


def run(output: Path = DEFAULT_OUTPUT, markdown: Path = DEFAULT_MARKDOWN) -> Path:
    """Run uncertainty-aware claim audit over known-truth v2 cases."""

    rows: list[dict[str, Any]] = []
    unsupported_supported = 0
    supported_uncertain = 0
    for case in build_cases():
        truth = set(case.true_claims)
        for claim in CLAIM_TYPES:
            probability = _claim_probability(case.evidence, claim)
            status = _status(probability)
            expected = claim in truth
            if status == "supported" and not expected:
                unsupported_supported += 1
            if status == "uncertain" and expected:
                supported_uncertain += 1
            rows.append(
                {
                    "case": case.name,
                    "family": case.family,
                    "claim": claim,
                    "probability": probability,
                    "status": status,
                    "expected": expected,
                }
            )
    status_counts = {
        status: sum(row["status"] == status for row in rows)
        for status in ("supported", "uncertain", "blocked")
    }
    payload = {
        "version": __version__,
        "git_revision": code_revision(),
        "analysis": "uncertainty_claim_gate_v2",
        "status_semantics": "local_numerical_stability_not_workflow_disposition",
        "sampling_rule": {
            "nominal_samples": 1,
            "numeric_fields": 5,
            "signed_perturbations_per_field": 2,
            "total_samples_per_case": 11,
            "sigma": 0.03,
            "joint_perturbations": False,
        },
        "num_rows": len(rows),
        "status_counts": status_counts,
        "unsupported_supported": unsupported_supported,
        "supported_uncertain": supported_uncertain,
        "rows": rows,
        "decision": (
            "uncertainty_gate_blocks_unsupported_support"
            if unsupported_supported == 0
            else "uncertainty_gate_requires_revision"
        ),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2))
    write_markdown(payload, markdown)
    return output


def write_markdown(payload: dict[str, Any], markdown: Path) -> None:
    """Write uncertainty-aware gate report."""

    lines = [
        "# Uncertainty-Aware Claim Gate v2",
        "",
        f"- Decision: `{payload['decision']}`",
        f"- Rows: `{payload['num_rows']}`",
        f"- Unsupported claims marked supported: `{payload['unsupported_supported']}`",
        f"- Supported claims marked uncertain: `{payload['supported_uncertain']}`",
        f"- Status counts: `{payload['status_counts']}`",
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
