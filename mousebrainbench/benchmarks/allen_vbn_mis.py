"""Apply mechanistic-identifiability scoring to sealed Allen VBN results."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from mousebrainbench import __version__
from mousebrainbench.artifacts import code_revision
from mousebrainbench.validation.mechanistic_identifiability import (
    Criterion,
    MechanisticIdentifiabilityScore,
    build_mis_from_blocks,
)


DEFAULT_PHASE2C = Path("results/phase2c_confirmation_metrics.json")
DEFAULT_PHASE3 = Path("results/phase3_confirmation_metrics.json")
DEFAULT_PHASE4 = Path("results/phase4_identifiability_metrics.json")
DEFAULT_OUTPUT = Path("results/allen_vbn_mechanistic_identifiability_score.json")


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"required result file not found: {path}")
    return json.loads(path.read_text())


def build_allen_vbn_mis(
    *,
    phase2c: dict[str, Any],
    phase3: dict[str, Any],
    phase4: dict[str, Any],
) -> MechanisticIdentifiabilityScore:
    """Build the preregistered MIS used for Allen VBN negative-case evaluation.

    The thresholds are deliberately conservative and interpretable. Phase 2c asks
    whether the target itself is reproducible; Phase 3 asks whether Allen
    anatomy improves prediction over topology controls; Phase 4 asks whether a
    directed timing signature can be resolved from the recorded data.
    """

    target = phase2c["target_result"]
    latency = phase4["latency"]
    lead_lag = phase4["lead_lag"]
    return build_mis_from_blocks(
        reproducibility=(
            Criterion(
                "median_cross_mouse_correlation",
                float(target["median_cross_mouse_correlation"]),
                0.50,
            ),
            Criterion(
                "median_split_half_correlation",
                float(target["median_split_half_correlation"]),
                0.70,
            ),
            Criterion(
                "fraction_above_own_null_95th",
                float(target["fraction_above_own_null_95th"]),
                0.50,
                "gte",
            ),
        ),
        topology_specificity=(
            Criterion(
                "allen_minus_median_permutation",
                float(phase3["allen_minus_median_permutation"]),
                0.05,
            ),
            Criterion(
                "fraction_permutations_outperformed",
                float(phase3["fraction_permutations_outperformed"]),
                0.95,
                "gte",
            ),
            Criterion(
                "paired_advantage_bootstrap_95_lower",
                float(phase3["paired_advantage_bootstrap_95_interval"][0]),
                0.0,
            ),
            Criterion(
                "allen_minus_disconnected",
                float(phase3["allen_minus_disconnected"]),
                0.05,
            ),
        ),
        directed_identifiability=(
            Criterion(
                "latency_median_cross_mouse_tau",
                float(latency["median_cross_mouse_tau"]),
                0.30,
            ),
            Criterion(
                "latency_median_split_half_tau",
                float(latency["median_split_half_tau"]),
                0.50,
            ),
            Criterion(
                "latency_fraction_above_own_null_95th",
                float(latency["fraction_above_own_null_95th"]),
                0.50,
                "gte",
            ),
            Criterion(
                "latency_median_resolvable_pair_fraction",
                float(latency["median_resolvable_pair_fraction"]),
                0.50,
                "gte",
            ),
            Criterion(
                "lead_lag_fraction_above_own_null_95th",
                float(lead_lag["fraction_above_own_null_95th"]),
                0.50,
                "gte",
            ),
            Criterion(
                "lead_lag_median_nonzero_pair_fraction",
                float(lead_lag["median_nonzero_pair_fraction"]),
                0.25,
                "gte",
            ),
        ),
    )


def run(
    *,
    phase2c_path: Path = DEFAULT_PHASE2C,
    phase3_path: Path = DEFAULT_PHASE3,
    phase4_path: Path = DEFAULT_PHASE4,
    output: Path = DEFAULT_OUTPUT,
) -> Path:
    """Score Allen VBN with fixed mechanistic-identifiability gates."""

    phase2c = _read_json(phase2c_path)
    phase3 = _read_json(phase3_path)
    phase4 = _read_json(phase4_path)
    mis = build_allen_vbn_mis(phase2c=phase2c, phase3=phase3, phase4=phase4)
    payload = {
        "version": __version__,
        "git_revision": code_revision(),
        "interpretation": "allen_vbn_negative_mechanistic_identifiability_case",
        "inputs": {
            "phase2c": str(phase2c_path),
            "phase3": str(phase3_path),
            "phase4": str(phase4_path),
        },
        "mis": mis.as_dict(),
        "decision": (
            "mechanistic_claim_supported"
            if mis.passed
            else "reproducible_target_without_mechanistic_identifiability"
        ),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2))
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase2c", type=Path, default=DEFAULT_PHASE2C)
    parser.add_argument("--phase3", type=Path, default=DEFAULT_PHASE3)
    parser.add_argument("--phase4", type=Path, default=DEFAULT_PHASE4)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    path = run(
        phase2c_path=args.phase2c,
        phase3_path=args.phase3,
        phase4_path=args.phase4,
        output=args.output,
    )
    print(json.dumps({"output": str(path.resolve())}))


if __name__ == "__main__":
    main()
