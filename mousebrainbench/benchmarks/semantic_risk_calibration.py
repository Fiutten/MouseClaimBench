"""Freeze v3 LTT thresholds using only consumed version-2 cases.

The frozen v2 evaluation matrix is now development material. Its labels are
used only to calibrate support thresholds for the unchanged v2 score model.
Fresh v3 synthetic and external outcomes are not read by this module.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from mousebrainbench import __version__
from mousebrainbench.artifacts import code_revision
from mousebrainbench.benchmarks.hybrid_selective_policy import (
    semantic_admissibility_matrix,
)
from mousebrainbench.knowledge import load_default_profile
from mousebrainbench.validation.semantic_risk_control import calibrate_semantic_risk

DEFAULT_PROTOCOL = Path("configs/benchmarks/semantic_risk_control_v3.yaml")
DEFAULT_CASES = Path("results/hybrid_selective_confirmation/cases.npz")
DEFAULT_V2_POLICY = Path("results/hybrid_selective_policy/model.json")
DEFAULT_OUTPUT = Path("results/semantic_risk_policy/model.json")


def _sha256(path: Path) -> str:
    return f"sha256:{hashlib.sha256(path.read_bytes()).hexdigest()}"


def _load_protocol(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text())
    if payload.get("status") != "frozen_before_external_outcome_access":
        raise ValueError("v3 protocol is not frozen before external outcome access")
    return payload


def run(
    *,
    protocol_path: Path = DEFAULT_PROTOCOL,
    cases_path: Path = DEFAULT_CASES,
    v2_policy_path: Path = DEFAULT_V2_POLICY,
    output: Path = DEFAULT_OUTPUT,
) -> Path:
    """Calibrate constrained and unconstrained LTT policies on consumed cases."""

    protocol = _load_protocol(protocol_path)
    archive = np.load(cases_path, allow_pickle=False)
    v2_policy = json.loads(v2_policy_path.read_text())
    claim_names = tuple(str(value) for value in v2_policy["claim_names"])
    labels = archive["labels"].astype(bool)
    probabilities = archive["full_calibrated_probabilities"].astype(float)
    features = archive["features"].astype(float)
    if labels.shape != probabilities.shape:
        raise ValueError("consumed labels and probabilities have incompatible shapes")
    if labels.shape[1] != len(claim_names):
        raise ValueError("consumed claim columns differ from frozen v2 policy")

    profile = load_default_profile()
    complete_requirements = {
        requirement.claim: requirement.required_blocks
        for requirement in profile.requirements
    }
    admissible = semantic_admissibility_matrix(
        features,
        claim_names=claim_names,
        feature_names=v2_policy["feature_names"],
        support_vetoes=complete_requirements,
    )
    variable_indices = np.asarray(
        [index for index in range(labels.shape[1]) if len(np.unique(labels[:, index])) == 2],
        dtype=int,
    )
    variable_claims = tuple(claim_names[index] for index in variable_indices)
    if not variable_claims:
        raise RuntimeError("consumed calibration set has no variable claim families")

    target = float(protocol["risk_control"]["target"]["maximum"])
    confidence = float(protocol["risk_control"]["familywise_confidence"])
    semantic_policy = calibrate_semantic_risk(
        probabilities[:, variable_indices],
        labels[:, variable_indices],
        admissible[:, variable_indices],
        claim_names=variable_claims,
        target_sfar=target,
        familywise_confidence=confidence,
    )
    unconstrained_policy = calibrate_semantic_risk(
        probabilities[:, variable_indices],
        labels[:, variable_indices],
        np.ones((len(labels), len(variable_indices)), dtype=bool),
        claim_names=variable_claims,
        target_sfar=target,
        familywise_confidence=confidence,
    )
    payload = {
        "version": __version__,
        "git_revision": code_revision(),
        "analysis": "semantic_risk_ltt_calibration_v3",
        "data_role": "consumed_v2_development_only",
        "protocol_id": protocol["protocol_id"],
        "protocol_version": str(protocol["version"]),
        "protocol_hash": _sha256(protocol_path),
        "consumed_cases_hash": _sha256(cases_path),
        "v2_score_model_hash": _sha256(v2_policy_path),
        "score_model_refitted": False,
        "thresholds_selected_before_v3_outcome_access": True,
        "calibration_cases": len(labels),
        "all_claims": list(claim_names),
        "variable_claims": list(variable_claims),
        "nonvariable_claims_excluded_from_multiplicity": [
            claim for claim in claim_names if claim not in variable_claims
        ],
        "complete_semantic_requirements": {
            key: list(value) for key, value in complete_requirements.items()
        },
        "semantic_policy": semantic_policy.as_dict(),
        "unconstrained_policy": unconstrained_policy.as_dict(),
        "decision": (
            "semantic_ltt_policy_frozen_for_v3_confirmation"
            if any(item.certified for item in semantic_policy.certificates)
            else "no_variable_claim_obtained_nontrivial_ltt_certificate"
        ),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES)
    parser.add_argument("--v2-policy", type=Path, default=DEFAULT_V2_POLICY)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    path = run(
        protocol_path=args.protocol,
        cases_path=args.cases,
        v2_policy_path=args.v2_policy,
        output=args.output,
    )
    print(json.dumps({"output": str(path.resolve())}))


if __name__ == "__main__":
    main()
