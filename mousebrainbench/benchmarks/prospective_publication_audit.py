"""Adversarial publication audit for the prospective validation package."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any

import yaml

from mousebrainbench import __version__
from mousebrainbench.artifacts import code_revision
from mousebrainbench.benchmarks.prospective_probabilistic_baseline import _protocol_hash
from mousebrainbench.knowledge import load_default_profile_basis

DEFAULT_OUTPUT = Path("results/prospective_publication_audit/summary.json")
DEFAULT_MARKDOWN = Path("results/prospective_publication_audit/summary.md")


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _is_clean_revision(value: Any) -> bool:
    return isinstance(value, str) and value != "unknown" and not value.endswith("-dirty")


def _commit_is_ancestor(root: Path, ancestor: str) -> bool:
    result = subprocess.run(
        ["git", "merge-base", "--is-ancestor", ancestor, "HEAD"],
        cwd=root,
        check=False,
        capture_output=True,
    )
    return result.returncode == 0


def run(
    *,
    output: Path = DEFAULT_OUTPUT,
    markdown: Path = DEFAULT_MARKDOWN,
    root: Path = Path("."),
) -> Path:
    """Audit temporal integrity, outcomes, claim limits, and reviewer-facing risk."""

    paths = {
        "protocol": root / "configs/benchmarks/prospective_claim_validation_v1.yaml",
        "model": root / "results/prospective_probabilistic_baseline/model.json",
        "prospective": root / "results/prospective_claim_validation/summary.json",
        "microns": root / "results/microns_network_inference/summary.json",
        "dyadic_calibration": root / "results/dyadic_inference_calibration/summary.json",
    }
    missing = [name for name, path in paths.items() if not path.exists()]
    if missing:
        raise FileNotFoundError(f"publication audit inputs missing: {missing}")

    protocol = yaml.safe_load(paths["protocol"].read_text())
    model = _load(paths["model"])
    prospective = _load(paths["prospective"])
    microns = _load(paths["microns"])
    calibration = _load(paths["dyadic_calibration"])
    basis = load_default_profile_basis()
    protocol_integrity = bool(
        protocol["status"] == "frozen_before_execution"
        and model["protocol_hash"] == _protocol_hash(paths["protocol"])
        and model["prospective_data_used_for_training"] is False
        and model["training_partition"] == "development_partition_only"
        and prospective["prospective_model_refitting_performed"] is False
        and prospective["scale_matches_frozen_protocol"] is True
        and _commit_is_ancestor(root, prospective["protocol_anchor_commit"])
    )
    artifact_revisions = {
        "probabilistic_model": model.get("git_revision"),
        "prospective_validation": prospective.get("git_revision"),
        "microns_network_inference": microns.get("git_revision"),
        "dyadic_calibration": calibration.get("git_revision"),
    }
    artifacts_clean = all(_is_clean_revision(value) for value in artifact_revisions.values())
    synthetic_primary_passed = bool(prospective["primary_endpoint"]["passed"])
    # v0.12.2 separates direction selection in discovery from confirmation in
    # the two pre-fixed hold-outs. The fallback keeps historical artifacts
    # readable without assigning their legacy field the new semantics.
    microns_passed = bool(
        microns.get("confirmation_passed", microns.get("all_cohorts_passed", False))
    )
    dyadic_calibration_passed = bool(calibration["calibration_sane"])
    expert_validation_performed = basis.get("independent_expert_validation") != "not_performed"

    supported_findings = [
        {
            "finding": "prospective protocol integrity",
            "supported": protocol_integrity,
            "scope": "temporal and partition separation for the computational experiment",
        },
        {
            "finding": "network-aware MICRONS association",
            "supported": microns_passed,
            "scope": (
                "fixed local observational endpoint confirmed in two pre-fixed "
                "hold-out windows after discovery selected the direction"
            ),
        },
        {
            "finding": "dyadic covariance implementation sanity",
            "supported": dyadic_calibration_passed,
            "scope": "additive sender/receiver dependence used in the calibration DGP",
        },
    ]
    rejected_findings = [
        {
            "claim": "general superiority of the non-compensatory contract",
            "rejected": not synthetic_primary_passed,
            "evidence": prospective["primary_endpoint"],
        },
        {
            "claim": "independent expert validation of the knowledge profile",
            "rejected": not expert_validation_performed,
            "evidence": basis.get("independent_expert_validation"),
        },
    ]
    reviewer_risks = [
        {
            "severity": "critical",
            "risk": "The frozen prospective superiority endpoint failed.",
            "consequence": (
                "A manuscript cannot claim that the hard contract is generally more accurate "
                "or safer than learned and compensatory alternatives."
            ),
            "resolved": False,
        },
        {
            "severity": "critical",
            "risk": "The author-proposed knowledge profile lacks independent expert content validation.",
            "consequence": (
                "Formal correctness and computational error rates do not establish domain consensus."
            ),
            "resolved": expert_validation_performed,
        },
        {
            "severity": "major",
            "risk": "The directional diagnostic has substantial prospective error.",
            "consequence": (
                "Direction and mechanistic authorizations are the dominant false-positive and "
                "false-negative source."
            ),
            "resolved": False,
        },
        {
            "severity": "major",
            "risk": "MICRONS cohorts are internal windows from one observational resource.",
            "consequence": (
                "Network-robust significance cannot be described as causal or independent "
                "biological replication."
            ),
            "resolved": False,
        },
        {
            "severity": "major",
            "risk": "The node-permutation test assumes residual-array exchangeability.",
            "consequence": (
                "Unmodeled spatial or cell-class heterogeneity may affect its calibration."
            ),
            "resolved": False,
        },
        {
            "severity": "moderate",
            "risk": "Known-truth validation remains low-dimensional and profile-specific.",
            "consequence": "Cross-domain and realistic neural-system generality remain untested.",
            "resolved": False,
        },
        {
            "severity": "moderate",
            "risk": "No human decision-support outcome has been measured.",
            "consequence": "Usability or improved reviewer decisions cannot be claimed.",
            "resolved": False,
        },
    ]
    strong_q1_superiority_ready = bool(
        protocol_integrity
        and artifacts_clean
        and synthetic_primary_passed
        and microns_passed
        and dyadic_calibration_passed
    )
    payload = {
        "version": __version__,
        "git_revision": code_revision(),
        "analysis": "prospective_publication_readiness_audit",
        "protocol_integrity": protocol_integrity,
        "artifact_revisions": artifact_revisions,
        "artifacts_clean": artifacts_clean,
        "synthetic_primary_endpoint_passed": synthetic_primary_passed,
        "microns_network_endpoint_passed": microns_passed,
        "dyadic_calibration_passed": dyadic_calibration_passed,
        "independent_expert_validation_performed": expert_validation_performed,
        "supported_findings": supported_findings,
        "rejected_findings": rejected_findings,
        "reviewer_risks": reviewer_risks,
        "strong_q1_superiority_ready": strong_q1_superiority_ready,
        "decision": (
            "strong_q1_superiority_package_ready"
            if strong_q1_superiority_ready
            else "strong_q1_superiority_claim_blocked_by_prospective_evidence"
        ),
        "recommended_disposition": (
            "Retain this package as a transparent negative validation and MICRONS "
            "robustness extension. Do not center a new Q1 manuscript on superiority of "
            "the current hard contract. A new version requires a better directional "
            "evidence model and a separately frozen evaluation partition."
        ),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2))
    write_markdown(payload, markdown)
    return output


def write_markdown(payload: dict[str, Any], markdown: Path) -> None:
    lines = [
        "# Prospective publication audit",
        "",
        f"- Decision: `{payload['decision']}`",
        f"- Protocol integrity: `{payload['protocol_integrity']}`",
        f"- Clean input artifacts: `{payload['artifacts_clean']}`",
        f"- Synthetic primary endpoint: `{payload['synthetic_primary_endpoint_passed']}`",
        f"- MICRONS network endpoint: `{payload['microns_network_endpoint_passed']}`",
        f"- Dyadic calibration: `{payload['dyadic_calibration_passed']}`",
        "",
        "## Reviewer risks",
        "",
    ]
    for row in payload["reviewer_risks"]:
        lines.append(
            f"- **{row['severity']}**: {row['risk']} "
            f"Resolved: `{row['resolved']}`. {row['consequence']}"
        )
    lines.extend(
        ["", "## Recommended disposition", "", payload["recommended_disposition"], ""]
    )
    markdown.parent.mkdir(parents=True, exist_ok=True)
    markdown.write_text("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--markdown", type=Path, default=DEFAULT_MARKDOWN)
    parser.add_argument("--root", type=Path, default=Path("."))
    args = parser.parse_args()
    print(
        json.dumps(
            {"output": str(run(output=args.output, markdown=args.markdown, root=args.root).resolve())}
        )
    )


if __name__ == "__main__":
    main()
