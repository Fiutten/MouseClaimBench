"""Release audit for the bounded semantic-risk-control v3 contribution."""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from mousebrainbench import __version__
from mousebrainbench.artifacts import code_revision

DEFAULT_OUTPUT = Path("results/semantic_risk_v3_release/summary.json")
DEFAULT_MARKDOWN = Path("results/semantic_risk_v3_release/summary.md")

ARTIFACTS = {
    "frozen_policy": Path("results/semantic_risk_policy/model.json"),
    "synthetic": Path("results/semantic_risk_confirmation/summary.json"),
    "asp_equivalence": Path("results/semantic_equivalence_audit/summary.json"),
    "causal_chambers": Path("results/causal_chambers_transport/summary.json"),
    "causalbench": Path("results/causalbench_transport/summary.json"),
    "ibl": Path("results/ibl_mouse_transport/summary.json"),
    "guarantee_scope": Path("results/guarantee_scope_audit/summary.json"),
    "sensitivity": Path("results/semantic_risk_sensitivity/summary.json"),
    "router_repair": Path("results/direction_router_precondition/summary.json"),
}


def _row(rows: list[dict[str, Any]], key: str, value: str) -> dict[str, Any]:
    return next(row for row in rows if row[key] == value)


def _dirty_revision(payload: Mapping[str, Any]) -> bool:
    revision = payload.get("git_revision")
    return not isinstance(revision, str) or revision.endswith("-dirty")


def evaluate(payloads: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    """Evaluate methodological and scientific states without conflating them."""

    policy = payloads["frozen_policy"]
    synthetic = payloads["synthetic"]
    asp = payloads["asp_equivalence"]
    chambers = payloads["causal_chambers"]
    causalbench = payloads["causalbench"]
    ibl = payloads["ibl"]
    scope = payloads["guarantee_scope"]
    sensitivity = payloads["sensitivity"]
    router = payloads["router_repair"]

    synthetic_primary = _row(
        synthetic["aggregate_by_policy"],
        "policy",
        "semantic_MAPIE_risk_control",
    )
    chambers_locked = _row(chambers["by_partition"], "partition", "locked_test")
    causalbench_locked = _row(
        causalbench["domains"], "role", "locked_transport_test"
    )
    ibl_locked = _row(ibl["partitions"], "role", "locked_mouse_test")
    scope_cases = {row["case"]: row for row in scope["cases"]}

    conditions = {
        "policy_frozen_before_fresh_outcome_access": bool(
            policy.get("thresholds_selected_before_v3_outcome_access")
        ),
        "synthetic_primary_passed": bool(synthetic.get("computational_primary_passed")),
        "synthetic_zero_semantic_violations": (
            synthetic_primary.get("semantic_support_violations") == 0
        ),
        "asp_python_semantics_equivalent_on_audited_space": (
            asp.get("decision") == "semantic_equivalence_observed"
            and asp.get("mismatch_count") == 0
            and int(asp.get("evaluated_case_count", 0)) >= 2_847
        ),
        "causal_chambers_zero_semantic_violations": (
            chambers_locked.get("semantic_support_violations") == 0
        ),
        "causalbench_zero_semantic_violations": (
            causalbench_locked.get("semantic_support_violations") == 0
        ),
        "ibl_zero_semantic_violations": (
            ibl_locked.get("semantic_support_violations") == 0
        ),
        "out_of_scope_certificates_blocked": (
            scope.get("decision") == "out_of_scope_certificates_blocked"
            and scope_cases["fresh_synthetic_v3"]["assessment"]["valid"] is True
            and all(
                scope_cases[name]["assessment"]["valid"] is False
                and scope_cases[name]["scope_enforced"]["authorizations"] == 0
                for name in (
                    "causal_chambers_locked",
                    "causalbench_rpe1_locked",
                    "ibl_locked_mice",
                )
            )
        ),
        "sensitivity_is_explicitly_exploratory": (
            sensitivity.get("analysis_role")
            == "post_confirmation_exploratory_sensitivity"
            and sensitivity.get("confirmatory_reuse_prohibited") is True
            and sensitivity.get("semantic_support_violations") == 0
        ),
        "router_repair_is_explicitly_post_hoc": (
            router.get("analysis_role") == "post_confirmation_exploratory_repair"
            and router.get("frozen_primary_router_unchanged") is True
            and router.get("decision")
            == "precondition_removes_archived_spurious_attempts"
        ),
    }
    dirty = sorted(name for name, value in payloads.items() if _dirty_revision(value))
    integrity_ready = all(conditions.values()) and not dirty

    external_raw_risk_failure_detected = (
        chambers_locked["semantic_false_authorization_risk"] > 0.05
        and causalbench_locked["semantic_false_authorization_risk"] > 0.05
    )
    external_positive_coverage = (
        ibl_locked["authorizations"] > 0
        and causalbench_locked["semantic_false_authorization_risk"] <= 0.05
        and causalbench_locked["supported_coverage"] >= 0.05
    )
    external_generalization_established = bool(
        external_positive_coverage
        and all(
            scope_cases[name]["assessment"]["valid"]
            for name in (
                "causal_chambers_locked",
                "causalbench_rpe1_locked",
                "ibl_locked_mice",
            )
        )
    )
    if not integrity_ready:
        decision = "semantic_risk_v3_release_requires_action"
    elif external_generalization_established:
        decision = "methodological_core_and_external_generalization_validated"
    else:
        decision = "methodological_core_validated_external_generalization_not_established"

    return {
        "conditions": conditions,
        "dirty_artifacts": dirty,
        "integrity_ready": integrity_ready,
        "methodological_core_validated": integrity_ready,
        "external_raw_risk_failure_detected": external_raw_risk_failure_detected,
        "external_positive_coverage": external_positive_coverage,
        "external_generalization_established": external_generalization_established,
        "decision": decision,
        "publication_posture": (
            "bounded_Q1_method_candidate_not_universal_risk_controller"
            if integrity_ready and not external_generalization_established
            else "not_ready_for_bounded_method_claim"
        ),
        "prohibited_claims": [
            "finite-sample risk control transfers unchanged across domains",
            "positive external biological generalization is established",
            "the direction router is universally valid",
            "MouseClaimBench verifies scientific truth",
            "a complete or causal mouse-brain digital twin has been built",
        ],
        "required_next_evidence_for_stronger_claim": [
            "domain-specific calibration on exchangeable external units",
            "a new untouched external test population with non-zero authorized coverage",
            "prospective confirmation of the association-aware direction router",
        ],
    }


def run(
    *,
    root: Path = Path("."),
    output: Path = DEFAULT_OUTPUT,
    markdown: Path = DEFAULT_MARKDOWN,
) -> Path:
    """Load immutable artifacts and write a claim-bounded release decision."""

    missing = [name for name, path in ARTIFACTS.items() if not (root / path).exists()]
    if missing:
        raise FileNotFoundError(f"missing semantic-risk artifacts: {', '.join(missing)}")
    payloads = {
        name: json.loads((root / path).read_text()) for name, path in ARTIFACTS.items()
    }
    assessment = evaluate(payloads)
    payload = {
        "version": __version__,
        "git_revision": code_revision(),
        "analysis": "semantic_risk_control_v3_release_audit",
        "artifacts": {
            name: {
                "path": str(path),
                "git_revision": payloads[name].get("git_revision"),
            }
            for name, path in ARTIFACTS.items()
        },
        **assessment,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n")
    _write_markdown(payload, markdown)
    return output


def _write_markdown(payload: Mapping[str, Any], path: Path) -> None:
    lines = [
        "# Semantic risk control v3 release audit",
        "",
        f"- Decision: `{payload['decision']}`",
        f"- Integrity ready: `{str(payload['integrity_ready']).lower()}`",
        (
            f"- Methodological core validated: "
            f"`{str(payload['methodological_core_validated']).lower()}`"
        ),
        (
            f"- External generalization established: "
            f"`{str(payload['external_generalization_established']).lower()}`"
        ),
        f"- Publication posture: `{payload['publication_posture']}`",
        "",
        "## Release conditions",
        "",
    ]
    lines.extend(
        f"- `{name}`: `{str(value).lower()}`"
        for name, value in payload["conditions"].items()
    )
    lines.extend(("", "## Prohibited claims", ""))
    lines.extend(f"- {item}" for item in payload["prohibited_claims"])
    lines.extend(("", "## Evidence required for a stronger claim", ""))
    lines.extend(
        f"- {item}" for item in payload["required_next_evidence_for_stronger_claim"]
    )
    lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--markdown", type=Path, default=DEFAULT_MARKDOWN)
    args = parser.parse_args()
    result = run(root=args.root, output=args.output, markdown=args.markdown)
    print(json.dumps({"output": str(result.resolve())}))


if __name__ == "__main__":
    main()
