"""Audit raw transfer against the executable population-scope contract."""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from mousebrainbench import __version__
from mousebrainbench.artifacts import code_revision
from mousebrainbench.validation.guarantee_scope import (
    PopulationScope,
    assess_guarantee_scope,
)

DEFAULT_SYNTHETIC = Path("results/semantic_risk_confirmation/summary.json")
DEFAULT_CHAMBERS = Path("results/causal_chambers_transport/summary.json")
DEFAULT_CAUSALBENCH = Path("results/causalbench_transport/summary.json")
DEFAULT_IBL = Path("results/ibl_mouse_transport/summary.json")
DEFAULT_OUTPUT = Path("results/guarantee_scope_audit/summary.json")
DEFAULT_MARKDOWN = Path("results/guarantee_scope_audit/summary.md")


CALIBRATION_SCOPE = PopulationScope(
    scope_id="consumed_v2_sem_calibration",
    population_family="synthetic_sem_claim_cases",
    independent_unit="complete_case",
    evidence_protocol="mouseclaimbench_hybrid_evidence_v2_v3",
    reference_protocol="declared_structural_equation_claim_truth",
)


def _scoped_row(
    *,
    name: str,
    target: PopulationScope,
    raw_authorizations: int,
    raw_false_authorizations: int,
    raw_coverage: float,
    raw_sfar: float,
    detected_shift: bool | None,
) -> dict[str, Any]:
    assessment = assess_guarantee_scope(
        CALIBRATION_SCOPE,
        target,
        detected_shift=detected_shift,
    )
    return {
        "case": name,
        "assessment": assessment.as_dict(),
        "raw_transfer": {
            "authorizations": raw_authorizations,
            "false_authorizations": raw_false_authorizations,
            "coverage": raw_coverage,
            "sfar": raw_sfar,
        },
        "scope_enforced": {
            "authorizations": raw_authorizations if assessment.valid else 0,
            "false_authorizations": raw_false_authorizations if assessment.valid else 0,
            "coverage": raw_coverage if assessment.valid else 0.0,
            "sfar": raw_sfar if assessment.valid else 0.0,
            "decision": "certificate_applied" if assessment.valid else "abstained_out_of_scope",
        },
    }


def run(
    *,
    synthetic_path: Path = DEFAULT_SYNTHETIC,
    chambers_path: Path = DEFAULT_CHAMBERS,
    causalbench_path: Path = DEFAULT_CAUSALBENCH,
    ibl_path: Path = DEFAULT_IBL,
    output: Path = DEFAULT_OUTPUT,
    markdown: Path = DEFAULT_MARKDOWN,
) -> Path:
    """Combine existing raw results without rerunning or editing numerical artifacts."""

    synthetic = json.loads(synthetic_path.read_text())
    chambers = json.loads(chambers_path.read_text())
    causalbench = json.loads(causalbench_path.read_text())
    ibl = json.loads(ibl_path.read_text())
    primary = next(
        row
        for row in synthetic["aggregate_by_policy"]
        if row["policy"] == "semantic_MAPIE_risk_control"
    )
    chamber_test = next(
        row for row in chambers["by_partition"] if row["partition"] == "locked_test"
    )
    causalbench_test = next(
        row for row in causalbench["domains"] if row["role"] == "locked_transport_test"
    )
    ibl_test = next(
        row for row in ibl["partitions"] if row["role"] == "locked_mouse_test"
    )
    rows = [
        _scoped_row(
            name="fresh_synthetic_v3",
            target=PopulationScope(
                scope_id="fresh_synthetic_v3",
                population_family="synthetic_sem_claim_cases",
                independent_unit="complete_case",
                evidence_protocol="mouseclaimbench_hybrid_evidence_v2_v3",
                reference_protocol="declared_structural_equation_claim_truth",
            ),
            raw_authorizations=int(primary["authorizations"]),
            raw_false_authorizations=int(primary["false_authorizations"]),
            raw_coverage=float(primary["supported_coverage"]),
            raw_sfar=float(primary["semantic_false_authorization_risk"]),
            detected_shift=False,
        ),
        _scoped_row(
            name="causal_chambers_locked",
            target=PopulationScope(
                "causal_chambers_locked",
                "physical_chamber_experiments",
                "experiment",
                "physical_pair_evidence",
                "published_physical_graph",
            ),
            raw_authorizations=int(chamber_test["authorizations"]),
            raw_false_authorizations=int(chamber_test["false_authorizations"]),
            raw_coverage=float(chamber_test["supported_coverage"]),
            raw_sfar=float(chamber_test["semantic_false_authorization_risk"]),
            detected_shift=None,
        ),
        _scoped_row(
            name="causalbench_rpe1_locked",
            target=PopulationScope(
                "causalbench_rpe1_locked",
                "single_cell_perturbseq_pairs",
                "source_gene_cluster",
                "two_fold_intervention_effect",
                "third_fold_intervention_effect",
            ),
            raw_authorizations=int(causalbench_test["authorizations"]),
            raw_false_authorizations=int(causalbench_test["false_authorizations"]),
            raw_coverage=float(causalbench_test["supported_coverage"]),
            raw_sfar=float(causalbench_test["semantic_false_authorization_risk"]),
            detected_shift=None,
        ),
        _scoped_row(
            name="ibl_locked_mice",
            target=PopulationScope(
                "ibl_locked_mice",
                "ibl_unit_anatomy_classification",
                "mouse",
                "two_fold_observational_prediction",
                "third_fold_observational_prediction",
            ),
            raw_authorizations=int(ibl_test["authorizations"]),
            raw_false_authorizations=int(ibl_test["false_authorizations"]),
            raw_coverage=float(ibl_test["supported_coverage"]),
            raw_sfar=float(ibl_test["semantic_false_authorization_risk"]),
            detected_shift=None,
        ),
    ]
    passed = rows[0]["assessment"]["valid"] and all(
        not row["assessment"]["valid"]
        and row["scope_enforced"]["authorizations"] == 0
        for row in rows[1:]
    )
    payload = {
        "version": __version__,
        "git_revision": code_revision(),
        "analysis": "finite_sample_guarantee_population_scope_audit_v1",
        "calibration_scope": CALIBRATION_SCOPE.as_dict(),
        "cases": rows,
        "decision": (
            "out_of_scope_certificates_blocked" if passed else "guarantee_scope_audit_failed"
        ),
        "interpretation": (
            "External raw-transfer metrics remain diagnostic results. Scope enforcement "
            "prevents those results from carrying the synthetic LTT guarantee."
        ),
        "limits": [
            "Population identity is a protocol assertion, not a statistical shift estimate.",
            "Out-of-scope abstention prevents misuse but provides no positive external coverage.",
            "A new domain requires its own calibration and a new untouched test population.",
        ],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n")
    _write_markdown(payload, markdown)
    return output


def _write_markdown(payload: Mapping[str, Any], path: Path) -> None:
    lines = [
        "# Finite-sample guarantee scope audit",
        "",
        f"- Decision: `{payload['decision']}`",
        "",
        "| Case | In scope | Raw authorizations | Raw SFAR | Scoped authorizations |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in payload["cases"]:
        lines.append(
            f"| `{row['case']}` | `{str(row['assessment']['valid']).lower()}` | "
            f"{row['raw_transfer']['authorizations']} | {row['raw_transfer']['sfar']:.4f} | "
            f"{row['scope_enforced']['authorizations']} |"
        )
    lines.extend(("", "## Limits", ""))
    lines.extend(f"- {limit}" for limit in payload["limits"])
    lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--markdown", type=Path, default=DEFAULT_MARKDOWN)
    args = parser.parse_args()
    print(json.dumps({"output": str(run(output=args.output, markdown=args.markdown).resolve())}))


if __name__ == "__main__":
    main()
