"""Verify the non-compensatory composition of structural, domain, and integrity gates."""

from __future__ import annotations

import argparse
import json
from itertools import product
from pathlib import Path

from mousebrainbench import __version__
from mousebrainbench.artifacts import code_revision
from mousebrainbench.knowledge import compose_final_authorization

DEFAULT_OUTPUT = Path("results/profile_v2_final_gate/summary.json")
DEFAULT_MARKDOWN = Path("results/profile_v2_final_gate/summary.md")


def evaluate() -> dict[str, object]:
    rows = []
    for structural, domain, integrity in product((False, True), repeat=3):
        expected = structural and domain and integrity
        observed = compose_final_authorization(structural, domain, integrity)
        rows.append(
            {
                "structural_conforms": structural,
                "domain_authorized": domain,
                "integrity_conforms": integrity,
                "expected_authorized": expected,
                "observed_authorized": observed,
                "exact": observed is expected,
            }
        )
    complete = all(row["exact"] for row in rows)
    return {
        "logical_combinations": len(rows),
        "exact_combinations": sum(bool(row["exact"]) for row in rows),
        "truth_table": rows,
        "all_endpoints_passed": complete,
        "interpretation": (
            "This verifies the executable Boolean composition S and A and I. "
            "It does not validate the scientific content of any gate."
        ),
    }


def run(
    *,
    output: Path = DEFAULT_OUTPUT,
    markdown: Path = DEFAULT_MARKDOWN,
) -> Path:
    assessment = evaluate()
    payload = {
        "version": __version__,
        "git_revision": code_revision(),
        "analysis": "profile_v2_three_gate_composition",
        **assessment,
        "decision": (
            "three_gate_composition_confirmed"
            if assessment["all_endpoints_passed"]
            else "three_gate_composition_failed"
        ),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n")
    lines = [
        "# Three-gate composition verification",
        "",
        f"- Decision: `{payload['decision']}`",
        f"- Exact combinations: `{payload['exact_combinations']}/8`",
        "",
        payload["interpretation"],
        "",
    ]
    markdown.parent.mkdir(parents=True, exist_ok=True)
    markdown.write_text("\n".join(lines))
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--markdown", type=Path, default=DEFAULT_MARKDOWN)
    args = parser.parse_args()
    print(run(output=args.output, markdown=args.markdown).resolve())


if __name__ == "__main__":
    main()
