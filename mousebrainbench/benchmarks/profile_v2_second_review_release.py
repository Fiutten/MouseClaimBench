"""Audit the bounded response package for the second major review."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import yaml

from mousebrainbench import __version__
from mousebrainbench.artifacts import code_revision

DEFAULT_PROTOCOL = Path(
    "configs/benchmarks/profile_v2_second_review_release.yaml"
)
DEFAULT_OUTPUT = Path("results/profile_v2_second_review_release/summary.json")
DEFAULT_MARKDOWN = Path("results/profile_v2_second_review_release/summary.md")


def _load(path: str) -> dict[str, Any]:
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"second-review artifact is missing: {source}")
    return json.loads(source.read_text())


def _clean_revision(payload: dict[str, Any]) -> bool:
    revision = payload.get("git_revision")
    return (
        isinstance(revision, str)
        and bool(revision)
        and revision != "unknown"
        and not revision.endswith("-dirty")
    )


def evaluate(protocol: dict[str, Any]) -> dict[str, Any]:
    """Evaluate the exact, non-compensatory release conditions."""

    artifacts = {
        name: _load(path) for name, path in protocol["artifacts"].items()
    }
    required = protocol["requirements"]
    parent = artifacts["major_revision"]
    mutation = artifacts["contract_mutation"]
    thresholds = artifacts["dandi_threshold_sensitivity"]
    conditions = {
        "major_revision_release_preserved": (
            parent.get("decision")
            == "profile_v2_major_revision_release_complete"
            and parent.get("all_release_conditions_passed") is True
        ),
        "asp_covers_every_contract_case": (
            mutation.get("cases") == required["contract_cases"]
            and mutation.get("asp_conformance", {}).get("cases")
            == required["asp_cases"]
            and mutation.get("asp_conformance", {}).get("rate") == 1.0
        ),
        "dandi_threshold_boundaries_complete": (
            thresholds.get("decision")
            == "dandi_threshold_sensitivity_complete"
            and len(thresholds.get("one_at_a_time_results", {}))
            == required["dandi_sensitivity_criteria"]
            and thresholds.get("completed") is True
        ),
        "artifact_revisions_clean": all(
            _clean_revision(payload) for payload in artifacts.values()
        ),
        "prohibited_claims_remain_false": not any(
            protocol["claim_policy"].values()
        ),
    }
    complete = all(conditions.values())
    return {
        "artifact_revisions": {
            name: payload.get("git_revision")
            for name, payload in artifacts.items()
        },
        "conditions": conditions,
        "all_release_conditions_passed": complete,
        "claim_policy": protocol["claim_policy"],
        "remaining_limits": [
            "profile v2 has no completed independent expert content validation",
            "the DANDI thresholds remain author-defined operational criteria",
            "one-edge relation perturbations are monotonicity probes, not alternative profile validation",
            "counterfactual fidelity does not establish human explanation utility",
            "the performance result is a core-engine microbenchmark on one host",
        ],
        "interpretation": protocol["interpretation"],
    }


def _write_markdown(payload: dict[str, Any], path: Path) -> None:
    lines = [
        "# Profile v2 second-review release",
        "",
        f"- Decision: `{payload['decision']}`",
        "",
        "| Release condition | Passed |",
        "|---|---:|",
    ]
    lines.extend(
        f"| `{name}` | {str(passed).lower()} |"
        for name, passed in payload["conditions"].items()
    )
    lines.extend(("", "## Remaining limits", ""))
    lines.extend(f"- {item}" for item in payload["remaining_limits"])
    lines.extend(("", payload["interpretation"], ""))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines))


def run(
    *,
    protocol_path: Path = DEFAULT_PROTOCOL,
    output: Path = DEFAULT_OUTPUT,
    markdown: Path = DEFAULT_MARKDOWN,
) -> Path:
    """Run and persist the second-review release audit."""

    protocol = yaml.safe_load(protocol_path.read_text())
    assessment = evaluate(protocol)
    payload = {
        "version": __version__,
        "git_revision": code_revision(),
        "analysis": "profile_v2_second_review_release",
        "protocol": str(protocol_path),
        "protocol_sha256": hashlib.sha256(protocol_path.read_bytes()).hexdigest(),
        **assessment,
        "decision": (
            "profile_v2_second_review_release_complete"
            if assessment["all_release_conditions_passed"]
            else "profile_v2_second_review_release_incomplete"
        ),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n")
    _write_markdown(payload, markdown)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--markdown", type=Path, default=DEFAULT_MARKDOWN)
    args = parser.parse_args()
    print(
        run(
            protocol_path=args.protocol,
            output=args.output,
            markdown=args.markdown,
        ).resolve()
    )


if __name__ == "__main__":
    main()
