"""Release readiness check for the claim-aware MouseBrainBench extension."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from mousebrainbench import __version__
from mousebrainbench.artifacts import code_revision


DEFAULT_OUTPUT = Path("results/claimbench_release/summary.json")
DEFAULT_MARKDOWN = Path("results/claimbench_release/summary.md")

REQUIRED_ARTIFACTS = (
    Path("docs/CLAIMBENCH_SOTA_AND_VIABILITY.md"),
    Path("results/claim_adversarial_benchmark/summary.json"),
    Path("results/real_case_claim_matrix/summary.json"),
    Path("results/claim_ledger/summary.json"),
    Path("results/microns_primary_robustness/summary.json"),
    Path("results/mis2_synthetic_calibration/summary.json"),
    Path("results/mis2_threshold_sensitivity/summary.json"),
)


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def run(output: Path = DEFAULT_OUTPUT, markdown: Path = DEFAULT_MARKDOWN, root: Path = Path(".")) -> Path:
    """Check whether the claim-aware extension has the required frozen artifacts."""

    artifact_rows: list[dict[str, Any]] = []
    missing = []
    dirty = []
    for relative in REQUIRED_ARTIFACTS:
        path = root / relative
        exists = path.exists()
        revision = None
        decision = None
        if exists and path.suffix == ".json":
            data = _load_json(path)
            revision = data.get("git_revision")
            decision = data.get("decision")
            if isinstance(revision, str) and revision.endswith("-dirty"):
                dirty.append(str(relative))
        if not exists:
            missing.append(str(relative))
        artifact_rows.append(
            {
                "artifact": str(relative),
                "exists": exists,
                "git_revision": revision,
                "decision": decision,
            }
        )

    decision = (
        "claimbench_release_ready"
        if not missing and not dirty
        else "claimbench_release_requires_cleanup"
    )
    payload = {
        "version": __version__,
        "git_revision": code_revision(),
        "analysis": "claimbench_release",
        "decision": decision,
        "missing_artifacts": missing,
        "dirty_artifacts": dirty,
        "required_artifacts": artifact_rows,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2))
    write_markdown(payload, markdown)
    return output


def write_markdown(payload: dict[str, Any], markdown: Path) -> None:
    """Write release readiness report."""

    lines = [
        "# ClaimBench Release Check",
        "",
        f"- Decision: `{payload['decision']}`",
        f"- Missing artifacts: `{len(payload['missing_artifacts'])}`",
        f"- Dirty artifacts: `{len(payload['dirty_artifacts'])}`",
        "",
        "| Artifact | Exists | Git revision | Decision |",
        "|---|---:|---|---|",
    ]
    for row in payload["required_artifacts"]:
        lines.append(
            f"| `{row['artifact']}` | `{row['exists']}` | "
            f"`{row['git_revision']}` | `{row['decision']}` |"
        )
    lines.append("")
    markdown.parent.mkdir(parents=True, exist_ok=True)
    markdown.write_text("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--markdown", type=Path, default=DEFAULT_MARKDOWN)
    parser.add_argument("--root", type=Path, default=Path("."))
    args = parser.parse_args()
    print(json.dumps({"output": str(run(args.output, args.markdown, args.root).resolve())}))


if __name__ == "__main__":
    main()
