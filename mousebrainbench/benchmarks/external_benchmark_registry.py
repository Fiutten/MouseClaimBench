"""Registry of external benchmarks relevant to ClaimBench novelty."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from mousebrainbench import __version__
from mousebrainbench.artifacts import code_revision


DEFAULT_OUTPUT = Path("results/external_benchmark_registry/summary.json")
DEFAULT_MARKDOWN = Path("results/external_benchmark_registry/summary.md")


@dataclass(frozen=True)
class ExternalBenchmark:
    """External benchmark integration target."""

    name: str
    domain: str
    source_url: str
    local_required_path: str
    integration_status: str
    novelty_role: str
    limitation: str


BENCHMARKS = (
    ExternalBenchmark(
        name="SciFact",
        domain="scientific_claim_verification",
        source_url="https://github.com/allenai/scifact",
        local_required_path="data/external/scifact/data/claims_dev.jsonl",
        integration_status="adapter_ready_if_data_available",
        novelty_role="tests cross-domain claim-evidence auditing against expert scientific claims",
        limitation="license and retrieval pipeline must be handled explicitly before redistribution",
    ),
    ExternalBenchmark(
        name="Tuebingen Cause-Effect Pairs",
        domain="causal_direction_benchmark",
        source_url="https://webdav.tuebingen.mpg.de/cause-effect/",
        local_required_path="data/external/tuebingen_cause_effect/pairmeta.txt",
        integration_status="adapter_ready_if_data_available",
        novelty_role="tests directional claim authorization on known cause-effect pairs",
        limitation="ground-truth caveats are explicitly stated by dataset authors",
    ),
    ExternalBenchmark(
        name="Sachs Protein Signaling",
        domain="biological_causal_network",
        source_url="https://www.science.org/doi/10.1126/science.1105809",
        local_required_path="data/external/sachs/README.md",
        integration_status="metadata_target_not_bundled",
        novelty_role="bridges biological causal discovery and mechanistic claim gates",
        limitation="raw data licensing and preprocessing choices must be documented",
    ),
)


def run(
    output: Path = DEFAULT_OUTPUT,
    markdown: Path = DEFAULT_MARKDOWN,
    root: Path = Path("."),
) -> Path:
    """Write external benchmark registry and local availability."""

    rows: list[dict[str, Any]] = []
    available = 0
    for benchmark in BENCHMARKS:
        path = root / benchmark.local_required_path
        exists = path.exists()
        available += int(exists)
        rows.append(
            {
                "name": benchmark.name,
                "domain": benchmark.domain,
                "source_url": benchmark.source_url,
                "local_required_path": benchmark.local_required_path,
                "local_available": exists,
                "integration_status": (
                    "locally_available" if exists else benchmark.integration_status
                ),
                "novelty_role": benchmark.novelty_role,
                "limitation": benchmark.limitation,
            }
        )
    payload = {
        "version": __version__,
        "git_revision": code_revision(),
        "analysis": "external_benchmark_registry",
        "registered_benchmarks": len(BENCHMARKS),
        "locally_available_benchmarks": available,
        "benchmarks": rows,
        "decision": (
            "external_benchmarks_registered_with_pending_data"
            if available < len(BENCHMARKS)
            else "external_benchmarks_available"
        ),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2))
    write_markdown(payload, markdown)
    return output


def write_markdown(payload: dict[str, Any], markdown: Path) -> None:
    """Write external benchmark registry."""

    lines = [
        "# External Benchmark Registry",
        "",
        f"- Decision: `{payload['decision']}`",
        f"- Registered: `{payload['registered_benchmarks']}`",
        f"- Locally available: `{payload['locally_available_benchmarks']}`",
        "",
        "| Benchmark | Domain | Local available | Novelty role |",
        "|---|---|---:|---|",
    ]
    for row in payload["benchmarks"]:
        lines.append(
            f"| `{row['name']}` | `{row['domain']}` | `{row['local_available']}` | "
            f"{row['novelty_role']} |"
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
