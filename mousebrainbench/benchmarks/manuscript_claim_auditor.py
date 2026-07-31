"""Audit manuscript wording against executable claim contracts."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

from mousebrainbench import __version__
from mousebrainbench.artifacts import code_revision
from mousebrainbench.claimdsl import audit_claim_specs, load_claim_specs


DEFAULT_CLAIMS = Path("configs/claims/mousebrainbench_claims.yaml")
DEFAULT_OUTPUT = Path("results/manuscript_claim_audit/summary.json")
DEFAULT_MARKDOWN = Path("results/manuscript_claim_audit/summary.md")
DEFAULT_MANUSCRIPT_GLOBS = (
    "paper/main.tex",
    "paper/README.md",
    "README.md",
)
FALLBACK_MANUSCRIPT_GLOBS = (
    "paper/main_anonymous.tex",
    "paper/sections/*.tex",
    "paper/tables/*.tex",
)
NEGATING_CLAIM_CONTEXT = (
    r"\b(not|no|without|does\s+not|do\s+not|cannot|not\s+a|must\s+not|"
    r"blocks?|blocking|rejects?|rejecting|downgrades?|downgraded|downgrading|"
    r"not\s+validate|does\s+not\s+validate|not\s+interpreted|not\s+promoted|"
    r"cannot\s+promote|not\s+allowed)\b"
)

RISK_PATTERNS: tuple[dict[str, str], ...] = (
    {
        "risk_id": "complete_mouse_brain_twin",
        "severity": "high",
        "claim_family": "digital_twin",
        "pattern": r"\b(complete|full|whole[- ]brain)\s+(mouse[- ]brain\s+)?digital\s+twin\b",
        "negation_window": NEGATING_CLAIM_CONTEXT,
        "reason": "Complete or whole-brain digital-twin wording requires evidence not present here.",
    },
    {
        "risk_id": "causal_mechanism_from_observation",
        "severity": "high",
        "claim_family": "causal",
        "pattern": (
            r"\b(causal\s+mechanism|causal\s+evidence|causal\s+effect|"
            r"(establishes?|validates?|proves?|supports?)\s+causality|"
            r"(establishes?|validates?|proves?|supports?)\s+causation|"
            r"mechanistic\s+cause)\b"
        ),
        "negation_window": rf"{NEGATING_CLAIM_CONTEXT}|non[- ]causal|no\s+es",
        "reason": "Causal wording requires interventional or causal-identification evidence.",
    },
    {
        "risk_id": "sota_without_comparable_baseline",
        "severity": "high",
        "claim_family": "benchmark_performance",
        "pattern": r"\b(state[- ]of[- ]the[- ]art|SOTA|outperforms\s+all|best[- ]performing)\b",
        "negation_window": NEGATING_CLAIM_CONTEXT,
        "reason": "SOTA wording requires comparable official baselines and matched protocols.",
    },
    {
        "risk_id": "universal_scientific_verification",
        "severity": "high",
        "claim_family": "claim_verification",
        "pattern": r"\b(verifies\s+all|universally\s+verifies|proves\s+all)\s+scientific\s+claims\b",
        "negation_window": r"\b(not|no|without|does\s+not|do\s+not|cannot)\b",
        "reason": "Universal scientific-claim verification is not supported by bounded benchmarks.",
    },
    {
        "risk_id": "reviewer_proof_wording",
        "severity": "medium",
        "claim_family": "reviewer_defense",
        "pattern": r"\b(no\s+limitations|reviewer\s+criticism\s+is\s+impossible|cannot\s+be\s+criticized)\b",
        "negation_window": r"\b(not|never|do\s+not|does\s+not)\b",
        "reason": "Reviewer attack suites expose known risks; they do not eliminate criticism.",
    },
)


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower()).strip()


def _resolve_manuscripts(root: Path, paths: tuple[Path, ...] | None) -> tuple[Path, ...]:
    """Resolve explicit manuscript files or the default paper source set."""

    if paths:
        return tuple(root / path for path in paths)
    resolved: list[Path] = []
    for pattern in DEFAULT_MANUSCRIPT_GLOBS:
        matches = sorted(root.glob(pattern))
        resolved.extend(path for path in matches if path.is_file())
    if not any(path.name == "main.tex" for path in resolved):
        for pattern in FALLBACK_MANUSCRIPT_GLOBS:
            matches = sorted(root.glob(pattern))
            resolved.extend(path for path in matches if path.is_file())
    return tuple(dict.fromkeys(resolved))


def _read_texts(paths: tuple[Path, ...]) -> tuple[str, list[str]]:
    chunks = []
    existing = []
    for path in paths:
        if path.exists():
            existing.append(str(path))
            chunks.append(path.read_text(errors="ignore"))
    return "\n".join(chunks), existing


def _is_negated(text: str, start: int, negation_pattern: str) -> bool:
    """Return true when a risky phrase is locally negated.

    The check is intentionally conservative and transparent. It only exempts
    risk patterns when a negation marker appears immediately before the match,
    which avoids blocking legitimate limitation statements such as "not a
    whole-brain digital twin".
    """

    prefix = text[max(0, start - 180) : start]
    return bool(re.search(negation_pattern, prefix, flags=re.IGNORECASE))


def _pattern_hits(text: str) -> list[dict[str, Any]]:
    """Detect risky manuscript claims beyond exact blocked-wording matches."""

    hits: list[dict[str, Any]] = []
    for item in RISK_PATTERNS:
        for match in re.finditer(item["pattern"], text, flags=re.IGNORECASE):
            negated = _is_negated(text, match.start(), item["negation_window"])
            hits.append(
                {
                    "risk_id": item["risk_id"],
                    "severity": item["severity"],
                    "claim_family": item["claim_family"],
                    "matched_text": match.group(0),
                    "negated": negated,
                    "reason": item["reason"],
                }
            )
    return hits


def run(
    claims: Path = DEFAULT_CLAIMS,
    manuscript: tuple[Path, ...] | None = None,
    output: Path = DEFAULT_OUTPUT,
    markdown: Path = DEFAULT_MARKDOWN,
    root: Path = Path("."),
) -> Path:
    """Audit claim contracts and manuscript wording."""

    specs = load_claim_specs(root / claims)
    audit_results = audit_claim_specs(specs, root=root)
    manuscript_paths = _resolve_manuscripts(root, manuscript)
    raw_text, existing_manuscripts = _read_texts(manuscript_paths)
    text = _normalize(raw_text)
    pattern_hits = _pattern_hits(raw_text)
    active_pattern_hits = [hit for hit in pattern_hits if not hit["negated"]]
    rows: list[dict[str, Any]] = []
    blocked_hits: list[dict[str, str]] = []
    for spec, result in zip(specs, audit_results, strict=True):
        permitted_present = _normalize(spec.permitted_wording) in text
        claim_blocked_hits = [
            wording for wording in spec.blocked_wording if _normalize(wording) in text
        ]
        pattern_blocked_hits = [
            pattern
            for pattern in spec.blocked_patterns
            if re.search(pattern, raw_text, flags=re.IGNORECASE)
        ]
        for wording in claim_blocked_hits:
            blocked_hits.append({"claim_id": spec.claim_id, "blocked_wording": wording})
        for pattern in pattern_blocked_hits:
            blocked_hits.append({"claim_id": spec.claim_id, "blocked_wording": pattern})
        rows.append(
            {
                "claim_id": spec.claim_id,
                "claim_type": spec.claim_type,
                "evidence_family": spec.evidence_family,
                "severity": spec.severity,
                "scope": spec.scope,
                "artifact_status": result.status,
                "permitted_wording_present": permitted_present,
                "blocked_wording_hits": claim_blocked_hits,
                "blocked_pattern_hits": pattern_blocked_hits,
                "non_compensatory_requirements": list(spec.non_compensatory_requirements),
                "missing_artifacts": list(result.missing_artifacts),
                "failed_expectations": list(result.failed_expectations),
            }
        )
    unsupported_present = [
        row["claim_id"]
        for row in rows
        if row["permitted_wording_present"] and row["artifact_status"] != "supported"
    ]
    payload = {
        "version": __version__,
        "git_revision": code_revision(),
        "analysis": "manuscript_claim_audit",
        "claims_file": str(claims),
        "manuscript_inputs": [
            str(path.relative_to(root)) if path.is_relative_to(root) else str(path)
            for path in manuscript_paths
        ],
        "existing_manuscript_inputs": existing_manuscripts,
        "manuscript_character_count": len(raw_text),
        "rows": rows,
        "blocked_wording_hits": blocked_hits,
        "risk_pattern_hits": pattern_hits,
        "active_risk_pattern_hits": active_pattern_hits,
        "unsupported_present_claims": unsupported_present,
        "decision": (
            "manuscript_claim_audit_passed"
            if not blocked_hits and not unsupported_present and not active_pattern_hits
            else "manuscript_claim_audit_blocks_release"
        ),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2))
    write_markdown(payload, markdown)
    return output


def write_markdown(payload: dict[str, Any], markdown: Path) -> None:
    """Write claim-audit report."""

    lines = [
        "# Manuscript Claim Audit",
        "",
        f"- Decision: `{payload['decision']}`",
        f"- Claims audited: `{len(payload['rows'])}`",
        f"- Manuscript inputs found: `{len(payload.get('existing_manuscript_inputs', []))}`",
        f"- Manuscript characters: `{payload.get('manuscript_character_count', 0)}`",
        f"- Blocked wording hits: `{len(payload['blocked_wording_hits'])}`",
        f"- Active risk-pattern hits: `{len(payload.get('active_risk_pattern_hits', []))}`",
        "",
        "| Claim | Type | Severity | Artifact status | Permitted wording present | Blocked hits |",
        "|---|---|---|---|---:|---:|",
    ]
    for row in payload["rows"]:
        lines.append(
            f"| `{row['claim_id']}` | `{row['claim_type']}` | `{row['severity']}` | "
            f"`{row['artifact_status']}` | `{row['permitted_wording_present']}` | "
            f"`{len(row['blocked_wording_hits']) + len(row.get('blocked_pattern_hits', []))}` |"
        )
    active_hits = payload.get("active_risk_pattern_hits", [])
    if active_hits:
        lines.extend(["", "## Active Risk Patterns", ""])
        lines.append("| Risk | Severity | Match | Reason |")
        lines.append("|---|---|---|---|")
        for hit in active_hits:
            lines.append(
                f"| `{hit['risk_id']}` | `{hit['severity']}` | "
                f"`{hit['matched_text']}` | {hit['reason']} |"
            )
    lines.append("")
    markdown.parent.mkdir(parents=True, exist_ok=True)
    markdown.write_text("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--claims", type=Path, default=DEFAULT_CLAIMS)
    parser.add_argument("--manuscript", type=Path, action="append")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--markdown", type=Path, default=DEFAULT_MARKDOWN)
    parser.add_argument("--root", type=Path, default=Path("."))
    args = parser.parse_args()
    manuscripts = tuple(args.manuscript) if args.manuscript else None
    print(
        json.dumps(
            {
                "output": str(
                    run(args.claims, manuscripts, args.output, args.markdown, args.root).resolve()
                )
            }
        )
    )


if __name__ == "__main__":
    main()
