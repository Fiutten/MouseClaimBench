"""LLM-assisted claim extraction audit with deterministic fallback.

This module adds the LLM layer without making the LLM authoritative.  The
default path is deterministic and reproducible: it extracts candidate claims
from manuscript text using transparent patterns, classifies them into the same
claim families used by ClaimBench, and records conservative wording suggestions.

An external LLM can later replace only the candidate-extraction step.  The final
authorization remains controlled by executable ClaimBench artifacts and claim
contracts, never by the LLM output.
"""

from __future__ import annotations

import argparse
import json
import os
import re
from collections import Counter
from pathlib import Path
from typing import Any

from mousebrainbench import __version__
from mousebrainbench.artifacts import code_revision
from mousebrainbench.benchmarks.manuscript_claim_auditor import (
    _read_texts,
    _resolve_manuscripts,
)


DEFAULT_OUTPUT = Path("results/llm_claim_extraction_audit/summary.json")
DEFAULT_MARKDOWN = Path("results/llm_claim_extraction_audit/summary.md")

CLAIM_PATTERNS: tuple[tuple[str, str], ...] = (
    ("digital_twin", r"\b(digital\s+twin|whole[- ]brain|complete\s+brain)\b"),
    ("causal", r"\b(causal|causality|causation|intervention|perturbation)\b"),
    ("mechanistic", r"\b(mechanistic|mechanism|identifiability|topology|direction)\b"),
    ("structure_function", r"\b(structure[- ]function|MICRONS|readout[- ]location)\b"),
    ("prediction", r"\b(predict|prediction|predictive|correlation|Sensorium)\b"),
    ("reproducibility", r"\b(reproducib|hold[- ]out|bootstrap|stability|release)\b"),
    ("sota", r"\b(state[- ]of[- ]the[- ]art|SOTA|leaderboard|best[- ]performing)\b"),
    ("generalization", r"\b(generaliz|out[- ]of[- ]distribution|OOD|external)\b"),
)

STRONG_VERBS = re.compile(
    r"\b(proves?|demonstrates?|establishes?|validates?|achieves?|outperforms?|"
    r"confirms?|shows?)\b",
    flags=re.IGNORECASE,
)
NEGATION = re.compile(
    r"\b(not|no|without|does\s+not|do\s+not|cannot|must\s+not|blocks?|rejects?)\b",
    flags=re.IGNORECASE,
)


def _sentences(text: str) -> list[str]:
    """Split manuscript text into compact candidate sentences."""

    cleaned = re.sub(r"%.*", " ", text)
    cleaned = re.sub(r"\\[a-zA-Z]+\*?(?:\[[^\]]*\])?(?:\{[^{}]*\})?", " ", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned)
    parts = re.split(r"(?<=[.!?])\s+", cleaned)
    return [part.strip() for part in parts if 40 <= len(part.strip()) <= 600]


def _claim_type(sentence: str) -> str | None:
    for claim_type, pattern in CLAIM_PATTERNS:
        if re.search(pattern, sentence, flags=re.IGNORECASE):
            return claim_type
    return None


def _is_claim_like(sentence: str) -> bool:
    return bool(STRONG_VERBS.search(sentence)) or any(
        re.search(pattern, sentence, flags=re.IGNORECASE) for _claim_type, pattern in CLAIM_PATTERNS
    )


def _is_negated(sentence: str) -> bool:
    return bool(NEGATION.search(sentence))


def _suggestion(claim_type: str, negated: bool) -> str:
    if negated:
        return "Keep as limitation/boundary wording; do not convert into a positive claim."
    suggestions = {
        "digital_twin": "Downgrade to partial digital-model or claim-audit wording unless whole-brain causal evidence exists.",
        "causal": "Use causal wording only with intervention or causal-identification evidence.",
        "mechanistic": "Require prediction, reproducibility, topology specificity, and direction before mechanistic wording.",
        "structure_function": "Keep as local observational structure-function association unless causal evidence exists.",
        "prediction": "Report predictive performance without promoting it to mechanism or causality.",
        "reproducibility": "Report reproducibility as stability evidence, not as mechanism by itself.",
        "sota": "Use SOTA wording only with comparable official baselines and matched protocols.",
        "generalization": "Restrict generalization wording to the evaluated dataset, animal, session, or domain.",
    }
    return suggestions.get(claim_type, "Map the claim to an executable evidence contract.")


def _load_optional_llm_candidates(root: Path) -> list[dict[str, Any]]:
    """Load optional LLM-extracted candidates from a local JSON file.

    The environment hook is intentionally file-based.  It lets a researcher run
    any LLM outside this package and then audit the resulting candidates without
    making the package depend on API credentials, model versions, or network
    calls.
    """

    path = os.environ.get("MOUSEBRAINBENCH_LLM_CLAIM_EXTRACTOR_JSON")
    if not path:
        return []
    candidate_path = Path(path)
    if not candidate_path.is_absolute():
        candidate_path = root / candidate_path
    if not candidate_path.exists():
        return []
    payload = json.loads(candidate_path.read_text())
    if isinstance(payload, list):
        return [dict(item) for item in payload]
    return [dict(item) for item in payload.get("candidates", [])]


def _deterministic_candidates(text: str, limit: int = 120) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    seen: set[str] = set()
    for sentence in _sentences(text):
        normalized = sentence.lower()
        if normalized in seen or not _is_claim_like(sentence):
            continue
        seen.add(normalized)
        claim_type = _claim_type(sentence)
        if claim_type is None:
            continue
        negated = _is_negated(sentence)
        candidates.append(
            {
                "source": "deterministic_extractor",
                "text": sentence,
                "claim_type": claim_type,
                "negated_or_boundary_context": negated,
                "llm_authoritative": False,
                "suggested_conservative_wording": _suggestion(claim_type, negated),
            }
        )
        if len(candidates) >= limit:
            break
    return candidates


def run(
    output: Path = DEFAULT_OUTPUT,
    markdown: Path = DEFAULT_MARKDOWN,
    root: Path = Path("."),
    manuscript: tuple[Path, ...] | None = None,
) -> Path:
    """Extract candidate claims and audit the LLM layer boundaries."""

    manuscript_paths = _resolve_manuscripts(root, manuscript)
    text, existing = _read_texts(manuscript_paths)
    deterministic = _deterministic_candidates(text)
    optional_llm = _load_optional_llm_candidates(root)
    candidates = [*deterministic, *optional_llm]
    by_type = Counter(str(item.get("claim_type", "unknown")) for item in candidates)
    authoritative_candidates = [
        item for item in candidates if bool(item.get("llm_authoritative", False))
    ]
    unsupported_positive_suggestions = [
        item
        for item in candidates
        if item.get("claim_type") in {"digital_twin", "causal", "sota"}
        and not bool(item.get("negated_or_boundary_context", False))
        and "Downgrade" not in str(item.get("suggested_conservative_wording", ""))
        and "only with" not in str(item.get("suggested_conservative_wording", ""))
    ]
    payload = {
        "version": __version__,
        "git_revision": code_revision(),
        "analysis": "llm_claim_extraction_audit",
        "mode": "deterministic_fallback_with_optional_llm_file",
        "llm_api_called": False,
        "llm_authoritative": False,
        "optional_llm_candidates_loaded": len(optional_llm),
        "manuscript_inputs": [
            str(path.relative_to(root)) if path.is_relative_to(root) else str(path)
            for path in manuscript_paths
        ],
        "existing_manuscript_inputs": existing,
        "num_candidates": len(candidates),
        "num_deterministic_candidates": len(deterministic),
        "candidate_counts_by_type": dict(by_type),
        "authoritative_llm_candidates": len(authoritative_candidates),
        "unsupported_positive_suggestions": len(unsupported_positive_suggestions),
        "candidates": candidates,
        "decision": (
            "llm_claim_extraction_layer_ready_non_authoritative"
            if candidates and not authoritative_candidates and not unsupported_positive_suggestions
            else "llm_claim_extraction_layer_requires_revision"
        ),
        "interpretation": (
            "The LLM layer is limited to candidate extraction and conservative wording support. "
            "Claim authorization remains controlled by executable ClaimBench artifacts."
        ),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2))
    write_markdown(payload, markdown)
    return output


def write_markdown(payload: dict[str, Any], markdown: Path) -> None:
    """Write the LLM claim-extraction audit report."""

    lines = [
        "# LLM Claim Extraction Audit",
        "",
        f"- Decision: `{payload['decision']}`",
        f"- Mode: `{payload['mode']}`",
        f"- LLM API called: `{payload['llm_api_called']}`",
        f"- LLM authoritative: `{payload['llm_authoritative']}`",
        f"- Candidates: `{payload['num_candidates']}`",
        f"- Candidate counts by type: `{payload['candidate_counts_by_type']}`",
        "",
        "## Boundary",
        "",
        str(payload["interpretation"]),
        "",
    ]
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
