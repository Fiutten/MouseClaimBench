"""Independent Answer Set Programming backend for claim decisions.

The Python rule engine remains authoritative. This module translates one
immutable knowledge profile and one evidence assignment into an ASP program
and asks Potassco clingo for its unique claim disposition. Keeping this backend
independent makes semantic equivalence executable instead of merely asserted.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from mousebrainbench.knowledge.profile import KnowledgeProfile
from mousebrainbench.validation.evidence_contract import (
    DecisionStatus,
    EvidenceBlock,
    EvidenceStatus,
)


@dataclass(frozen=True)
class AspDecision:
    """One clingo decision and the reproducibility metadata behind it."""

    claim: str
    status: DecisionStatus
    backend: str
    program_hash: str
    model_count: int
    shown_atoms: tuple[str, ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "claim": self.claim,
            "status": self.status.value,
            "backend": self.backend,
            "program_hash": self.program_hash,
            "model_count": self.model_count,
            "shown_atoms": list(self.shown_atoms),
        }


def _clingo_module():
    try:
        import clingo
    except ImportError as exc:
        raise RuntimeError(
            "ASP inference requires the `semantic-risk-v3` optional dependencies"
        ) from exc
    return clingo


def _quoted(value: str) -> str:
    """Return a clingo-compatible quoted string without identifier assumptions."""

    return json.dumps(value, ensure_ascii=True)


def build_asp_program(
    profile: KnowledgeProfile,
    claim: str,
    evidence_blocks: Mapping[str, EvidenceBlock],
) -> str:
    """Compile one claim and its complete required evidence assignment to ASP."""

    requirement = profile.requirement(claim)
    if requirement is None:
        return (
            f"decision({_quoted(claim)},\"out_of_scope\").\n"
            "#show decision/2.\n"
        )

    lines = [f"claim({_quoted(claim)})."]
    for block_name in requirement.required_blocks:
        block = evidence_blocks.get(block_name)
        status = block.status if block is not None else EvidenceStatus.UNKNOWN
        lines.append(f"required({_quoted(claim)},{_quoted(block_name)}).")
        lines.append(f"status({_quoted(block_name)},{_quoted(status.value)}).")

    lines.extend(
        (
            'blocked(C) :- claim(C), required(C,B), status(B,"failed").',
            'review(C) :- claim(C), not blocked(C), required(C,B), status(B,"requires_review").',
            'uncertain(C) :- claim(C), not blocked(C), not review(C), required(C,B), status(B,"unknown").',
            'outside(C) :- claim(C), not blocked(C), not review(C), not uncertain(C), required(C,B), status(B,"not_applicable").',
            'not_all_passed(C) :- required(C,B), not status(B,"passed").',
            'supported(C) :- claim(C), not not_all_passed(C).',
            'decision(C,"blocked") :- blocked(C).',
            'decision(C,"needs_external_review") :- review(C).',
            'decision(C,"uncertain") :- uncertain(C).',
            'decision(C,"out_of_scope") :- outside(C).',
            'decision(C,"supported") :- supported(C).',
            '#show decision/2.',
        )
    )
    return "\n".join(lines) + "\n"


def infer_with_clingo(
    profile: KnowledgeProfile,
    claim: str,
    evidence_blocks: Mapping[str, EvidenceBlock],
) -> AspDecision:
    """Execute the independent ASP semantics and require one unique decision."""

    clingo = _clingo_module()
    program = build_asp_program(profile, claim, evidence_blocks)
    control = clingo.Control(["--models=0", "--warn=none"])
    control.add("base", [], program)
    control.ground([("base", [])])
    models: list[tuple[str, ...]] = []
    symbols: list[Any] = []
    with control.solve(yield_=True) as handle:
        for model in handle:
            shown = tuple(sorted(str(symbol) for symbol in model.symbols(shown=True)))
            models.append(shown)
            symbols.extend(model.symbols(shown=True))

    if len(models) != 1:
        raise RuntimeError(f"ASP claim program produced {len(models)} stable models")
    decisions = [symbol for symbol in symbols if symbol.name == "decision"]
    if len(decisions) != 1:
        raise RuntimeError(f"ASP claim program produced {len(decisions)} decisions")
    decided_claim = decisions[0].arguments[0].string
    if decided_claim != claim:
        raise RuntimeError("ASP decision returned a different claim identifier")
    return AspDecision(
        claim=claim,
        status=DecisionStatus(decisions[0].arguments[1].string),
        backend=f"clingo-{clingo.__version__}",
        program_hash=f"sha256:{hashlib.sha256(program.encode()).hexdigest()}",
        model_count=len(models),
        shown_atoms=models[0],
    )
