"""Independent ASP execution of the hardened profile-authorization semantics."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from mousebrainbench.knowledge.authorization import (
    ClaimAuthorizationProfile,
    ProfileAuthorizationStatus,
)
from mousebrainbench.validation.evidence_contract import EvidenceBlock, EvidenceStatus


@dataclass(frozen=True)
class AspProfileAuthorizationDecision:
    """One unique ASP authorization and its complete deficit set."""

    claim: str
    status: ProfileAuthorizationStatus
    deficits: tuple[tuple[str, EvidenceStatus], ...]
    backend: str
    program_hash: str
    model_count: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "claim": self.claim,
            "status": self.status.value,
            "authorized": self.status is ProfileAuthorizationStatus.AUTHORIZED,
            "deficits": [
                {"name": name, "effective_status": status.value}
                for name, status in self.deficits
            ],
            "backend": self.backend,
            "program_hash": self.program_hash,
            "model_count": self.model_count,
        }


def _clingo_module():
    try:
        import clingo
    except ImportError as exc:
        raise RuntimeError(
            "ASP authorization requires the `semantic-risk-v3` optional dependencies"
        ) from exc
    return clingo


def _quoted(value: str) -> str:
    return json.dumps(value, ensure_ascii=True)


def _present(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, (tuple, list, dict, set)):
        return bool(value)
    return True


def _effective_status(
    profile: ClaimAuthorizationProfile,
    block_name: str,
    block: EvidenceBlock | None,
) -> EvidenceStatus:
    if block is None:
        return EvidenceStatus.UNKNOWN
    if block.status is not EvidenceStatus.PASSED:
        return block.status
    specification = profile.block_specification(block_name)
    observations = dict(block.observations)
    complete = all(
        field in observations and _present(observations[field])
        for field in specification.required_observations_when_passed
    )
    complete = complete and all(
        _present(value) for value in (block.source, block.rule, block.rationale)
    )
    return EvidenceStatus.PASSED if complete else EvidenceStatus.REQUIRES_REVIEW


def build_authorization_asp_program(
    profile: ClaimAuthorizationProfile,
    claim: str,
    evidence_blocks: Mapping[str, EvidenceBlock],
) -> str:
    """Compile one authorization case into a complete finite ASP program."""

    requirement = profile.requirement(claim)
    if requirement is None:
        return (
            f"decision({_quoted(claim)},\"outside_profile\").\n"
            "#show decision/2.\n"
        )
    lines = [f"claim({_quoted(claim)})."]
    for block_name in requirement.required_blocks:
        status = _effective_status(profile, block_name, evidence_blocks.get(block_name))
        lines.append(f"required({_quoted(claim)},{_quoted(block_name)}).")
        lines.append(f"effective({_quoted(block_name)},{_quoted(status.value)}).")
    lines.extend(
        (
            'deficit(C,B,S) :- required(C,B), effective(B,S), S != "passed".',
            "has_deficit(C) :- deficit(C,_,_).",
            'decision(C,"profile_not_authorized") :- claim(C), has_deficit(C).',
            'decision(C,"profile_authorized") :- claim(C), not has_deficit(C).',
            "#show decision/2.",
            "#show deficit/3.",
        )
    )
    return "\n".join(lines) + "\n"


def authorize_with_clingo_v2(
    profile: ClaimAuthorizationProfile,
    claim: str,
    evidence_blocks: Mapping[str, EvidenceBlock],
) -> AspProfileAuthorizationDecision:
    """Execute the ASP semantics and require one stable model and decision."""

    clingo = _clingo_module()
    program = build_authorization_asp_program(profile, claim, evidence_blocks)
    control = clingo.Control(["--models=0", "--warn=none"])
    control.add("base", [], program)
    control.ground([("base", [])])
    models: list[tuple[Any, ...]] = []
    with control.solve(yield_=True) as handle:
        for model in handle:
            models.append(tuple(model.symbols(shown=True)))
    if len(models) != 1:
        raise RuntimeError(f"ASP authorization produced {len(models)} stable models")
    decisions = [symbol for symbol in models[0] if symbol.name == "decision"]
    if len(decisions) != 1:
        raise RuntimeError(f"ASP authorization produced {len(decisions)} decisions")
    decided_claim = decisions[0].arguments[0].string
    if decided_claim != claim:
        raise RuntimeError("ASP authorization returned a different claim identifier")
    deficits = tuple(
        sorted(
            (
                (symbol.arguments[1].string, EvidenceStatus(symbol.arguments[2].string))
                for symbol in models[0]
                if symbol.name == "deficit"
            ),
            key=lambda item: item[0],
        )
    )
    return AspProfileAuthorizationDecision(
        claim=claim,
        status=ProfileAuthorizationStatus(decisions[0].arguments[1].string),
        deficits=deficits,
        backend=f"clingo-{clingo.__version__}",
        program_hash=f"sha256:{hashlib.sha256(program.encode()).hexdigest()}",
        model_count=len(models),
    )
