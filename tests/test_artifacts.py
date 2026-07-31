from __future__ import annotations

import pytest

from mousebrainbench import artifacts
from mousebrainbench.artifacts import code_revision


def test_code_revision_accepts_valid_publication_override(monkeypatch: pytest.MonkeyPatch) -> None:
    revision = "abcdef1234567890abcdef1234567890abcdef12"
    monkeypatch.setenv("MOUSEBRAINBENCH_GIT_REVISION", revision)
    monkeypatch.setattr(
        artifacts.subprocess,
        "check_output",
        lambda *args, **kwargs: f"{revision}\n",
    )

    assert code_revision() == revision


def test_code_revision_rejects_non_git_publication_override(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("MOUSEBRAINBENCH_GIT_REVISION", "main-dirty")

    with pytest.raises(ValueError, match="hexadecimal Git ID"):
        code_revision()


def test_code_revision_rejects_override_that_is_not_head(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    override = "abcdef1234567890abcdef1234567890abcdef12"
    head = "1234567890abcdef1234567890abcdef12345678"
    monkeypatch.setenv("MOUSEBRAINBENCH_GIT_REVISION", override)
    responses = iter((f"{override}\n", f"{head}\n"))
    monkeypatch.setattr(
        artifacts.subprocess,
        "check_output",
        lambda *args, **kwargs: next(responses),
    )

    with pytest.raises(ValueError, match="checked-out HEAD"):
        code_revision()
