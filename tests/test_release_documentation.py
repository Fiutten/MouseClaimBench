from pathlib import Path

ROOT = Path(__file__).parents[1]


def test_release_readme_installs_every_verification_lock() -> None:
    """Keep the documented clean-room setup aligned with the verify script."""

    readme = (ROOT / "README.md").read_text(encoding="utf-8")

    assert "requirements-semantic-risk-v3-lock.txt" in readme
    assert "requirements-standards-lock.txt" in readme
    assert ".[hybrid-validation]" in readme
    assert "reproduce_standards_prospective_v3.sh verify" in readme
