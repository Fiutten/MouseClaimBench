import json
from pathlib import Path

from mousebrainbench.benchmarks.knowledge_system_release import REQUIREMENTS, run


def _write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload))


def _write_valid_package(root: Path, revision: str = "clean-revision") -> None:
    for relative, (field, expected) in REQUIREMENTS.items():
        payload = {field: expected, "git_revision": revision}
        if "knowledge_system_audit" in relative:
            payload.update(
                {
                    "exact_decision_matches": 40,
                    "explanation_complete_count": 40,
                    "knowledge_profile": {
                        "profile_id": "mouse_brain_claims",
                        "version": "1.1.0",
                        "source_hash": "sha256:" + "0" * 64,
                    },
                }
            )
        _write(root / relative, payload)


def test_kbs_release_does_not_require_an_unclaimed_human_effect(tmp_path: Path) -> None:
    _write_valid_package(tmp_path)

    output = run(
        output=tmp_path / "release.json",
        markdown=tmp_path / "release.md",
        root=tmp_path,
    )
    payload = json.loads(output.read_text())

    assert payload["decision"] == (
        "knowledge_system_method_package_ready_with_declared_boundaries"
    )
    assert payload["human_study_required_for_stated_scope"] is False
    assert payload["knowledge_profile"]["structurally_valid"] is True
    assert payload["knowledge_profile"]["relation_records"] == 22
    assert payload["knowledge_profile"]["independent_expert_validation"] == "not_performed"
    assert len(payload["scientific_claim_boundaries"]) == 5


def test_kbs_release_rejects_dirty_or_unversioned_artifacts(tmp_path: Path) -> None:
    _write_valid_package(tmp_path, revision="revision-dirty")

    output = run(
        output=tmp_path / "release.json",
        markdown=tmp_path / "release.md",
        root=tmp_path,
    )
    payload = json.loads(output.read_text())

    assert payload["decision"] == "knowledge_system_release_requires_action"
    assert payload["dirty_artifacts"]
