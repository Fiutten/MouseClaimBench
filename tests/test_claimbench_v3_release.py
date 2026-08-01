import json
from pathlib import Path

from mousebrainbench.benchmarks.claimbench_v3_release import REQUIREMENTS, run


def _write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload))


def test_v3_release_is_ready_but_retains_human_and_biological_limits(tmp_path) -> None:
    for relative, (field, expected) in REQUIREMENTS.items():
        payload = {field: expected, "git_revision": "clean-revision"}
        if "human_evaluation_protocol" in relative:
            payload.update({"study_status": "not_executed", "results_available": False})
        _write(tmp_path / relative, payload)

    output = run(
        output=tmp_path / "release.json",
        markdown=tmp_path / "release.md",
        root=tmp_path,
    )
    payload = json.loads(output.read_text())

    assert payload["decision"] == "methodological_package_ready_human_effect_unvalidated"
    assert payload["human_study_executed"] is False
    assert len(payload["scientific_claim_blockers"]) == 3


def test_v3_release_rejects_dirty_artifact(tmp_path) -> None:
    for relative, (field, expected) in REQUIREMENTS.items():
        payload = {field: expected, "git_revision": "revision-dirty"}
        if "human_evaluation_protocol" in relative:
            payload.update({"study_status": "not_executed", "results_available": False})
        _write(tmp_path / relative, payload)

    output = run(
        output=tmp_path / "release.json",
        markdown=tmp_path / "release.md",
        root=tmp_path,
    )
    payload = json.loads(output.read_text())

    assert payload["decision"] == "claimbench_v3_release_requires_action"
    assert payload["dirty_artifacts"]
