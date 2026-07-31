import json

from mousebrainbench.benchmarks import sensorium_official_baseline_audit as official_audit


class _AvailableProbe:
    package = "toy"
    purpose = "test"

    def as_dict(self):
        return {"package": self.package, "purpose": self.purpose, "available": True}


def test_official_smoke_does_not_count_as_trained_baseline(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(official_audit, "OFFICIAL_STACK", (_AvailableProbe(),))
    official_repo = tmp_path / "sensorium_2023"
    (official_repo / "sensorium").mkdir(parents=True)
    (official_repo / "env.yml").write_text("name: sensorium\n")
    (official_repo / "pyproject.toml").write_text("[project]\nname='sensorium_2023'\n")
    smoke = tmp_path / "smoke.json"
    smoke.write_text(json.dumps({"official_stack_forward_ok": True}))
    mlp = tmp_path / "mlp.json"
    mlp.write_text("{}")

    payload = official_audit.audit(
        official_repo=official_repo,
        official_smoke=smoke,
        official_trained_summary=tmp_path / "missing_trained.json",
        local_mlp_summary=mlp,
    )

    assert payload["official_stack_forward_ok"]
    assert not payload["official_trained_baseline_available"]
    assert not payload["official_baseline_viable"]
    assert payload["decision"] == "official_sensorium_stack_integrated_training_pending"


def test_bounded_trained_baseline_available_but_not_q1_qualified(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setattr(official_audit, "OFFICIAL_STACK", (_AvailableProbe(),))
    official_repo = tmp_path / "sensorium_2023"
    (official_repo / "sensorium").mkdir(parents=True)
    (official_repo / "env.yml").write_text("name: sensorium\n")
    (official_repo / "pyproject.toml").write_text("[project]\nname='sensorium_2023'\n")
    smoke = tmp_path / "smoke.json"
    smoke.write_text(json.dumps({"official_stack_forward_ok": True}))
    trained = tmp_path / "trained.json"
    trained.write_text(
        json.dumps(
            {
                "trained_baseline": True,
                "official_loader": True,
                "official_model_factory": True,
                "q1_baseline_qualified": False,
                "n_usable_mice": 5,
            }
        )
    )

    payload = official_audit.audit(
        official_repo=official_repo,
        official_smoke=smoke,
        official_trained_summary=trained,
        local_mlp_summary=tmp_path / "missing_mlp.json",
    )

    assert payload["official_trained_baseline_available"]
    assert not payload["official_q1_baseline_qualified"]
    assert not payload["official_baseline_viable"]
    assert payload["decision"] == (
        "official_sensorium_bounded_trained_baseline_available_not_q1_qualified"
    )
