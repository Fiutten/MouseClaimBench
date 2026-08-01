import json

from mousebrainbench.benchmarks.dyadic_inference_calibration import run


def test_small_calibration_run_is_reproducible(tmp_path) -> None:
    first = run(
        output=tmp_path / "first.json",
        markdown=tmp_path / "first.md",
        n_units=24,
        trials=12,
        seed=44,
    )
    second = run(
        output=tmp_path / "second.json",
        markdown=tmp_path / "second.md",
        n_units=24,
        trials=12,
        seed=44,
    )
    left = json.loads(first.read_text())
    right = json.loads(second.read_text())

    assert left["null"] == right["null"]
    assert left["alternative"] == right["alternative"]
    assert left["directed_pairs_per_trial"] == 24 * 23

