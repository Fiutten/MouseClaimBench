import json

import numpy as np

from mousebrainbench.benchmarks.causal_direction_anm import (
    anm_direction_evidence,
    run,
)


def test_anm_direction_is_deterministic_and_never_claims_causal_proof() -> None:
    rng = np.random.default_rng(9)
    x = rng.normal(size=260)
    y = np.tanh(1.3 * x) + rng.normal(scale=0.35, size=len(x))

    first = anm_direction_evidence(x, y, seed=77)
    second = anm_direction_evidence(x, y, seed=77)

    assert first == second
    assert first["sampling"]["used_samples"] == 200
    assert first["causal_proof_allowed"] is False
    assert first["predicted_direction"] in {"forward", "reverse", "uncertain"}
    assert first["status"] in {"passed", "failed", "requires_review"}


def test_anm_direction_escalates_constant_data_to_review() -> None:
    result = anm_direction_evidence(np.ones(50), np.arange(50), seed=3)

    assert result["status"] == "requires_review"
    assert result["execution_error"].startswith("ValueError")


def test_missing_tuebingen_data_is_reported_without_fabrication(tmp_path) -> None:
    output = run(
        root=tmp_path / "missing",
        output=tmp_path / "summary.json",
        markdown=tmp_path / "summary.md",
    )
    payload = json.loads(output.read_text())

    assert payload["decision"] == "tuebingen_data_missing"

