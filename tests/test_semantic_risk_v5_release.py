import copy
import json
from pathlib import Path

from mousebrainbench.benchmarks.semantic_risk_v5_release import ARTIFACTS, evaluate


def _payloads() -> dict:
    return {name: json.loads(path.read_text()) for name, path in ARTIFACTS.items()}


def test_v5_release_preserves_positive_core_and_external_limits() -> None:
    result = evaluate(_payloads())
    assert result["methodological_core_confirmed"]
    assert not result["global_population_supported"]
    assert not result["real_domain_external_risk_confirmed"]
    assert not result["knowledge_profile_content_validated"]
    assert not result["strong_q1_second_paper_ready"]


def test_synthetic_success_cannot_compensate_for_missing_external_evidence() -> None:
    payloads = copy.deepcopy(_payloads())
    primary = "v5_1_fixed_hierarchical_threshold"
    payloads["topology_v5_1"]["risk_lock"]["comparators"][primary]["certified"] = True
    payloads["topology_v5_1"]["final_evaluation"]["comparators"][primary]["certified"] = True

    result = evaluate(payloads)

    assert result["methodological_core_confirmed"]
    assert not result["strong_q1_second_paper_ready"]


def test_v5_release_artifact_paths_exist() -> None:
    assert all(Path(path).exists() for path in ARTIFACTS.values())
