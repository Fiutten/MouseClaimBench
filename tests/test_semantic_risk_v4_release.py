import copy
import json
from pathlib import Path

from mousebrainbench.benchmarks.semantic_risk_v4_release import ARTIFACTS, evaluate


def _payloads() -> dict:
    return {
        name: json.loads(path.read_text()) for name, path in ARTIFACTS.items()
    }


def test_release_audit_is_noncompensatory() -> None:
    result = evaluate(_payloads())
    assert result["implementation_complete"]
    assert result["router_repair_confirmed"]
    assert not result["external_authorization_confirmed"]
    assert not result["strong_q1_second_paper_ready"]


def test_router_success_cannot_override_external_failure() -> None:
    payloads = copy.deepcopy(_payloads())
    payloads["router_v4_2"]["primary_passed"] = True
    payloads["router_v4_2"]["decision"] = "strict_association_router_prospectively_confirmed"
    result = evaluate(payloads)
    assert result["router_repair_confirmed"]
    assert not result["strong_q1_second_paper_ready"]


def test_release_artifact_paths_exist() -> None:
    assert all(Path(path).exists() for path in ARTIFACTS.values())

