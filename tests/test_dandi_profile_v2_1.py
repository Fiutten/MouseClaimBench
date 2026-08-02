from pathlib import Path

import h5py
import numpy as np
import yaml

from mousebrainbench.benchmarks.dandi_profile_v2_1 import (
    _contrast_features,
    _fact,
    analyze_contrast_subject,
)
from mousebrainbench.knowledge import (
    ClaimAuthorizationSystem,
    load_authorization_profile_v2,
)
from mousebrainbench.validation.evidence_contract import EvidenceStatus


def test_contrast_feature_map_has_frozen_five_columns() -> None:
    features = _contrast_features(
        np.asarray([0.05, 0.80]), np.asarray([0.0, 180.0])
    )

    assert features.shape == (2, 5)
    assert np.isfinite(features).all()


def test_zero_exclusions_are_explicit_not_missing() -> None:
    block = _fact(
        "data_quality",
        EvidenceStatus.PASSED,
        "manifest.json",
        "frozen quality rule",
        {
            "source": "DANDI:test",
            "lineage": "manifest.json",
            "exclusions": {"count": 0, "assets": [], "rule": "schema only"},
            "missingness": "finite trials only",
            "quality_checks": {"usable_subjects": 32},
            "result": True,
        },
    )

    decision = ClaimAuthorizationSystem(
        load_authorization_profile_v2(), {"data_quality": block}
    ).infer("bounded_predictive_performance")
    fact = next(item for item in decision.facts if item.name == "data_quality")

    assert fact.effective_status is EvidenceStatus.PASSED
    assert fact.missing_required_observations == ()


def test_synthetic_nwb_schema_executes_frozen_subject_contract(tmp_path: Path) -> None:
    protocol = yaml.safe_load(
        Path("configs/benchmarks/dandi_contrast_profile_v2_1.yaml").read_text()
    )
    path = tmp_path / "subject.nwb"
    rng = np.random.default_rng(7)
    trials = 300
    starts = np.arange(trials) * 4
    stops = starts + 4
    contrast = np.resize(np.asarray([0.05, 0.1, 0.2, 0.4, 0.8]), trials)
    direction = np.resize(np.arange(0.0, 360.0, 45.0), trials)
    signal = np.repeat(np.log1p(100 * contrast), 4) + rng.normal(0, 0.05, trials * 4)
    events = np.column_stack((signal, signal + 0.01, signal - 0.01))
    with h5py.File(path, "w") as handle:
        handle.create_dataset("general/subject/subject_id", data=np.bytes_("synthetic"))
        handle.create_dataset(
            "processing/brain_observatory_pipeline/l0_events/dff_events/data",
            data=events,
        )
        handle.create_dataset("intervals/epochs/contrast", data=contrast)
        handle.create_dataset("intervals/epochs/direction", data=direction)
        handle.create_dataset("intervals/epochs/start_time", data=starts)
        handle.create_dataset("intervals/epochs/stop_time", data=stops)

    result = analyze_contrast_subject(path, protocol)

    assert result["usable"] is True
    assert result["trials"] == trials
    assert result["correlation"] > 0.9
    assert result["model_mse_sum"] < result["baseline_mse_sum"]
