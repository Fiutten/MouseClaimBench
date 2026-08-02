import pytest

from scripts.fetch_ibl_behavior_v5 import select_trial_dataset


def _row(*, default: bool, qc: str, public: int = 1) -> dict:
    return {
        "name": "_ibl_trials.table.pqt",
        "default_dataset": default,
        "qc": qc,
        "public": public,
        "file_size": 123,
        "hash": "abc",
        "url": "https://openalyx.internationalbrainlab.org/datasets/dataset-id",
        "revision": "2025-03-03",
        "file_records": [
            {
                "exists": True,
                "data_repository": "flatiron_lab",
                "data_url": "https://flatiron.example/trials.pqt",
            },
            {
                "exists": True,
                "data_repository": "aws_lab",
                "data_url": "https://aws.example/trials.pqt",
            },
        ],
    }


def test_selects_only_default_qc_passed_table_and_prefers_aws() -> None:
    selected = select_trial_dataset(
        [_row(default=False, qc="NOT_SET"), _row(default=True, qc="PASS")]
    )
    assert selected["dataset_uuid"] == "dataset-id"
    assert selected["revision"] == "2025-03-03"
    assert selected["repository"] == "aws_lab"


def test_rejects_ambiguous_default_tables() -> None:
    with pytest.raises(ValueError, match="expected one"):
        select_trial_dataset([_row(default=True, qc="PASS"), _row(default=True, qc="PASS")])
