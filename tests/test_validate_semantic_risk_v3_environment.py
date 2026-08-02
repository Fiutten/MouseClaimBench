import hashlib

from scripts.validate_semantic_risk_v3_environment import file_digest


def test_file_digest_streams_expected_hash(tmp_path) -> None:
    path = tmp_path / "fixture.bin"
    payload = b"mouseclaimbench-v3" * 100
    path.write_bytes(payload)

    assert file_digest(path, "sha256") == hashlib.sha256(payload).hexdigest()
