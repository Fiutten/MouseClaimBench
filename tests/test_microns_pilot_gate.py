import json

from mousebrainbench.benchmarks.microns_pilot_gate import evaluate_manifest


def test_microns_gate_defers_without_manifest(tmp_path) -> None:
    payload = evaluate_manifest(tmp_path / "missing.json")

    assert not payload["approved"]
    assert payload["decision"] == "defer_microns_until_small_manifest_available"


def test_microns_gate_approves_bounded_manifest(tmp_path) -> None:
    manifest = tmp_path / "pilot_manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "n_neurons": 1200,
                "has_spatial_coordinates": True,
                "has_functional_responses": True,
                "has_structural_edges": True,
                "n_structural_edges": 900,
                "estimated_download_gb": 1.5,
            }
        )
    )

    payload = evaluate_manifest(manifest)

    assert payload["approved"]
    assert payload["q1_pilot_approved"]
    assert payload["micro_pilot_approved"]
    assert payload["decision"] == "approve_bounded_microns_structure_function_pilot"


def test_microns_gate_blocks_coregistered_units_without_structural_edges(tmp_path) -> None:
    manifest = tmp_path / "pilot_manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "n_neurons": 172,
                "has_spatial_coordinates": True,
                "has_functional_responses": True,
                "has_structural_edges": False,
                "estimated_download_gb": 0.01779,
            }
        )
    )

    payload = evaluate_manifest(manifest)

    assert not payload["approved"]
    assert payload["decision"] == "defer_microns_pilot_manifest_insufficient"


def test_microns_gate_allows_micro_pilot_without_q1_unlock(tmp_path) -> None:
    manifest = tmp_path / "pilot_manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "n_neurons": 172,
                "has_spatial_coordinates": True,
                "has_functional_responses": True,
                "has_structural_edges": True,
                "n_structural_edges": 62,
                "estimated_download_gb": 0.02,
            }
        )
    )

    payload = evaluate_manifest(manifest)

    assert not payload["approved"]
    assert not payload["q1_pilot_approved"]
    assert payload["micro_pilot_approved"]
    assert payload["decision"] == "approve_microns_structure_function_micro_pilot"


def test_microns_gate_rejects_too_few_real_edges(tmp_path) -> None:
    manifest = tmp_path / "pilot_manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "n_neurons": 172,
                "has_spatial_coordinates": True,
                "has_functional_responses": True,
                "has_structural_edges": True,
                "n_structural_edges": 1,
                "estimated_download_gb": 0.02,
            }
        )
    )

    payload = evaluate_manifest(manifest)

    assert not payload["approved"]
    assert not payload["micro_pilot_approved"]
    assert payload["decision"] == "defer_microns_pilot_manifest_insufficient"
