import pytest

from scripts.build_elsevier_submission import _assert_unique_basenames, run


def test_elsevier_bundle_is_flat_and_rewrites_authoring_paths(tmp_path) -> None:
    output = run(tmp_path / "submission")
    main = (output / "main.tex").read_text()

    assert not any(path.is_dir() for path in output.iterdir())
    assert "sections/" not in main
    assert "tables/" not in main
    assert "figures/" not in main
    assert (output / "elsarticle-num.bst").exists()
    assert (output / "figure_captions.txt").exists()
    assert (output / "highlights.txt").exists()
    assert (output / "related_manuscript_statement.txt").exists()
    assert (output / "standards_workflow.pdf").exists()
    assert (output / "integrity_ablation_v2.tex").exists()
    assert (output / "integrity_ablation_v2_figure.tex").exists()
    assert "fig:integrity-ablation-v2" in (
        output / "integrity_ablation_v2_figure.tex"
    ).read_text()
    assert not (output / "claimbench_workflow.png").exists()
    assert (output / "manifest.json").exists()

    captions = (output / "figure_captions.txt").read_text()
    assert captions.count("Figure ") == 5
    assert "Figure 1. Three-gate MouseClaimBench workflow." in captions
    assert "Figure 5. Prospective DANDI outcomes" in captions


def test_elsevier_bundle_rejects_duplicate_flat_names(tmp_path) -> None:
    first = tmp_path / "figures" / "duplicate.tex"
    second = tmp_path / "tables" / "duplicate.tex"
    first.parent.mkdir()
    second.parent.mkdir()
    first.write_text("figure")
    second.write_text("table")

    with pytest.raises(ValueError, match="dependency collision"):
        _assert_unique_basenames([first, second])
