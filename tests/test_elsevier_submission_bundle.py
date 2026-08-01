from scripts.build_elsevier_submission import run


def test_elsevier_bundle_is_flat_and_rewrites_authoring_paths(tmp_path) -> None:
    output = run(tmp_path / "submission")
    main = (output / "main.tex").read_text()

    assert not any(path.is_dir() for path in output.iterdir())
    assert "sections/" not in main
    assert "tables/" not in main
    assert "figures/" not in main
    assert (output / "manifest.json").exists()
