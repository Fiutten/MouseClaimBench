import numpy as np

from mousebrainbench.data.loaders.sensorium import load_sensorium_directory


def test_load_sensorium_static_directory(tmp_path) -> None:
    root = tmp_path / "sensorium"
    (root / "data" / "images").mkdir(parents=True)
    (root / "data" / "responses").mkdir(parents=True)
    (root / "data" / "behavior").mkdir(parents=True)
    (root / "data" / "pupil_center").mkdir(parents=True)
    (root / "meta" / "trials").mkdir(parents=True)
    (root / "meta" / "neurons").mkdir(parents=True)
    for trial in range(4):
        np.save(root / "data" / "images" / f"{trial}.npy", np.full((4, 4), trial))
        np.save(root / "data" / "responses" / f"{trial}.npy", np.asarray([trial, trial + 1]))
        np.save(root / "data" / "behavior" / f"{trial}.npy", np.asarray([trial, 0, 1]))
        np.save(root / "data" / "pupil_center" / f"{trial}.npy", np.asarray([trial + 2, trial + 3]))
    np.save(root / "meta" / "trials" / "tiers.npy", np.asarray(["train", "train", "validation", "validation"]))
    np.save(root / "meta" / "trials" / "frame_image_id.npy", np.asarray([0, 1, 2, 2]))
    np.save(root / "meta" / "neurons" / "unit_ids.npy", np.asarray([10, 11]))

    table = load_sensorium_directory(root)

    assert table.modality == "static"
    assert table.n_trials == 4
    assert table.n_neurons == 2
    assert table.stimulus_features.shape == (4, 71)
    assert table.context_features.shape == (4, 5)
    assert table.stimulus_ids.tolist() == [0, 1, 2, 2]


def test_load_sensorium_dynamic_temporal_filterbank_has_fixed_width(tmp_path) -> None:
    root = tmp_path / "dynamic_sensorium"
    (root / "data" / "videos").mkdir(parents=True)
    (root / "data" / "responses").mkdir(parents=True)
    (root / "meta" / "trials").mkdir(parents=True)
    for trial, frames in enumerate((2, 3, 4)):
        video = np.full((frames, 4, 4), trial, dtype=float)
        video[:, trial % 4, :] += np.arange(frames).reshape(-1, 1)
        np.save(root / "data" / "videos" / f"{trial}.npy", video)
        np.save(root / "data" / "responses" / f"{trial}.npy", np.ones((2, frames)))
    np.save(root / "meta" / "trials" / "tiers.npy", np.asarray(["train", "train", "oracle"]))

    table = load_sensorium_directory(root, feature_mode="temporal_filterbank")

    assert table.modality == "dynamic"
    assert table.feature_mode == "temporal_filterbank"
    assert table.stimulus_features.shape == (3, 238)
    assert np.isfinite(table.stimulus_features).all()


def test_load_sensorium_aligns_sparse_trial_ids_by_file_stem(tmp_path) -> None:
    root = tmp_path / "sparse_sensorium"
    (root / "data" / "images").mkdir(parents=True)
    (root / "data" / "responses").mkdir(parents=True)
    (root / "data" / "behavior").mkdir(parents=True)
    (root / "meta" / "trials").mkdir(parents=True)
    for trial in (2, 5):
        np.save(root / "data" / "images" / f"{trial}.npy", np.full((3, 3), trial))
        np.save(root / "data" / "responses" / f"{trial}.npy", np.asarray([trial, trial + 1]))
    np.save(root / "data" / "behavior" / "2.npy", np.asarray([20]))
    np.save(root / "data" / "images" / "7.npy", np.full((3, 3), 7))
    np.save(root / "meta" / "trials" / "tiers.npy", np.asarray(["none"] * 8))
    np.save(root / "meta" / "trials" / "frame_image_id.npy", np.arange(100, 108))

    table = load_sensorium_directory(root)

    assert table.trial_ids.tolist() == [2, 5]
    assert table.tiers.tolist() == ["none", "none"]
    assert table.stimulus_ids.tolist() == [102, 105]
    assert table.responses.tolist() == [[2, 3], [5, 6]]
    assert table.context_features[:, 0].tolist() == [20, 0]
