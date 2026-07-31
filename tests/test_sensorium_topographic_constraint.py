import numpy as np

from mousebrainbench.benchmarks.sensorium_topographic_constraint import (
    evaluate_topographic_constraint,
)


def test_topographic_constraint_detects_coordinate_organized_tuning(tmp_path) -> None:
    root = tmp_path / "toy_sensorium"
    (root / "data" / "responses").mkdir(parents=True)
    (root / "meta" / "trials").mkdir(parents=True)
    (root / "meta" / "neurons").mkdir(parents=True)

    rng = np.random.default_rng(123)
    n_neurons = 80
    n_stimuli = 20
    repeats = 2
    coords_1d = np.linspace(-1, 1, n_neurons)
    coords = np.column_stack([coords_1d, np.zeros(n_neurons), np.zeros(n_neurons)])
    preferred = coords_1d[:, None]
    stimuli = np.linspace(-1, 1, n_stimuli)[None, :]
    tuning = np.exp(-((preferred - stimuli) ** 2) / 0.12)

    trial = 0
    tiers = []
    frame_ids = []
    for stimulus in range(n_stimuli):
        for _ in range(repeats):
            response = tuning[:, stimulus] + rng.normal(scale=0.01, size=n_neurons)
            np.save(root / "data" / "responses" / f"{trial}.npy", response)
            tiers.append("test")
            frame_ids.append(stimulus)
            trial += 1

    np.save(root / "meta" / "trials" / "tiers.npy", np.asarray(tiers))
    np.save(root / "meta" / "trials" / "frame_image_id.npy", np.asarray(frame_ids))
    np.save(root / "meta" / "neurons" / "cell_motor_coordinates.npy", coords)

    result = evaluate_topographic_constraint(
        root,
        eval_tier="test",
        max_neurons=80,
        pair_count=3000,
        null_repeats=40,
        seed=7,
        min_effect=0.05,
    )

    assert result["passed"]
    assert result["observed_spearman_similarity_vs_inverse_distance"] > result["null_p95"]
