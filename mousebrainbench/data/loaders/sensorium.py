"""Load lightweight Sensorium/Sensorium-style trial directories.

The official Sensorium releases store each trial as NumPy arrays under
``data/images`` or ``data/videos`` and ``data/responses``, with split labels in
``meta/trials/tiers.npy``. This loader intentionally covers that documented
structure without depending on the heavier official PyTorch stack.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np


Modality = Literal["static", "dynamic"]
FeatureMode = Literal["summary", "temporal_filterbank"]


@dataclass(frozen=True)
class SensoriumTrialTable:
    """In-memory tabular representation of a small Sensorium cohort."""

    root: Path
    modality: Modality
    stimulus_features: np.ndarray
    context_features: np.ndarray
    responses: np.ndarray
    tiers: np.ndarray
    trial_ids: np.ndarray
    stimulus_ids: np.ndarray
    neuron_ids: np.ndarray
    feature_mode: str = "summary"

    @property
    def n_trials(self) -> int:
        return int(self.responses.shape[0])

    @property
    def n_neurons(self) -> int:
        return int(self.responses.shape[1])


def _numeric_key(path: Path) -> tuple[int, str]:
    try:
        return int(path.stem), path.name
    except ValueError:
        return 10**12, path.name


def _load_trial_arrays(directory: Path, *, max_trials: int | None = None) -> list[np.ndarray]:
    files = sorted(directory.glob("*.npy"), key=_numeric_key)
    if max_trials is not None:
        files = files[:max_trials]
    if not files:
        raise FileNotFoundError(f"no .npy trial files found in {directory}")
    return [np.load(file) for file in files]


def _load_trial_file_map(directory: Path) -> dict[int, Path]:
    """Index trial files by their numeric stem."""

    files: dict[int, Path] = {}
    for path in directory.glob("*.npy"):
        try:
            files[int(path.stem)] = path
        except ValueError:
            continue
    if not files:
        raise FileNotFoundError(f"no numeric .npy trial files found in {directory}")
    return files


def _load_optional_trial_matrix(
    directory: Path,
    *,
    n_trials: int | None = None,
    trial_ids: np.ndarray | None = None,
) -> np.ndarray:
    """Load optional trial-level covariates as ``trial x feature`` arrays."""

    if not directory.exists():
        if n_trials is None and trial_ids is not None:
            n_trials = len(trial_ids)
        return np.empty((int(n_trials or 0), 0), dtype=float)
    if n_trials is None and trial_ids is not None:
        n_trials = len(trial_ids)
    if trial_ids is None:
        if n_trials is None:
            raise ValueError("n_trials is required when trial_ids is not provided")
        arrays = _load_trial_arrays(directory, max_trials=n_trials)
        features = [
            np.nan_to_num(np.asarray(array, dtype=float).reshape(-1), nan=0.0)
            for array in arrays[:n_trials]
        ]
    else:
        files = _load_trial_file_map(directory)
        loaded: list[np.ndarray | None] = [
            np.load(files[int(trial_id)]) if int(trial_id) in files else None
            for trial_id in trial_ids
        ]
        first = next((array for array in loaded if array is not None), None)
        if first is None:
            return np.empty((int(n_trials or 0), 0), dtype=float)
        width = np.asarray(first, dtype=float).reshape(-1).shape[0]
        features = [
            np.nan_to_num(np.asarray(array, dtype=float).reshape(-1), nan=0.0)
            if array is not None
            else np.zeros(width, dtype=float)
            for array in loaded
        ]
    return np.vstack(features) if features else np.empty((n_trials, 0), dtype=float)


def _feature_vector(
    stimulus: np.ndarray,
    modality: Modality,
    *,
    feature_mode: FeatureMode = "summary",
) -> np.ndarray:
    """Extract cheap stimulus descriptors for transparent baselines.

    These features are not intended to be state of the art. They provide a
    deterministic baseline that can be tested before adding deep visual models.
    """

    values = np.nan_to_num(np.asarray(stimulus, dtype=float), nan=0.0)
    flat = values.reshape(-1)
    features = [
        float(np.mean(flat)),
        float(np.std(flat)),
        float(np.percentile(flat, 25)),
        float(np.percentile(flat, 75)),
    ]
    if modality == "dynamic" and values.ndim >= 3:
        frame_axis = 0
        frame_means = values.mean(axis=tuple(range(1, values.ndim)))
        features.extend(
            [
                float(np.std(frame_means)),
                float(np.mean(np.abs(np.diff(frame_means)))) if len(frame_means) > 1 else 0.0,
                float(values.shape[frame_axis]),
            ]
        )
    else:
        features.extend([0.0, 0.0, 1.0])

    # A global intensity summary is too weak for natural images. The pooled
    # spatial descriptor keeps the baseline transparent while preserving coarse
    # retinotopic information that a visual cortical response can exploit.
    image = values.mean(axis=0) if modality == "dynamic" and values.ndim >= 3 else values
    features.extend(_pooled_spatial_features(np.squeeze(image), bins=(8, 8)).tolist())
    if modality == "dynamic" and feature_mode == "temporal_filterbank":
        features.extend(_dynamic_temporal_filterbank_features(values).tolist())
    elif feature_mode != "summary":
        raise ValueError(f"unknown Sensorium feature mode: {feature_mode}")
    return np.asarray(features, dtype=float)


def _pooled_spatial_features(image: np.ndarray, *, bins: tuple[int, int]) -> np.ndarray:
    """Average an image into a fixed grid without image-processing dependencies."""

    values = np.asarray(image, dtype=float)
    if values.ndim > 2:
        values = values.reshape(values.shape[0], -1)
    if values.ndim != 2:
        return np.zeros(bins[0] * bins[1], dtype=float)
    row_edges = np.linspace(0, values.shape[0], bins[0] + 1, dtype=int)
    col_edges = np.linspace(0, values.shape[1], bins[1] + 1, dtype=int)
    pooled = []
    for row in range(bins[0]):
        for col in range(bins[1]):
            patch = values[row_edges[row] : row_edges[row + 1], col_edges[col] : col_edges[col + 1]]
            pooled.append(float(np.mean(patch)) if patch.size else 0.0)
    return np.asarray(pooled, dtype=float)


def _as_frame_images(stimulus: np.ndarray) -> np.ndarray:
    """Return a ``frame x row x col`` view for common Sensorium video layouts."""

    values = np.nan_to_num(np.asarray(stimulus, dtype=float), nan=0.0)
    if values.ndim == 3:
        return values
    if values.ndim == 4:
        if values.shape[-1] <= 4:
            return values.mean(axis=-1)
        if values.shape[1] <= 4:
            return values.mean(axis=1)
    if values.ndim < 3:
        return values.reshape(1, values.shape[0], -1)
    return values.reshape(values.shape[0], values.shape[1], -1)


def _temporal_pooled_features(
    values: np.ndarray, *, bins: int, spatial_bins: tuple[int, int]
) -> list[float]:
    """Pool temporal windows into a fixed-size spatial descriptor."""

    features: list[float] = []
    for indices in np.array_split(np.arange(values.shape[0]), bins):
        if len(indices):
            image = values[indices].mean(axis=0)
            features.extend(_pooled_spatial_features(image, bins=spatial_bins).tolist())
        else:
            features.extend([0.0] * (spatial_bins[0] * spatial_bins[1]))
    return features


def _dynamic_temporal_filterbank_features(stimulus: np.ndarray) -> np.ndarray:
    """Extract auditable temporal descriptors from a dynamic Sensorium video.

    The filterbank keeps explicit time information that the original summary
    descriptor discarded: coarse temporal windows, frame-to-frame motion energy,
    and low-frequency modulation of global luminance. This is still a baseline,
    not a replacement for official deep Sensorium models.
    """

    frames = _as_frame_images(stimulus)
    frame_means = frames.mean(axis=(1, 2))
    frame_stds = frames.std(axis=(1, 2))
    motion = np.abs(np.diff(frames, axis=0)) if frames.shape[0] > 1 else np.zeros_like(frames)
    motion_means = motion.mean(axis=(1, 2)) if motion.size else np.asarray([0.0])

    features: list[float] = []
    for trace in (frame_means, frame_stds, motion_means):
        trace = np.asarray(trace, dtype=float)
        centered = trace - trace.mean()
        spectrum = np.abs(np.fft.rfft(centered))
        features.extend(
            [
                float(trace.mean()),
                float(trace.std()),
                float(np.percentile(trace, 10)),
                float(np.percentile(trace, 50)),
                float(np.percentile(trace, 90)),
            ]
        )
        spectral_features = spectrum[1:9].tolist() if spectrum.size > 1 else []
        features.extend(spectral_features)
        features.extend([0.0] * (8 - len(spectral_features)))

    features.extend(_temporal_pooled_features(frames, bins=4, spatial_bins=(4, 4)))
    features.extend(_temporal_pooled_features(motion, bins=4, spatial_bins=(4, 4)))
    return np.asarray(features, dtype=float)


def _response_vector(response: np.ndarray) -> np.ndarray:
    """Collapse official static/dynamic response arrays to trial x neuron."""

    values = np.asarray(response, dtype=float)
    if values.ndim == 1:
        return np.nan_to_num(values, nan=0.0)
    if values.ndim == 2:
        # Dynamic Sensorium responses are neuron x frame; static responses are
        # usually already one value per neuron. Missing dynamic samples are
        # encoded as NaN, so the transparent trial-level target uses nanmean
        # rather than treating missing frames as zero activity.
        return np.nan_to_num(np.nanmean(values, axis=1), nan=0.0)
    collapsed = np.nanmean(values.reshape(values.shape[0], -1), axis=1)
    return np.nan_to_num(collapsed, nan=0.0)


def _load_meta_array(path: Path, *, fallback: np.ndarray) -> np.ndarray:
    return np.load(path) if path.exists() else fallback


def load_sensorium_directory(
    root: str | Path,
    *,
    modality: Modality | None = None,
    max_trials: int | None = None,
    feature_mode: FeatureMode = "summary",
) -> SensoriumTrialTable:
    """Load a documented Sensorium 2022/2023 directory into compact arrays."""

    base = Path(root)
    data_dir = base / "data"
    if modality is None:
        if (data_dir / "videos").exists():
            modality = "dynamic"
        elif (data_dir / "images").exists():
            modality = "static"
        else:
            raise FileNotFoundError("expected data/images or data/videos under Sensorium root")
    stimulus_dir = data_dir / ("videos" if modality == "dynamic" else "images")
    response_dir = data_dir / "responses"
    stimulus_files = _load_trial_file_map(stimulus_dir)
    response_files = _load_trial_file_map(response_dir)
    trial_keys = np.asarray(sorted(set(stimulus_files) & set(response_files)), dtype=int)
    if max_trials is not None:
        trial_keys = trial_keys[:max_trials]
    if len(trial_keys) == 0:
        raise ValueError("stimulus and response trial ids do not overlap")
    stimuli = [np.load(stimulus_files[int(trial_id)]) for trial_id in trial_keys]
    responses = [np.load(response_files[int(trial_id)]) for trial_id in trial_keys]

    n_trials = len(responses)
    fallback_trial_ids = trial_keys
    tier_fallback = np.full(n_trials, "train", dtype=object)
    all_tiers = _load_meta_array(base / "meta" / "trials" / "tiers.npy", fallback=tier_fallback)
    if int(trial_keys.max()) < len(all_tiers):
        tiers = all_tiers[trial_keys].astype(str)
    else:
        tiers = tier_fallback.astype(str)
    all_trial_ids = _load_meta_array(
        base / "meta" / "trials" / "trial_idx.npy", fallback=fallback_trial_ids
    )
    if int(trial_keys.max()) < len(all_trial_ids):
        trial_ids = all_trial_ids[trial_keys]
    else:
        trial_ids = fallback_trial_ids
    all_stimulus_ids = _load_meta_array(
        base / "meta" / "trials" / "frame_image_id.npy", fallback=fallback_trial_ids
    )
    if int(trial_keys.max()) < len(all_stimulus_ids):
        stimulus_ids = all_stimulus_ids[trial_keys]
    else:
        stimulus_ids = fallback_trial_ids
    response_matrix = np.stack([_response_vector(response) for response in responses])
    context_features = np.column_stack(
        [
            _load_optional_trial_matrix(
                data_dir / "behavior", n_trials=n_trials, trial_ids=trial_keys
            ),
            _load_optional_trial_matrix(
                data_dir / "pupil_center", n_trials=n_trials, trial_ids=trial_keys
            ),
        ]
    )
    neuron_ids = _load_meta_array(
        base / "meta" / "neurons" / "unit_ids.npy",
        fallback=np.arange(response_matrix.shape[1]),
    )[: response_matrix.shape[1]]
    return SensoriumTrialTable(
        root=base,
        modality=modality,
        stimulus_features=np.stack(
            [_feature_vector(stimulus, modality, feature_mode=feature_mode) for stimulus in stimuli]
        ),
        context_features=context_features,
        responses=response_matrix,
        tiers=tiers,
        trial_ids=trial_ids,
        stimulus_ids=stimulus_ids,
        neuron_ids=neuron_ids,
        feature_mode=feature_mode,
    )
