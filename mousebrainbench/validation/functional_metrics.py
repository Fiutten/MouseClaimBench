"""Metrics for simulated or empirical regional activity."""

from __future__ import annotations

import numpy as np


def functional_connectivity(activity: np.ndarray) -> np.ndarray:
    """Pearson functional connectivity across regional time series."""

    if activity.ndim != 2 or activity.shape[0] < 2:
        raise ValueError("activity must be time by region with at least two time points")
    with np.errstate(invalid="ignore", divide="ignore"):
        connectivity = np.corrcoef(activity, rowvar=False)
    return np.nan_to_num(connectivity, nan=0.0)


def fc_correlation(simulated_fc: np.ndarray, empirical_fc: np.ndarray) -> float:
    if simulated_fc.shape != empirical_fc.shape or simulated_fc.ndim != 2:
        raise ValueError("functional connectivity matrices must have equal 2D shape")
    upper = np.triu_indices_from(simulated_fc, k=1)
    if len(upper[0]) < 2:
        return 0.0
    correlation = np.corrcoef(simulated_fc[upper], empirical_fc[upper])[0, 1]
    return float(np.nan_to_num(correlation, nan=0.0))


def power_spectrum(activity: np.ndarray, dt: float) -> tuple[np.ndarray, np.ndarray]:
    if dt <= 0:
        raise ValueError("dt must be positive")
    centered = activity - activity.mean(axis=0, keepdims=True)
    frequencies = np.fft.rfftfreq(len(activity), d=dt)
    power = np.abs(np.fft.rfft(centered, axis=0)) ** 2 / len(activity)
    return frequencies, power


def region_autocorrelation(activity: np.ndarray, lag: int = 1) -> np.ndarray:
    if lag < 1 or lag >= len(activity):
        raise ValueError("lag must fall inside activity time axis")
    result = np.empty(activity.shape[1])
    for region in range(activity.shape[1]):
        result[region] = np.corrcoef(activity[:-lag, region], activity[lag:, region])[0, 1]
    return np.nan_to_num(result, nan=0.0)


def evoked_response_amplitude(
    activity: np.ndarray, time: np.ndarray, onset: float, duration: float
) -> np.ndarray:
    baseline = activity[time < onset]
    response = activity[(time >= onset) & (time < onset + duration)]
    if len(baseline) == 0 or len(response) == 0:
        raise ValueError("baseline and response windows must contain samples")
    return response.mean(axis=0) - baseline.mean(axis=0)


def response_latency(
    activity: np.ndarray,
    time: np.ndarray,
    *,
    onset: float,
    threshold_std: float = 2.0,
) -> np.ndarray:
    """Return first post-onset threshold crossing per region, or NaN if absent."""

    baseline = activity[time < onset]
    if len(baseline) < 2:
        raise ValueError("latency requires at least two baseline samples")
    threshold = baseline.mean(axis=0) + threshold_std * baseline.std(axis=0)
    post_indices = np.flatnonzero(time >= onset)
    latencies = np.full(activity.shape[1], np.nan)
    for region in range(activity.shape[1]):
        crossings = post_indices[activity[post_indices, region] > threshold[region]]
        if len(crossings):
            latencies[region] = time[crossings[0]] - onset
    return latencies
