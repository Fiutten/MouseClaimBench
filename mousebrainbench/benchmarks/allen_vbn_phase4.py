"""Test whether Allen VBN contains reproducible directed temporal signatures."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import kendalltau
import yaml

from mousebrainbench import __version__
from mousebrainbench.artifacts import code_revision
from mousebrainbench.benchmarks.allen_vbn_phase2b import _run_id
from mousebrainbench.data.loaders.allen_vbn import AllenVBNRepository


def load_config(path: str | Path) -> dict[str, Any]:
    source = Path(path).resolve()
    config = yaml.safe_load(source.read_text())
    config["_config_path"] = str(source)
    return config


def _kendall_tau(left: np.ndarray, right: np.ndarray) -> float:
    value = kendalltau(left, right, nan_policy="omit").statistic
    return float(np.nan_to_num(value, nan=0.0))


def _agreement(left: np.ndarray, right: np.ndarray) -> float:
    return float(np.mean(left == right))


def _baseline_correct(activity: np.ndarray, time: np.ndarray) -> np.ndarray:
    return activity - activity[time < 0].mean(axis=0, keepdims=True)


def latency_signature(activity: np.ndarray, time: np.ndarray, response_stop: float) -> np.ndarray:
    """Return peak-latency bins after stimulus onset for each region."""

    corrected = _baseline_correct(activity, time)
    mask = (time > 0) & (time <= response_stop)
    if np.count_nonzero(mask) < 2:
        raise ValueError("latency signature requires at least two postevent bins")
    return time[mask][np.argmax(corrected[mask], axis=0)]


def resolvable_pair_fraction(latencies: np.ndarray, bin_size: float) -> float:
    """Fraction of region pairs separated by at least one temporal bin."""

    differences = np.abs(latencies[:, None] - latencies[None, :])
    pair_mask = np.triu(np.ones_like(differences, dtype=bool), k=1)
    return float(np.mean(differences[pair_mask] >= bin_size))


def lead_lag_signature(
    activity: np.ndarray,
    time: np.ndarray,
    response_stop: float,
    max_lag_bins: int,
) -> np.ndarray:
    """Return signs of pairwise lag maximizing cross-correlation."""

    corrected = _baseline_correct(activity, time)
    mask = (time > 0) & (time <= response_stop)
    post = corrected[mask]
    signatures = []
    for left in range(post.shape[1] - 1):
        for right in range(left + 1, post.shape[1]):
            best_lag, best_score = 0, -np.inf
            for lag in range(-max_lag_bins, max_lag_bins + 1):
                if lag < 0:
                    a, b = post[-lag:, left], post[:lag, right]
                elif lag > 0:
                    a, b = post[:-lag, left], post[lag:, right]
                else:
                    a, b = post[:, left], post[:, right]
                if len(a) < 3 or np.std(a) == 0 or np.std(b) == 0:
                    score = -np.inf
                else:
                    score = float(np.corrcoef(a, b)[0, 1])
                if score > best_score:
                    best_score, best_lag = score, lag
            signatures.append(np.sign(best_lag))
    return np.asarray(signatures, dtype=int)


def _extract(
    repository: AllenVBNRepository,
    session_ids: list[int],
    config: dict[str, Any],
    *,
    event_time_offset_seconds: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    data, target = config["data"], config["target"]
    full, odd, even, times = [], [], [], []
    for session_id in session_ids:
        response = repository.extract_change_response_timecourse(
            session_id,
            tuple(data["regions"]),
            min_units_per_region=int(data["min_units_per_region"]),
            start_seconds=float(target["start_seconds"]),
            stop_seconds=float(target["stop_seconds"]),
            bin_size_seconds=float(target["bin_size_seconds"]),
            event_time_offset_seconds=event_time_offset_seconds,
        )
        full.append(response.activity_hz)
        odd.append(response.odd_event_activity_hz)
        even.append(response.even_event_activity_hz)
        times.append(response.time)
    if not all(np.array_equal(times[0], item) for item in times[1:]):
        raise ValueError("time axes are not identical across sessions")
    return times[0], np.stack(full), np.stack(odd), np.stack(even)


def _leave_one_mouse_reference(signatures: np.ndarray, mouse_ids: np.ndarray, index: int) -> np.ndarray:
    candidates = signatures[mouse_ids != mouse_ids[index]]
    return np.median(candidates, axis=0)


def _evaluate_latency(
    full: np.ndarray,
    odd: np.ndarray,
    even: np.ndarray,
    time: np.ndarray,
    mouse_ids: np.ndarray,
    config: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    target = config["target"]
    bin_size = float(target["bin_size_seconds"])
    response_stop = float(target["response_stop_seconds"])
    full_lat = np.stack([latency_signature(item, time, response_stop) for item in full])
    odd_lat = np.stack([latency_signature(item, time, response_stop) for item in odd])
    even_lat = np.stack([latency_signature(item, time, response_stop) for item in even])
    rng = np.random.default_rng(int(config["analysis"]["seed"]))
    real, split = [], []
    null = np.empty((len(full_lat), int(config["analysis"]["permutations"])))
    for index in range(len(full_lat)):
        reference = _leave_one_mouse_reference(full_lat, mouse_ids, index)
        real.append(_kendall_tau(full_lat[index], reference))
        split.append(_kendall_tau(odd_lat[index], even_lat[index]))
        for permutation in range(null.shape[1]):
            null[index, permutation] = _kendall_tau(
                full_lat[index],
                reference[rng.permutation(reference.shape[0])],
            )
    resolvable = np.asarray([resolvable_pair_fraction(item, bin_size) for item in full_lat])
    real_array, split_array = np.asarray(real), np.asarray(split)
    return {
        "median_cross_mouse_tau": float(np.median(real_array)),
        "median_split_half_tau": float(np.median(split_array)),
        "fraction_above_own_null_95th": float(np.mean(real_array > np.quantile(null, 0.95, axis=1))),
        "median_resolvable_pair_fraction": float(np.median(resolvable)),
    }, {
        "full_latency": full_lat,
        "odd_latency": odd_lat,
        "even_latency": even_lat,
        "cross_mouse_tau": real_array,
        "split_half_tau": split_array,
        "null": null,
        "resolvable_pair_fraction": resolvable,
    }


def _evaluate_lead_lag(
    full: np.ndarray,
    odd: np.ndarray,
    even: np.ndarray,
    shifted_full: np.ndarray,
    time: np.ndarray,
    mouse_ids: np.ndarray,
    config: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    target = config["target"]
    response_stop = float(target["response_stop_seconds"])
    max_lag = int(target["maximum_lead_lag_bins"])
    full_sig = np.stack([lead_lag_signature(item, time, response_stop, max_lag) for item in full])
    odd_sig = np.stack([lead_lag_signature(item, time, response_stop, max_lag) for item in odd])
    even_sig = np.stack([lead_lag_signature(item, time, response_stop, max_lag) for item in even])
    shifted_sig = np.stack([lead_lag_signature(item, time, response_stop, max_lag) for item in shifted_full])
    rng = np.random.default_rng(int(config["analysis"]["seed"]) + 1)
    real, split, shifted = [], [], []
    null = np.empty((len(full_sig), int(config["analysis"]["permutations"])))
    for index in range(len(full_sig)):
        reference = _leave_one_mouse_reference(full_sig, mouse_ids, index)
        shifted_reference = _leave_one_mouse_reference(shifted_sig, mouse_ids, index)
        real.append(_agreement(full_sig[index], reference))
        split.append(_agreement(odd_sig[index], even_sig[index]))
        shifted.append(_agreement(shifted_sig[index], shifted_reference))
        for permutation in range(null.shape[1]):
            null[index, permutation] = _agreement(
                full_sig[index],
                reference[rng.permutation(reference.shape[0])],
            )
    real_array, split_array, shifted_array = map(np.asarray, (real, split, shifted))
    nonzero_fraction = np.mean(full_sig != 0, axis=1)
    return {
        "median_cross_mouse_agreement": float(np.median(real_array)),
        "median_split_half_agreement": float(np.median(split_array)),
        "fraction_above_own_null_95th": float(np.mean(real_array > np.quantile(null, 0.95, axis=1))),
        "shifted_event_median_cross_mouse_agreement": float(np.median(shifted_array)),
        "actual_minus_shifted_median_agreement": float(np.median(real_array) - np.median(shifted_array)),
        "median_nonzero_pair_fraction": float(np.median(nonzero_fraction)),
    }, {
        "full_signature": full_sig,
        "odd_signature": odd_sig,
        "even_signature": even_sig,
        "shifted_signature": shifted_sig,
        "cross_mouse_agreement": real_array,
        "split_half_agreement": split_array,
        "shifted_cross_mouse_agreement": shifted_array,
        "nonzero_pair_fraction": nonzero_fraction,
        "null": null,
    }


def run(config: dict[str, Any]) -> Path:
    repository = AllenVBNRepository(config["data"]["root"])
    session_ids = [int(value) for value in config["data"]["development_session_ids"]]
    sessions = repository.sessions().set_index("ecephys_session_id")
    mouse_ids = np.asarray([int(sessions.at[session_id, "mouse_id"]) for session_id in session_ids])
    time, full, odd, even = _extract(repository, session_ids, config)
    shifted_time, shifted_full, _, _ = _extract(
        repository,
        session_ids,
        config,
        event_time_offset_seconds=float(config["target"]["shifted_event_offset_seconds"]),
    )
    if not np.array_equal(time, shifted_time):
        raise ValueError("shifted control changed the time axis")
    latency_metrics, latency_arrays = _evaluate_latency(full, odd, even, time, mouse_ids, config)
    lead_lag_metrics, lead_lag_arrays = _evaluate_lead_lag(
        full, odd, even, shifted_full, time, mouse_ids, config
    )
    thresholds = config["analysis"]["thresholds"]
    checks = {
        "latency_split_half_passed": latency_metrics["median_split_half_tau"]
        > thresholds["median_latency_split_half_tau_gt"],
        "latency_cross_mouse_passed": latency_metrics["median_cross_mouse_tau"]
        > thresholds["median_latency_cross_mouse_tau_gt"],
        "latency_null_passed": latency_metrics["fraction_above_own_null_95th"]
        >= thresholds["latency_fraction_above_null_95_gte"],
        "lead_lag_split_half_passed": lead_lag_metrics["median_split_half_agreement"]
        > thresholds["median_lead_lag_split_half_agreement_gt"],
        "lead_lag_cross_mouse_passed": lead_lag_metrics["median_cross_mouse_agreement"]
        > thresholds["median_lead_lag_cross_mouse_agreement_gt"],
        "lead_lag_null_passed": lead_lag_metrics["fraction_above_own_null_95th"]
        >= thresholds["lead_lag_fraction_above_null_95_gte"],
        "lead_lag_shifted_control_passed": lead_lag_metrics["actual_minus_shifted_median_agreement"]
        > thresholds["actual_minus_shifted_lead_lag_agreement_gt"],
        "latency_resolution_passed": latency_metrics["median_resolvable_pair_fraction"]
        >= thresholds["median_resolvable_pair_fraction_gte"],
        "lead_lag_nonzero_fraction_passed": lead_lag_metrics["median_nonzero_pair_fraction"]
        >= thresholds["median_nonzero_lead_lag_pair_fraction_gte"],
    }
    passed = all(checks.values())
    metrics = {
        "interpretation": "phase4_development_identifiability_only",
        "sessions": len(session_ids),
        "unique_mice": len(set(mouse_ids)),
        "latency": latency_metrics,
        "lead_lag": lead_lag_metrics,
        **checks,
        "identifiability_passed": passed,
        "decision": "candidate_directed_signature" if passed else "directed_signature_not_supported",
    }
    output = Path(config["output"]["root"]) / f"{config['output']['name']}-{_run_id(config)}"
    output.mkdir(parents=True, exist_ok=True)
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2))
    (output / "config.yaml").write_text(yaml.safe_dump({k: v for k, v in config.items() if k != "_config_path"}))
    (output / "provenance.json").write_text(
        json.dumps({"version": __version__, "git_revision": code_revision()}, indent=2)
    )
    np.savez_compressed(
        output / "arrays.npz",
        session_ids=np.asarray(session_ids),
        mouse_ids=mouse_ids,
        time=time,
        **{f"latency_{key}": value for key, value in latency_arrays.items()},
        **{f"lead_lag_{key}": value for key, value in lead_lag_arrays.items()},
    )
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    args = parser.parse_args()
    output = run(load_config(args.config))
    print(json.dumps({"output": str(output.resolve())}))


if __name__ == "__main__":
    main()
