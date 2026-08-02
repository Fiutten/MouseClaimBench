"""Freeze unused IBL mice without reading behavioral outcomes."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd

DEFAULT_ROOT = Path("data/external/ibl")
DEFAULT_PRIOR = Path("configs/benchmarks/ibl_bwm_v3_selection.json")
DEFAULT_OUTPUT = Path("configs/benchmarks/ibl_behavior_v5_selection.json")
FIXTURE = "2023_12_bwm_release.csv"
CLUSTERS = "clusters_2024_Q2.pqt"
MIN_GOOD_UNITS = 50
MIN_COSMOS_CLASSES = 2
ROLE_SIZES = {"calibration": 40, "risk_lock": 35, "final": 35}
NAMESPACE = "mouseclaimbench-ibl-behavior-v5-selection"


def _hash_text(*values: str) -> str:
    return hashlib.sha256(":".join(values).encode()).hexdigest()


def _file_hash(path: Path, algorithm: str) -> str:
    digest = hashlib.new(algorithm)
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def build_selection(
    fixture: pd.DataFrame,
    clusters: pd.DataFrame,
    *,
    excluded_subjects: set[str],
) -> list[dict[str, Any]]:
    """Choose one eligible insertion per previously unused mouse by identifier hash."""

    good = clusters[
        (pd.to_numeric(clusters["label"], errors="coerce") >= 1.0)
        & ~clusters["acronym"].astype(str).isin(("root", "void"))
    ]
    counts = (
        good.groupby(good["pid"].astype(str))
        .agg(good_units=("pid", "size"), cosmos_classes=("acronym", "nunique"))
        .reset_index(names="pid")
    )
    candidates = fixture.copy()
    candidates["pid"] = candidates["pid"].astype(str)
    candidates["subject"] = candidates["subject"].astype(str)
    candidates = candidates.merge(counts, on="pid", how="left")
    candidates[["good_units", "cosmos_classes"]] = candidates[
        ["good_units", "cosmos_classes"]
    ].fillna(0)
    candidates = candidates[
        (candidates["good_units"] >= MIN_GOOD_UNITS)
        & (candidates["cosmos_classes"] >= MIN_COSMOS_CLASSES)
        & ~candidates["subject"].isin(excluded_subjects)
    ].copy()
    candidates["pid_order"] = [
        _hash_text(NAMESPACE, subject, pid)
        for subject, pid in zip(candidates["subject"], candidates["pid"], strict=True)
    ]
    candidates = candidates.sort_values(["subject", "pid_order"]).drop_duplicates("subject")
    candidates["subject_order"] = [
        _hash_text(NAMESPACE, subject) for subject in candidates["subject"]
    ]
    candidates = candidates.sort_values("subject_order").reset_index(drop=True)
    expected = sum(ROLE_SIZES.values())
    if len(candidates) != expected:
        raise ValueError(f"expected exactly {expected} unused eligible mice, found {len(candidates)}")

    boundaries: list[tuple[int, int, str]] = []
    start = 0
    for role, count in ROLE_SIZES.items():
        boundaries.append((start, start + count, role))
        start += count

    rows: list[dict[str, Any]] = []
    for index, row in candidates.iterrows():
        role = next(role for lower, upper, role in boundaries if lower <= index < upper)
        rows.append(
            {
                "rank": int(index),
                "role": role,
                "subject": str(row["subject"]),
                "subject_hash": _hash_text(NAMESPACE, str(row["subject"])),
                "pid": str(row["pid"]),
                "eid": str(row["eid"]),
                "probe_name": str(row["probe_name"]),
                "session_number": int(row["session_number"]),
                "date": str(row["date"]),
                "lab": str(row["lab"]),
                "good_units": int(row["good_units"]),
                "cosmos_classes": int(row["cosmos_classes"]),
            }
        )
    return rows


def prepare(root: Path, prior_path: Path, output: Path) -> Path:
    prior = json.loads(prior_path.read_text())
    excluded = {str(row["subject"]) for row in prior["insertions"]}
    fixture_path = root / FIXTURE
    clusters_path = root / CLUSTERS
    fixture = pd.read_csv(fixture_path)
    clusters = pd.read_parquet(clusters_path, columns=["pid", "label", "acronym"])
    rows = build_selection(fixture, clusters, excluded_subjects=excluded)
    payload = {
        "protocol": "ibl_behavior_mouse_population_v5",
        "status": "frozen_before_behavioral_table_download_or_outcome_generation",
        "selection_uses_behavioral_outcomes": False,
        "published_dataset_doi": "10.1038/s41586-025-09235-0",
        "source_release": "Brainwidemap 2023_12 fixture and 2024_Q2 aggregate",
        "source_files": {
            FIXTURE: {
                "sha256": _file_hash(fixture_path, "sha256"),
                "rows": len(fixture),
            },
            CLUSTERS: {
                "md5": _file_hash(clusters_path, "md5"),
                "rows": len(clusters),
            },
            str(prior_path): {"sha256": _file_hash(prior_path, "sha256")},
        },
        "eligibility": {
            "minimum_good_units": MIN_GOOD_UNITS,
            "minimum_non_root_cosmos_classes": MIN_COSMOS_CLASSES,
            "one_insertion_per_mouse": True,
            "excluded_previously_consumed_mice": len(excluded),
            "replacement_after_behavioral_failure": False,
        },
        "selection_rule": (
            "one eligible insertion per unused subject by SHA-256 subject/pid order, then "
            "SHA-256 subject order allocated 40 calibration, 35 risk-lock, 35 final"
        ),
        "role_counts": ROLE_SIZES,
        "unique_mice": len(rows),
        "labs": sorted({row["lab"] for row in rows}),
        "insertions": rows,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--prior", type=Path, default=DEFAULT_PRIOR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    print(prepare(args.root, args.prior, args.output).resolve())


if __name__ == "__main__":
    main()
