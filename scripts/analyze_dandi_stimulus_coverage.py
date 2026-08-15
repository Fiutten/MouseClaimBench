#!/usr/bin/env python3
"""Read only DANDI stimulus vectors and audit the frozen chronological split.

The script uses HTTP range requests against official DANDI asset URLs. It does
not download the 7.6 GB selected corpus and does not read neural response data.
Exact comparability is enforced by matching each finite stimulus count to the
trial count already frozen by the predictive endpoint.
"""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import h5py
import requests

from mousebrainbench import __version__
from mousebrainbench.artifacts import code_revision
from mousebrainbench.benchmarks.dandi_stimulus_coverage import (
    aggregate_subject_coverage,
    analyze_stimulus_arrays,
)

API = "https://api.dandiarchive.org/api"
DANDISET = "000039"
VERSION = "0.230223.1216"
RESULT = Path("results/dandi_profile_v2_1/summary.json")
OUTPUT = Path("results/dandi_stimulus_coverage/summary.json")
MARKDOWN = Path("results/dandi_stimulus_coverage/summary.md")
CONTRAST_PATH = "intervals/epochs/contrast"
DIRECTION_PATH = "intervals/epochs/direction"


def _request_json(url: str) -> dict[str, Any]:
    response = requests.get(url, timeout=120)
    response.raise_for_status()
    return response.json()


def _selected_assets() -> list[dict[str, Any]]:
    url = f"{API}/dandisets/{DANDISET}/versions/{VERSION}/assets/?page_size=100"
    rows: list[dict[str, Any]] = []
    while url:
        page = _request_json(url)
        rows.extend(page["results"])
        url = page.get("next")
    eligible = [row for row in rows if row["path"].endswith("behavior+ophys.nwb")]
    subjects = sorted({row["path"].split("/")[0] for row in eligible})
    return [
        min(
            (row for row in eligible if row["path"].split("/")[0] == subject),
            key=lambda row: row["path"],
        )
        for subject in subjects
    ]


def _expected_trials(path: Path) -> dict[str, int]:
    payload = json.loads(path.read_text())
    application = next(
        row for row in payload["applications"] if row["resource"] == "DANDI:000039"
    )
    return {
        str(row["subject"]).removeprefix("sub-"): int(row["trials"])
        for row in application["subject_results"]
        if row["usable"]
    }


def _analyze_asset(asset: dict[str, Any], expected: dict[str, int]) -> dict[str, Any]:
    try:
        import fsspec
    except ImportError as exc:
        raise RuntimeError(
            "remote coverage requires the `dandi-remote` optional dependencies"
        ) from exc
    subject = asset["path"].split("/")[0].removeprefix("sub-")
    detail = _request_json(f"{API}/assets/{asset['asset_id']}/")
    source_url = next(
        url for url in detail["contentUrl"] if "dandiarchive.s3.amazonaws.com" in url
    )
    with fsspec.open(
        source_url,
        "rb",
        block_size=1024 * 1024,
        cache_type="blockcache",
    ) as remote, h5py.File(remote, "r") as handle:
        contrast = handle[CONTRAST_PATH][:]
        direction = handle[DIRECTION_PATH][:]
    coverage = analyze_stimulus_arrays(
        contrast,
        direction,
        expected_trials=expected[subject],
    )
    return {
        "subject": subject,
        "asset_id": asset["asset_id"],
        "asset_path": asset["path"],
        "asset_size": int(asset["size"]),
        **coverage,
    }


def run(*, result: Path, output: Path, markdown: Path, workers: int) -> Path:
    expected = _expected_trials(result)
    assets = _selected_assets()
    rows: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(_analyze_asset, asset, expected): asset for asset in assets
        }
        for future in as_completed(futures):
            row = future.result()
            rows.append(row)
            print(f"coverage {len(rows)}/{len(assets)}: {row['subject']}")
    rows.sort(key=lambda row: row["subject"])
    if {row["subject"] for row in rows} != set(expected):
        raise RuntimeError("coverage subjects do not match the frozen endpoint subjects")
    aggregate = aggregate_subject_coverage(rows)
    payload = {
        "version": __version__,
        "git_revision": code_revision(),
        "analysis": "descriptive_dandi_chronological_stimulus_coverage",
        "resource": f"DANDI:{DANDISET}",
        "resource_version": VERSION,
        "selection_rule": "lexicographically first behavior+ophys asset per subject",
        "split": "first_60_percent_train_next_20_percent_reserved_last_20_percent_test",
        "coverage_definition": (
            "held-out contrast-direction conditions represented in the training segment"
        ),
        "response_data_read": False,
        "frozen_trial_counts_matched": True,
        "subjects": rows,
        "aggregate": aggregate,
        "decision": "descriptive_coverage_complete_without_authorization_change",
        "interpretation": (
            "This transparency check describes stimulus support under the frozen split. "
            "It did not tune the model, thresholds, split, or authorization decision."
        ),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n")
    summary = aggregate["unique_test_condition_coverage"]
    trial = aggregate["test_trial_condition_coverage"]
    markdown.write_text(
        "\n".join(
            (
                "# DANDI chronological stimulus coverage",
                "",
                f"- Subjects: `{aggregate['subjects']}`",
                f"- Minimum unique-condition coverage: `{summary['minimum']:.4f}`",
                f"- Median unique-condition coverage: `{summary['median']:.4f}`",
                f"- Minimum held-out-trial coverage: `{trial['minimum']:.4f}`",
                f"- Subjects with a test-only contrast: `{aggregate['subjects_with_test_only_contrast']}`",
                f"- Subjects with a test-only direction: `{aggregate['subjects_with_test_only_direction']}`",
                f"- Subjects with a test-only condition: `{aggregate['subjects_with_test_only_condition']}`",
                "- Authorization rule changed: `false`",
                "",
                payload["interpretation"],
                "",
            )
        )
    )
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result", type=Path, default=RESULT)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--markdown", type=Path, default=MARKDOWN)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    print(
        run(
            result=args.result,
            output=args.output,
            markdown=args.markdown,
            workers=max(1, args.workers),
        ).resolve()
    )


if __name__ == "__main__":
    main()
