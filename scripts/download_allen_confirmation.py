"""Download and verify the sealed Allen VBN confirmation cohort."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import subprocess
from pathlib import Path


def blake2b(path: Path) -> str:
    digest = hashlib.blake2b()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sealed_plan", type=Path)
    parser.add_argument("sessions_root", type=Path)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    plan = json.loads(args.sealed_plan.read_text())
    def download(record: dict[str, object]) -> dict[str, object]:
        session_id = int(record["session_id"])
        destination = args.sessions_root / str(session_id) / f"ecephys_session_{session_id}.nwb"
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists() and blake2b(destination) == record["file_hash_blake2b"]:
            return {"session_id": session_id, "status": "already_verified"}
        partial = destination.with_suffix(".nwb.part")
        subprocess.run(
            [
                "/usr/bin/curl",
                "-L",
                "--fail",
                "--retry",
                "5",
                "--retry-delay",
                "5",
                "--silent",
                "--show-error",
                "-C",
                "-",
                "-o",
                str(partial),
                record["url"],
            ],
            check=True,
        )
        actual = blake2b(partial)
        if actual != record["file_hash_blake2b"]:
            raise RuntimeError(f"BLAKE2b mismatch for session {session_id}: {actual}")
        partial.replace(destination)
        return {"session_id": session_id, "status": "downloaded_verified"}

    report = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(download, record): record for record in plan["sessions"]}
        for future in as_completed(futures):
            result = future.result()
            report.append(result)
            print(json.dumps(result), flush=True)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
