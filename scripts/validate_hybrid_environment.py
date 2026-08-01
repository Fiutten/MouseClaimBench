"""Validate exact high-level dependencies for hybrid-selective experiments."""

from __future__ import annotations

import json
import platform
from importlib.metadata import version


EXPECTED = {
    "causal-learn": "0.1.4.8",
    "scikit-learn": "1.9.0",
}


def main() -> None:
    observed = {package: version(package) for package in EXPECTED}
    mismatches = {
        package: {"expected": expected, "observed": observed[package]}
        for package, expected in EXPECTED.items()
        if observed[package] != expected
    }
    payload = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "expected": EXPECTED,
        "observed": observed,
        "valid": not mismatches,
        "mismatches": mismatches,
    }
    print(json.dumps(payload, indent=2))
    if mismatches:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

