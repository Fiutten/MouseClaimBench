"""Re-evaluate saved Gate 2 models with additional policy-collapse diagnostics."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from cognitive_organism.evaluation.learning_gate import evaluate_learning_model
from cognitive_organism.learning.baselines import BaselineKind, load_baseline


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=Path("runs/gate2_preregistered"))
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--steps", type=int, default=30_000)
    args = parser.parse_args()

    rows: list[dict[str, int | float | str]] = []
    for kind in BaselineKind:
        for path in sorted(args.input.glob(f"{kind.value}_seed_*.zip")):
            seed = int(path.stem.rsplit("_", maxsplit=1)[-1])
            model = load_baseline(kind, str(path))
            parameter_count = sum(
                parameter.numel()
                for parameter in model.policy.parameters()
                if parameter.requires_grad
            )
            summary = evaluate_learning_model(
                model,
                algorithm=kind,
                train_seed=seed,
                train_steps=args.steps,
                trainable_parameters=parameter_count,
                episodes=args.episodes,
                evaluation_seed=100_000,
            )
            rows.append(summary.to_dict())
            print(json.dumps(summary.to_dict(), sort_keys=True))

    report_path = args.input / "diagnostic_results.csv"
    with report_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
