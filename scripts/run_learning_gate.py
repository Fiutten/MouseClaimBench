"""Train and evaluate the prespecified PPO/RecurrentPPO Gate 2 pilot."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from cognitive_organism.evaluation.learning_gate import evaluate_learning_model
from cognitive_organism.learning.baselines import (
    BaselineKind,
    build_baseline,
    count_trainable_parameters,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=30_000)
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--seeds", type=int, nargs="+", default=[11, 23, 37])
    parser.add_argument("--output", type=Path, default=Path("runs/gate2"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, int | float | str]] = []

    for kind in BaselineKind:
        for seed in args.seeds:
            model = build_baseline(kind, seed=seed)
            parameter_count = count_trainable_parameters(model)
            model.learn(total_timesteps=args.steps, progress_bar=False)
            model.save(args.output / f"{kind.value}_seed_{seed}")
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

    report_path = args.output / "results.csv"
    with report_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
