"""Train PPO and RecurrentPPO on the isolated memory diagnostic."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from cognitive_organism.envs.advisor_switch_task import AdvisorSwitchTask
from cognitive_organism.evaluation.advisor_memory_gate import evaluate_advisor_memory_model
from cognitive_organism.learning.baselines import (
    BaselineKind,
    build_baseline,
    count_trainable_parameters,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=30_000)
    parser.add_argument("--episodes", type=int, default=200)
    parser.add_argument("--seeds", type=int, nargs="+", default=[11, 23, 37])
    parser.add_argument("--output", type=Path, default=Path("runs/advisor_memory_gate"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, int | float | str]] = []

    for kind in BaselineKind:
        for seed in args.seeds:
            model = build_baseline(kind, seed=seed, env=AdvisorSwitchTask())
            parameter_count = count_trainable_parameters(model)
            model.learn(total_timesteps=args.steps, progress_bar=False)
            model.save(args.output / f"{kind.value}_seed_{seed}")
            summary = evaluate_advisor_memory_model(
                model,
                algorithm=kind,
                train_seed=seed,
                train_steps=args.steps,
                trainable_parameters=parameter_count,
                episodes=args.episodes,
            )
            rows.append(summary.to_dict())
            print(json.dumps(summary.to_dict(), sort_keys=True))

    with (args.output / "results.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
