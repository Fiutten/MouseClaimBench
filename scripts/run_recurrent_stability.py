"""Develop or confirm a stable RecurrentPPO baseline using frozen protocols."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from cognitive_organism.envs.advisor_switch_task import AdvisorSwitchTask
from cognitive_organism.evaluation.advisor_memory_gate import evaluate_advisor_memory_model
from cognitive_organism.evaluation.recurrent_stability import score_configurations
from cognitive_organism.learning.baselines import (
    RECURRENT_CONFIGS,
    BaselineKind,
    build_baseline,
    count_trainable_parameters,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("develop", "confirm"), required=True)
    parser.add_argument("--steps", type=int, default=30_000)
    parser.add_argument("--episodes", type=int, default=200)
    parser.add_argument("--config", choices=tuple(RECURRENT_CONFIGS))
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.mode == "develop":
        config_names = list(RECURRENT_CONFIGS)
        seeds = [41, 43, 47]
    else:
        if args.config is None:
            raise SystemExit("--config is required in confirm mode")
        config_names = [args.config]
        seeds = [101, 103, 107, 109, 113]

    args.output.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, int | float | str]] = []
    for config_name in config_names:
        config = RECURRENT_CONFIGS[config_name]
        for seed in seeds:
            model = build_baseline(
                BaselineKind.RECURRENT_PPO,
                seed=seed,
                env=AdvisorSwitchTask(),
                recurrent_config=config,
            )
            parameters = count_trainable_parameters(model)
            model.learn(total_timesteps=args.steps, progress_bar=False)
            model.save(args.output / f"{config_name}_seed_{seed}")
            summary = evaluate_advisor_memory_model(
                model,
                algorithm=BaselineKind.RECURRENT_PPO,
                train_seed=seed,
                train_steps=args.steps,
                trainable_parameters=parameters,
                episodes=args.episodes,
            )
            row = {"config_name": config_name, **summary.to_dict()}
            rows.append(row)
            print(json.dumps(row, sort_keys=True))

    with (args.output / "results.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    scores = score_configurations(rows)
    with (args.output / "scores.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(scores[0].__dict__))
        writer.writeheader()
        writer.writerows(score.__dict__ for score in scores)
    print("selection_order=" + ",".join(score.config_name for score in scores))


if __name__ == "__main__":
    main()
