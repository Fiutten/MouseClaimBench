"""Run the sealed PPO control with previous action exposed in observation."""

from __future__ import annotations

import csv
import json
from pathlib import Path

from cognitive_organism.envs.markov_advisor_switch_task import MarkovAdvisorSwitchTask
from cognitive_organism.evaluation.advisor_memory_gate import evaluate_advisor_memory_model
from cognitive_organism.learning.baselines import (
    BaselineKind,
    build_baseline,
    count_trainable_parameters,
)


def main() -> None:
    seeds = [127, 131, 137, 139, 149]
    train_steps = 30_000
    episodes = 200
    output = Path("runs/markov_control_gate")
    output.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, int | float | str]] = []

    for seed in seeds:
        model = build_baseline(BaselineKind.PPO, seed=seed, env=MarkovAdvisorSwitchTask())
        parameters = count_trainable_parameters(model)
        model.learn(total_timesteps=train_steps, progress_bar=False)
        model.save(output / f"ppo_seed_{seed}")
        summary = evaluate_advisor_memory_model(
            model,
            algorithm=BaselineKind.PPO,
            train_seed=seed,
            train_steps=train_steps,
            trainable_parameters=parameters,
            episodes=episodes,
            env=MarkovAdvisorSwitchTask(),
        )
        row = {**summary.to_dict(), "passes_threshold": summary.mean_post_switch_accuracy > 0.90}
        rows.append(row)
        print(json.dumps(row, sort_keys=True))

    with (output / "results.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
