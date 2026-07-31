"""Run the first scientific gate before training any learning agent."""

from cognitive_organism.evaluation import evaluate_environment_gate


def main() -> None:
    summaries = evaluate_environment_gate(episodes=250, seed=20260612)
    print("policy,mean_return,mean_danger,mean_danger_rate,mean_resources,mean_steps")
    for name, summary in summaries.items():
        print(
            f"{name},{summary.mean_return:.4f},"
            f"{summary.mean_danger:.4f},{summary.mean_danger_rate:.4f},"
            f"{summary.mean_resources:.4f},{summary.mean_steps:.4f}"
        )


if __name__ == "__main__":
    main()
