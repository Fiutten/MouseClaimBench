"""Validate the redesigned two-adviser environment before retraining agents."""

from cognitive_organism.evaluation.inference_environment_gate import (
    evaluate_inference_environment_gate,
)


def main() -> None:
    summaries = evaluate_inference_environment_gate(episodes=250, seed=20260612)
    print("policy,mean_return,mean_danger_rate,mean_resources,mean_steps")
    for name, summary in summaries.items():
        print(
            f"{name},{summary.mean_return:.4f},{summary.mean_danger_rate:.4f},"
            f"{summary.mean_resources:.4f},{summary.mean_steps:.4f}"
        )


if __name__ == "__main__":
    main()
