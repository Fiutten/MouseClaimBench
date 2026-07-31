"""Validate AdvisorSwitchTask with fixed memory-free and memory-based references."""

from cognitive_organism.evaluation.advisor_task_gate import evaluate_reference


def main() -> None:
    print("reference,mean_return,pre_switch_accuracy,post_switch_accuracy,early_post_accuracy")
    for name in ("random", "always_zero", "win_stay_lose_shift"):
        summary = evaluate_reference(name, episodes=500, seed=20260612)
        print(
            f"{name},{summary.mean_return:.4f},{summary.pre_switch_accuracy:.4f},"
            f"{summary.post_switch_accuracy:.4f},{summary.early_post_switch_accuracy:.4f}"
        )


if __name__ == "__main__":
    main()
