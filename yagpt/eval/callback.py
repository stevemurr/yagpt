"""
Benchmark Callback - Run lm-eval tasks during training.

Runs standard benchmarks at configurable intervals, logging results
to console and optionally to wandb.
"""

from typing import TYPE_CHECKING

from yagpt.training.callbacks import Callback, TrainState

if TYPE_CHECKING:
    from yagpt.training.trainer import Trainer


class BenchmarkCallback(Callback):
    """
    Runs lm-eval-harness benchmarks at regular intervals during training.

    Args:
        tasks: List of task names (e.g., ["hellaswag", "arc_easy"])
        interval: Run benchmarks every N steps
        batch_size: Evaluation batch size
        num_fewshot: Number of few-shot examples (None = task default)
    """

    def __init__(
        self,
        tasks: list[str],
        interval: int = 5000,
        batch_size: int = 16,
        num_fewshot: int | None = None,
    ):
        self.tasks = tasks
        self.interval = interval
        self.batch_size = batch_size
        self.num_fewshot = num_fewshot

    def on_step_end(self, trainer: "Trainer", state: TrainState) -> None:
        if state.step == 0 or state.step % self.interval != 0:
            return

        from yagpt.eval.harness import evaluate_model

        print(f"\nRunning benchmarks: {', '.join(self.tasks)}")

        base_model = (
            trainer.model._orig_mod
            if hasattr(trainer.model, "_orig_mod")
            else trainer.model
        )

        results = evaluate_model(
            model=base_model,
            tokenizer=trainer.tokenizer,
            tasks=self.tasks,
            device=trainer.config.device,
            batch_size=self.batch_size,
            num_fewshot=self.num_fewshot,
        )

        # Print results
        if "results" in results:
            for task_name, task_results in results["results"].items():
                metrics = {k: v for k, v in task_results.items() if isinstance(v, (int, float))}
                metrics_str = ", ".join(f"{k}={v:.4f}" for k, v in metrics.items())
                print(f"  {task_name}: {metrics_str}")

        print()
