"""
GRPO - Group Relative Policy Optimization.

Samples G completions per prompt, scores them with a reward function,
normalizes advantages within the group, and applies a PPO-style clipped
objective. No critic model needed.

Reference: https://arxiv.org/abs/2402.03300
"""

from pathlib import Path
from typing import Callable

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from yagpt.models import GPT
from yagpt.training.callbacks import Callback, TrainState


class GRPOTrainer:
    """
    Group Relative Policy Optimization trainer.

    For each prompt, samples G completions, scores them, normalizes
    advantages within the group, and updates with clipped PPO loss.

    Args:
        model: Policy model to train
        reward_fn: Callable(prompt_tokens, completion_tokens) -> float
        group_size: Number of completions to sample per prompt
        beta: KL penalty coefficient
        clip_eps: PPO clipping epsilon
        lr: Learning rate
        max_steps: Maximum training steps
        device: Device
        max_gen_tokens: Max tokens to generate per completion
    """

    def __init__(
        self,
        model: GPT,
        reward_fn: Callable[[list[int], list[int]], float],
        group_size: int = 4,
        beta: float = 0.04,
        clip_eps: float = 0.2,
        lr: float = 1e-6,
        max_steps: int = 1000,
        device: str = "cuda",
        max_gen_tokens: int = 256,
        callbacks: list[Callback] | None = None,
    ):
        self.device = device
        self.model = model.to(device)
        self.reward_fn = reward_fn
        self.group_size = group_size
        self.beta = beta
        self.clip_eps = clip_eps
        self.max_steps = max_steps
        self.max_gen_tokens = max_gen_tokens
        self.callbacks = callbacks or []

        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr)
        self.step = 0

    def _sample_completions(
        self, prompt_ids: torch.Tensor,
    ) -> list[torch.Tensor]:
        """Sample G completions for a prompt."""
        self.model.eval()
        completions = []

        for _ in range(self.group_size):
            with torch.no_grad():
                output = self.model.generate(
                    prompt_ids.unsqueeze(0),
                    max_new_tokens=self.max_gen_tokens,
                    temperature=0.8,
                    top_k=50,
                )
            completions.append(output[0])

        self.model.train()
        return completions

    def _compute_log_probs(
        self, model: GPT, input_ids: torch.Tensor, prompt_len: int,
    ) -> torch.Tensor:
        """Compute sum of log-probs for completion tokens."""
        logits, _, _ = model(input_ids.unsqueeze(0), return_logits=True)
        log_probs = F.log_softmax(logits, dim=-1)

        shifted_log_probs = log_probs[0, :-1, :]
        shifted_targets = input_ids[1:]

        token_log_probs = shifted_log_probs.gather(
            1, shifted_targets.unsqueeze(1)
        ).squeeze(1)

        # Only completion tokens
        completion_log_probs = token_log_probs[prompt_len - 1:]
        return completion_log_probs.sum()

    def train_step(self, prompt_ids: torch.Tensor) -> float:
        """Execute one GRPO training step for a single prompt."""
        prompt_ids = prompt_ids.to(self.device)
        prompt_len = prompt_ids.shape[0]

        # Sample G completions
        completions = self._sample_completions(prompt_ids)

        # Score completions
        rewards = []
        for comp in completions:
            prompt_tokens = comp[:prompt_len].tolist()
            completion_tokens = comp[prompt_len:].tolist()
            reward = self.reward_fn(prompt_tokens, completion_tokens)
            rewards.append(reward)

        rewards_t = torch.tensor(rewards, device=self.device, dtype=torch.float32)

        # Normalize advantages within group
        if rewards_t.std() > 1e-8:
            advantages = (rewards_t - rewards_t.mean()) / (rewards_t.std() + 1e-8)
        else:
            advantages = torch.zeros_like(rewards_t)

        # Compute old log-probs (before update)
        old_log_probs = []
        for comp in completions:
            with torch.no_grad():
                lp = self._compute_log_probs(self.model, comp, prompt_len)
            old_log_probs.append(lp)
        old_log_probs = torch.stack(old_log_probs)

        # Policy gradient with PPO clipping
        total_loss = 0.0
        for i, comp in enumerate(completions):
            new_log_prob = self._compute_log_probs(self.model, comp, prompt_len)
            ratio = torch.exp(new_log_prob - old_log_probs[i].detach())

            # Clipped surrogate
            clipped_ratio = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps)
            surr1 = ratio * advantages[i]
            surr2 = clipped_ratio * advantages[i]
            total_loss -= torch.min(surr1, surr2)

        loss = total_loss / self.group_size
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)

        return loss.item()

    def train(self, dataloader: DataLoader) -> None:
        """Run the GRPO training loop."""
        print(f"\nStarting GRPO training (G={self.group_size})")

        for callback in self.callbacks:
            callback.on_train_start(self)

        data_iter = iter(dataloader)
        while self.step < self.max_steps:
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(dataloader)
                batch = next(data_iter)

            # Process one prompt at a time for GRPO
            # Extract prompt tokens (without padding) using prompt_len
            prompt_len = batch["prompt_len"][0].item()
            prompt_ids = batch["prompt_ids"][0][:prompt_len]
            loss = self.train_step(prompt_ids)
            self.step += 1

            state = TrainState(step=self.step, loss=loss, lr=self.optimizer.param_groups[0]["lr"])
            for callback in self.callbacks:
                callback.on_step_end(self, state)

        for callback in self.callbacks:
            callback.on_train_end(self)

        print("GRPO training complete!")

    def save(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "step": self.step,
            "model": self.model.state_dict(),
            "config": {"group_size": self.group_size, "beta": self.beta, "method": "grpo"},
        }, path)
