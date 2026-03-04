"""
SimPO - Simple Preference Optimization.

Reference-model-free alignment. Uses length-normalized log-probability
as an implicit reward, with a margin term for separation.

loss = -log σ(β * (r_chosen - r_rejected - γ))
where r = (1/|y|) * log π(y|x)

Reference: https://arxiv.org/abs/2405.14734
"""

from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from yagpt.models import GPT
from yagpt.training.callbacks import Callback, TrainState


class SimPOTrainer:
    """
    Simple Preference Optimization trainer.

    No reference model needed. Uses length-normalized log-prob as implicit
    reward with a target reward margin between chosen and rejected.

    Args:
        model: Policy model to train
        beta: Temperature (much larger than DPO, typically 2.0-2.5)
        gamma: Target reward margin (0.5-1.0 typical)
        lr: Learning rate
        max_steps: Maximum training steps
        device: Device
    """

    def __init__(
        self,
        model: GPT,
        beta: float = 2.5,
        gamma: float = 0.5,
        lr: float = 5e-7,
        max_steps: int = 1000,
        device: str = "cuda",
        callbacks: list[Callback] | None = None,
    ):
        self.device = device
        self.beta = beta
        self.gamma = gamma
        self.max_steps = max_steps
        self.callbacks = callbacks or []

        self.model = model.to(device)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr)
        self.step = 0

    def _get_reward(
        self, input_ids: torch.Tensor, prompt_len: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute implicit reward: length-normalized log-prob of completion.

        r(x, y) = (1/|y|) * Σ log π(y_t | x, y_{<t})
        """
        logits, _, _ = self.model(input_ids, return_logits=True)
        log_probs = F.log_softmax(logits, dim=-1)

        shifted_log_probs = log_probs[:, :-1, :]
        shifted_targets = input_ids[:, 1:]

        token_log_probs = shifted_log_probs.gather(
            2, shifted_targets.unsqueeze(2)
        ).squeeze(2)

        # Mask prompt tokens
        batch_size, seq_len = shifted_targets.shape
        positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)
        mask = positions >= (prompt_len.unsqueeze(1) - 1)

        token_log_probs = token_log_probs * mask.to(token_log_probs.dtype)

        # Length-normalized reward
        completion_len = mask.sum(dim=1).clamp(min=1)
        return token_log_probs.sum(dim=1) / completion_len

    def train_step(self, batch: dict[str, torch.Tensor]) -> float:
        """Execute one SimPO training step."""
        self.model.train()

        chosen_ids = batch["chosen_ids"].to(self.device)
        rejected_ids = batch["rejected_ids"].to(self.device)
        prompt_len = batch["prompt_len"].to(self.device)

        # Compute implicit rewards
        r_chosen = self._get_reward(chosen_ids, prompt_len)
        r_rejected = self._get_reward(rejected_ids, prompt_len)

        # SimPO loss with margin
        logits = self.beta * (r_chosen - r_rejected - self.gamma)
        loss = -F.logsigmoid(logits).mean()

        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)

        return loss.item()

    def train(self, dataloader: DataLoader) -> None:
        """Run the SimPO training loop."""
        print(f"\nStarting SimPO training (β={self.beta}, γ={self.gamma})")

        for callback in self.callbacks:
            callback.on_train_start(self)

        data_iter = iter(dataloader)
        while self.step < self.max_steps:
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(dataloader)
                batch = next(data_iter)

            loss = self.train_step(batch)
            self.step += 1

            state = TrainState(step=self.step, loss=loss, lr=self.optimizer.param_groups[0]["lr"])
            for callback in self.callbacks:
                callback.on_step_end(self, state)

        for callback in self.callbacks:
            callback.on_train_end(self)

        print("SimPO training complete!")

    def save(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "step": self.step,
            "model": self.model.state_dict(),
            "config": {"beta": self.beta, "gamma": self.gamma, "method": "simpo"},
        }, path)
