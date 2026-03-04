"""
DPO - Direct Preference Optimization.

Trains a policy to prefer chosen over rejected completions without a
separate reward model. Uses a frozen reference copy of the policy.

loss = -log σ(β * (log π(chosen)/π_ref(chosen) - log π(rejected)/π_ref(rejected)))

Reference: https://arxiv.org/abs/2305.18290
"""

import copy
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from yagpt.models import GPT
from yagpt.training.callbacks import Callback, TrainState


class DPOTrainer:
    """
    Direct Preference Optimization trainer.

    Args:
        model: Policy model to train
        beta: Temperature parameter (0.1-0.5 typical)
        lr: Learning rate (very low, e.g., 5e-7)
        max_steps: Maximum training steps
        device: Device to train on
        label_smoothing: Label smoothing factor (0 = none)
        length_normalize: Whether to normalize log-probs by length
    """

    def __init__(
        self,
        model: GPT,
        beta: float = 0.1,
        lr: float = 5e-7,
        max_steps: int = 1000,
        device: str = "cuda",
        label_smoothing: float = 0.0,
        length_normalize: bool = True,
        callbacks: list[Callback] | None = None,
    ):
        self.device = device
        self.beta = beta
        self.max_steps = max_steps
        self.label_smoothing = label_smoothing
        self.length_normalize = length_normalize
        self.callbacks = callbacks or []

        # Policy model
        self.model = model.to(device)

        # Reference model (frozen deepcopy)
        self.ref_model = copy.deepcopy(model).to(device)
        self.ref_model.eval()
        for p in self.ref_model.parameters():
            p.requires_grad_(False)

        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr)
        self.step = 0

    def _get_log_probs(
        self, model: GPT, input_ids: torch.Tensor, prompt_len: torch.Tensor,
    ) -> torch.Tensor:
        """Compute per-token log-probs for completion tokens only."""
        logits, _, _ = model(input_ids, return_logits=True)
        log_probs = F.log_softmax(logits, dim=-1)

        # Shift: log_prob of token t is at position t-1
        shifted_log_probs = log_probs[:, :-1, :]
        shifted_targets = input_ids[:, 1:]

        # Gather log-probs of actual tokens
        token_log_probs = shifted_log_probs.gather(
            2, shifted_targets.unsqueeze(2)
        ).squeeze(2)

        # Mask: only count completion tokens (after prompt)
        batch_size, seq_len = shifted_targets.shape
        positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)
        mask = positions >= (prompt_len.unsqueeze(1) - 1)  # -1 for shift

        # Zero out prompt tokens
        token_log_probs = token_log_probs * mask.to(token_log_probs.dtype)

        if self.length_normalize:
            completion_len = mask.sum(dim=1).clamp(min=1)
            return token_log_probs.sum(dim=1) / completion_len
        else:
            return token_log_probs.sum(dim=1)

    def train_step(self, batch: dict[str, torch.Tensor]) -> float:
        """Execute one DPO training step."""
        self.model.train()

        chosen_ids = batch["chosen_ids"].to(self.device)
        rejected_ids = batch["rejected_ids"].to(self.device)
        prompt_len = batch["prompt_len"].to(self.device)

        # Policy log-probs
        pi_chosen = self._get_log_probs(self.model, chosen_ids, prompt_len)
        pi_rejected = self._get_log_probs(self.model, rejected_ids, prompt_len)

        # Reference log-probs
        with torch.no_grad():
            ref_chosen = self._get_log_probs(self.ref_model, chosen_ids, prompt_len)
            ref_rejected = self._get_log_probs(self.ref_model, rejected_ids, prompt_len)

        # DPO loss
        log_ratio_chosen = pi_chosen - ref_chosen
        log_ratio_rejected = pi_rejected - ref_rejected
        logits = self.beta * (log_ratio_chosen - log_ratio_rejected)

        if self.label_smoothing > 0:
            loss = (
                (1 - self.label_smoothing) * F.logsigmoid(logits)
                + self.label_smoothing * F.logsigmoid(-logits)
            )
            loss = -loss.mean()
        else:
            loss = -F.logsigmoid(logits).mean()

        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)

        return loss.item()

    def train(self, dataloader: DataLoader) -> None:
        """Run the DPO training loop."""
        print(f"\nStarting DPO training (β={self.beta})")

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

        print("DPO training complete!")

    def save(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "step": self.step,
            "model": self.model.state_dict(),
            "config": {"beta": self.beta, "method": "dpo"},
        }, path)
