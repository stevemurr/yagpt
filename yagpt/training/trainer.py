"""
Trainer - Clean, minimal training loop.

Handles:
- Model initialization and compilation
- Optimizer setup (dual optimizer: Muon + AdamW)
- Training loop with gradient accumulation
- Checkpoint saving and loading
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from yagpt.models import GPT, GPTConfig
from yagpt.optim import Muon, get_lr_scheduler
from yagpt.tokenizer import Tokenizer

from .callbacks import Callback, TrainState
from .config import TrainConfig


class Trainer:
    """
    GPT Trainer.

    Handles model training with:
    - Dual optimizer (Muon for transformer, AdamW for embeddings)
    - Gradient accumulation
    - Learning rate scheduling
    - Callbacks for logging, checkpointing, evaluation
    """

    def __init__(
        self,
        config: TrainConfig,
        train_loader: DataLoader,
        val_loader: DataLoader | None = None,
        callbacks: list[Callback] | None = None,
    ):
        self.config = config
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.callbacks = callbacks or []

        # Initialize components
        self.tokenizer = Tokenizer(config.tokenizer)
        self.model = self._create_model()
        self.optimizers = self._create_optimizers()
        self.lr_schedule = self._create_lr_schedule()

        # Training state
        self.step = 0
        self.last_loss = 0.0
        self.current_lr = config.learning_rate

        # Load checkpoint if resuming
        if config.resume_from:
            self._load_checkpoint(config.resume_from)

    def _create_model(self) -> GPT:
        """Create and optionally compile the model."""
        model_config = GPTConfig(
            vocab_size=self.tokenizer.vocab_size,
            n_layers=self.config.n_layers,
            n_heads=self.config.n_heads,
            n_kv_heads=self.config.n_kv_heads,
            dim=self.config.dim,
            hidden_dim=self.config.hidden_dim,
            max_seq_len=self.config.max_seq_len,
        )

        model = GPT(model_config)
        model = model.to(self.config.device)

        print(f"Model: {model.num_parameters()/1e6:.1f}M parameters")

        if self.config.compile:
            print("Compiling model with torch.compile...")
            model = torch.compile(model)

        return model

    def _create_optimizers(self) -> list[torch.optim.Optimizer]:
        """Create optimizer(s) based on config."""
        # Get the underlying model if compiled
        base_model = self.model._orig_mod if hasattr(self.model, "_orig_mod") else self.model

        if self.config.optimizer == "adamw":
            # Simple AdamW for all parameters
            optimizer = torch.optim.AdamW(
                base_model.parameters(),
                lr=self.config.learning_rate,
                betas=(self.config.beta1, self.config.beta2),
                weight_decay=self.config.weight_decay,
                fused=True,
            )
            return [optimizer]

        elif self.config.optimizer == "muon":
            # Muon for all parameters
            optimizer = Muon(
                base_model.parameters(),
                lr=self.config.muon_lr,
            )
            return [optimizer]

        elif self.config.optimizer == "dual":
            # Dual optimizer: Muon for transformer, AdamW for embeddings
            # Separate parameter groups
            embedding_params = list(base_model.tok_emb.parameters())
            embedding_ids = {id(p) for p in embedding_params}

            # Transformer block parameters (for Muon)
            transformer_params = [
                p for p in base_model.blocks.parameters()
            ]

            # Norm and any other parameters (for AdamW)
            other_params = [
                p for p in base_model.parameters()
                if id(p) not in embedding_ids and p not in transformer_params
            ]

            # AdamW for embeddings and other params
            adamw = torch.optim.AdamW(
                [
                    {"params": embedding_params, "lr": self.config.learning_rate},
                    {"params": other_params, "lr": self.config.learning_rate},
                ],
                betas=(self.config.beta1, self.config.beta2),
                weight_decay=self.config.weight_decay,
                fused=True,
            )

            # Muon for transformer blocks
            muon = Muon(
                transformer_params,
                lr=self.config.muon_lr,
            )

            return [adamw, muon]

        else:
            raise ValueError(f"Unknown optimizer: {self.config.optimizer}")

    def _create_lr_schedule(self):
        """Create learning rate schedule."""
        return get_lr_scheduler(
            schedule=self.config.lr_schedule,
            max_lr=self.config.learning_rate,
            min_lr=self.config.min_lr,
            total_steps=self.config.max_steps,
            warmup_steps=self.config.warmup_steps,
        )

    def _load_checkpoint(self, path: str) -> None:
        """Load training state from checkpoint."""
        print(f"Loading checkpoint: {path}")
        checkpoint = torch.load(path, map_location=self.config.device)

        # Get base model if compiled
        base_model = self.model._orig_mod if hasattr(self.model, "_orig_mod") else self.model
        base_model.load_state_dict(checkpoint["model"])

        # Load optimizer states
        for opt, state_dict in zip(self.optimizers, checkpoint["optimizer"]):
            opt.load_state_dict(state_dict)

        self.step = checkpoint["step"]
        self.last_loss = checkpoint.get("loss", 0.0)
        print(f"Resumed from step {self.step}")

    def _update_lr(self) -> float:
        """Update learning rate based on schedule."""
        lr = self.lr_schedule(self.step)

        for optimizer in self.optimizers:
            for group in optimizer.param_groups:
                # Scale Muon LR proportionally
                if isinstance(optimizer, Muon):
                    group["lr"] = self.config.muon_lr * (lr / self.config.learning_rate)
                else:
                    group["lr"] = lr

        return lr

    def train_step(self, batch: tuple[torch.Tensor, torch.Tensor]) -> float:
        """
        Execute one training step with gradient accumulation.

        Args:
            batch: Tuple of (input_ids, targets) tensors

        Returns:
            Average loss over the batch
        """
        self.model.train()
        x, y = batch

        # Gradient accumulation
        total_loss = 0.0
        for micro_step in range(self.config.grad_accum_steps):
            # Get micro-batch slice
            start = micro_step * self.config.batch_size
            end = start + self.config.batch_size
            micro_x = x[start:end]
            micro_y = y[start:end]

            # Forward pass with autocast
            with torch.amp.autocast(device_type="cuda", dtype=self.config.torch_dtype):
                _, loss, _ = self.model(micro_x, micro_y)
                loss = loss / self.config.grad_accum_steps

            # Backward pass
            loss.backward()
            total_loss += loss.item()

        # Gradient clipping
        grad_norm = None
        if self.config.grad_clip > 0:
            base_model = self.model._orig_mod if hasattr(self.model, "_orig_mod") else self.model
            grad_norm = nn.utils.clip_grad_norm_(
                base_model.parameters(),
                self.config.grad_clip,
            ).item()

        # Optimizer step
        for optimizer in self.optimizers:
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        return total_loss, grad_norm

    @torch.no_grad()
    def evaluate(self, num_steps: int) -> float:
        """
        Run evaluation on validation data.

        Args:
            num_steps: Number of evaluation steps

        Returns:
            Average validation loss
        """
        if self.val_loader is None:
            return 0.0

        self.model.eval()
        total_loss = 0.0

        val_iter = iter(self.val_loader)
        for _ in range(num_steps):
            try:
                x, y = next(val_iter)
            except StopIteration:
                break

            x = x.to(self.config.device)
            y = y.to(self.config.device)

            with torch.amp.autocast(device_type="cuda", dtype=self.config.torch_dtype):
                _, loss, _ = self.model(x, y)

            total_loss += loss.item()

        self.model.train()
        return total_loss / num_steps

    def train(self) -> None:
        """Run the training loop."""
        print(f"\nStarting training from step {self.step}")
        print(f"  Total steps: {self.config.max_steps}")
        print(f"  Batch size: {self.config.batch_size}")
        print(f"  Gradient accumulation: {self.config.grad_accum_steps}")
        print(f"  Total batch size: {self.config.total_batch_size} tokens")
        print()

        # Notify callbacks
        for callback in self.callbacks:
            callback.on_train_start(self)

        # Training loop
        train_iter = iter(self.train_loader)

        while self.step < self.config.max_steps:
            # Get batch
            try:
                x, y = next(train_iter)
            except StopIteration:
                train_iter = iter(self.train_loader)
                x, y = next(train_iter)

            x = x.to(self.config.device)
            y = y.to(self.config.device)

            # Update learning rate
            self.current_lr = self._update_lr()

            # Training step
            loss, grad_norm = self.train_step((x, y))
            self.last_loss = loss
            self.step += 1

            # Create state for callbacks
            state = TrainState(
                step=self.step,
                loss=loss,
                lr=self.current_lr,
                grad_norm=grad_norm,
            )

            # Notify callbacks
            for callback in self.callbacks:
                callback.on_step_end(self, state)

        # Training complete
        for callback in self.callbacks:
            callback.on_train_end(self)

        print("\nTraining complete!")
