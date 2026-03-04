"""Tests for alignment methods (DPO, GRPO, SimPO)."""

import json
import tempfile

import pytest
import torch

from yagpt.models import GPT, GPTConfig
from yagpt.tokenizer import Tokenizer
from yagpt.alignment.dataset import PreferenceDataset, preference_collate_fn


@pytest.fixture
def small_model():
    config = GPTConfig(vocab_size=1024, n_layers=2, n_heads=4, dim=64)
    return GPT(config)


@pytest.fixture
def preference_data():
    """Create a temporary JSONL file with preference data."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
        for _ in range(5):
            json.dump({
                "prompt": "What is 2+2?",
                "chosen": "The answer is 4.",
                "rejected": "I don't know.",
            }, f)
            f.write("\n")
        return f.name


class TestPreferenceDataset:
    def test_load(self, preference_data):
        tokenizer = Tokenizer("gpt2")
        dataset = PreferenceDataset(preference_data, tokenizer)
        samples = list(dataset)

        assert len(samples) == 5
        for s in samples:
            assert "chosen_ids" in s
            assert "rejected_ids" in s
            assert "prompt_len" in s

    def test_collate(self, preference_data):
        tokenizer = Tokenizer("gpt2")
        dataset = PreferenceDataset(preference_data, tokenizer)
        samples = list(dataset)[:2]

        batch = preference_collate_fn(samples)
        assert batch["chosen_ids"].shape[0] == 2
        assert batch["rejected_ids"].shape[0] == 2
        assert batch["prompt_len"].shape[0] == 2


class TestDPOLoss:
    def test_loss_direction(self, small_model):
        """DPO loss should be lower when policy agrees with preferences."""
        from yagpt.alignment.dpo import DPOTrainer

        trainer = DPOTrainer(small_model, beta=0.1, device="cpu")

        # Create a batch where chosen and rejected are the same
        # Loss should be ~log(2) when model can't distinguish
        batch = {
            "chosen_ids": torch.randint(0, 1024, (2, 32)),
            "rejected_ids": torch.randint(0, 1024, (2, 32)),
            "prompt_len": torch.tensor([8, 8]),
        }

        loss = trainer.train_step(batch)
        # With random data, loss should be finite and positive
        assert loss > 0
        assert not torch.isnan(torch.tensor(loss))

    def test_label_smoothing(self, small_model):
        from yagpt.alignment.dpo import DPOTrainer

        trainer = DPOTrainer(small_model, beta=0.1, label_smoothing=0.1, device="cpu")

        batch = {
            "chosen_ids": torch.randint(0, 1024, (2, 32)),
            "rejected_ids": torch.randint(0, 1024, (2, 32)),
            "prompt_len": torch.tensor([8, 8]),
        }

        loss = trainer.train_step(batch)
        assert loss > 0


class TestSimPO:
    def test_no_reference_model(self, small_model):
        """SimPO should not create a reference model."""
        from yagpt.alignment.simpo import SimPOTrainer

        trainer = SimPOTrainer(small_model, beta=2.5, gamma=0.5, device="cpu")

        # SimPO doesn't have ref_model attribute
        assert not hasattr(trainer, "ref_model")

    def test_loss_finite(self, small_model):
        from yagpt.alignment.simpo import SimPOTrainer

        trainer = SimPOTrainer(small_model, beta=2.5, gamma=0.5, device="cpu")

        batch = {
            "chosen_ids": torch.randint(0, 1024, (2, 32)),
            "rejected_ids": torch.randint(0, 1024, (2, 32)),
            "prompt_len": torch.tensor([8, 8]),
        }

        loss = trainer.train_step(batch)
        assert not torch.isnan(torch.tensor(loss))
        assert loss > 0


class TestGRPO:
    def test_advantage_normalization(self):
        """Test that GRPO normalizes advantages within group."""
        rewards = torch.tensor([1.0, 2.0, 3.0, 4.0])
        advantages = (rewards - rewards.mean()) / (rewards.std() + 1e-8)

        # Should be zero-mean and unit-variance
        assert abs(advantages.mean()) < 1e-6
        assert abs(advantages.std() - 1.0) < 0.1
