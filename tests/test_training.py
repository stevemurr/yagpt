"""Tests for training components."""

import pytest

from yagpt.optim import Muon, WarmupCosineSchedule, get_lr_scheduler
from yagpt.training import TrainConfig


class TestTrainConfig:
    def test_default_config(self):
        cfg = TrainConfig()

        assert cfg.n_layers == 12
        assert cfg.batch_size == 32
        assert cfg.grad_accum_steps > 0

    def test_grad_accum_calculation(self):
        cfg = TrainConfig(
            batch_size=8,
            max_seq_len=1024,
            total_batch_size=8 * 1024 * 4,  # 4 accumulation steps
        )

        assert cfg.grad_accum_steps == 4

    def test_invalid_batch_size(self):
        with pytest.raises(ValueError):
            TrainConfig(
                batch_size=7,  # Doesn't divide evenly
                max_seq_len=1024,
                total_batch_size=10000,
            )


class TestLRSchedule:
    def test_warmup_cosine(self):
        schedule = WarmupCosineSchedule(
            max_lr=1e-3,
            min_lr=1e-5,
            warmup_steps=100,
            total_steps=1000,
        )

        # During warmup, LR should increase
        assert schedule(0) < schedule(50) < schedule(100)

        # At warmup end, should be at max
        assert abs(schedule(100) - 1e-3) < 1e-6

        # After warmup, should decay
        assert schedule(500) < schedule(100)

        # At end, should be at min
        assert abs(schedule(1000) - 1e-5) < 1e-6

    def test_get_lr_scheduler_factory(self):
        schedule = get_lr_scheduler(
            "warmup_cosine",
            max_lr=1e-3,
            min_lr=1e-5,
            total_steps=1000,
            warmup_steps=100,
        )

        assert callable(schedule)
        assert schedule(0) < schedule(100)


class TestMuon:
    def test_muon_creation(self):
        import torch

        params = [torch.randn(10, 10, requires_grad=True)]
        opt = Muon(params, lr=0.02)

        assert len(opt.param_groups) == 1

    def test_muon_step(self):
        import torch

        param = torch.randn(10, 10, requires_grad=True)
        opt = Muon([param], lr=0.02)

        # Simulate gradient
        param.grad = torch.randn_like(param)
        initial_param = param.clone()

        opt.step()

        # Parameter should have changed
        assert not torch.equal(param, initial_param)
