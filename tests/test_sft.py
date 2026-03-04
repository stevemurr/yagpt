"""Tests for supervised fine-tuning components."""

import json
import tempfile

import pytest
import torch

from yagpt.tokenizer import Tokenizer
from yagpt.sft.chat_format import (
    FormattedSample,
    format_chat,
    setup_chat_tokenizer,
    IGNORE_INDEX,
)
from yagpt.sft.dataset import SFTDataset, sft_collate_fn
from yagpt.sft.config import SFTConfig


class TestChatFormat:
    @pytest.fixture
    def tokenizer(self):
        tok = Tokenizer("gpt2")
        return tok

    def test_setup_chat_tokenizer(self, tokenizer):
        im_start_id, im_end_id = setup_chat_tokenizer(tokenizer)
        assert im_start_id != im_end_id
        assert im_start_id >= 50257  # Beyond base vocab

    def test_format_basic_chat(self, tokenizer):
        im_start_id, im_end_id = setup_chat_tokenizer(tokenizer)

        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there"},
        ]

        sample = format_chat(
            messages, tokenizer, max_seq_len=2048,
            im_start_id=im_start_id, im_end_id=im_end_id,
        )

        assert sample is not None
        assert len(sample.input_ids) == len(sample.labels)
        assert len(sample.input_ids) > 0

    def test_loss_masking(self, tokenizer):
        im_start_id, im_end_id = setup_chat_tokenizer(tokenizer)

        messages = [
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Hello"},
        ]

        sample = format_chat(
            messages, tokenizer, max_seq_len=2048,
            im_start_id=im_start_id, im_end_id=im_end_id,
        )

        assert sample is not None
        # User turn should be fully masked
        # At least some labels should be IGNORE_INDEX (masked)
        assert IGNORE_INDEX in sample.labels
        # At least some labels should NOT be IGNORE_INDEX (assistant tokens)
        assert any(l != IGNORE_INDEX for l in sample.labels)

    def test_system_message_masked(self, tokenizer):
        im_start_id, im_end_id = setup_chat_tokenizer(tokenizer)

        messages = [
            {"role": "system", "content": "Be helpful"},
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Hello"},
        ]

        sample = format_chat(
            messages, tokenizer, max_seq_len=2048,
            im_start_id=im_start_id, im_end_id=im_end_id,
        )

        assert sample is not None
        # Count unmasked labels - should only be from assistant content + im_end
        unmasked = [l for l in sample.labels if l != IGNORE_INDEX]
        assert len(unmasked) > 0

    def test_truncation(self, tokenizer):
        im_start_id, im_end_id = setup_chat_tokenizer(tokenizer)

        messages = [
            {"role": "user", "content": "A" * 10000},
            {"role": "assistant", "content": "B" * 10000},
        ]

        sample = format_chat(
            messages, tokenizer, max_seq_len=128,
            im_start_id=im_start_id, im_end_id=im_end_id,
        )

        assert sample is not None
        assert len(sample.input_ids) == 128
        assert len(sample.labels) == 128

    def test_empty_messages(self, tokenizer):
        im_start_id, im_end_id = setup_chat_tokenizer(tokenizer)
        sample = format_chat(
            [], tokenizer, max_seq_len=2048,
            im_start_id=im_start_id, im_end_id=im_end_id,
        )
        assert sample is None


class TestSFTDataset:
    def test_load_jsonl(self):
        tokenizer = Tokenizer("gpt2")

        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            for _ in range(5):
                json.dump({
                    "messages": [
                        {"role": "user", "content": "What is 2+2?"},
                        {"role": "assistant", "content": "4"},
                    ]
                }, f)
                f.write("\n")
            f.flush()

            dataset = SFTDataset(f.name, tokenizer, max_seq_len=256)
            samples = list(dataset)

            assert len(samples) == 5
            assert all(isinstance(s, FormattedSample) for s in samples)


class TestSFTCollate:
    def test_padding(self):
        # Create samples of different lengths
        s1 = FormattedSample(input_ids=[1, 2, 3], labels=[-100, 4, 5])
        s2 = FormattedSample(input_ids=[1, 2, 3, 4, 5], labels=[-100, -100, 6, 7, 8])

        input_ids, labels = sft_collate_fn([s1, s2])

        assert input_ids.shape == (2, 5)
        assert labels.shape == (2, 5)
        # First sample should be padded
        assert input_ids[0, 3].item() == 0
        assert labels[0, 3].item() == IGNORE_INDEX


class TestSFTConfig:
    def test_defaults(self):
        cfg = SFTConfig()
        assert cfg.epochs == 3
        assert cfg.learning_rate == 2e-5
        assert cfg.batch_size == 4
