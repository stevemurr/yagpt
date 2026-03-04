"""
lm-eval-harness Integration.

Wraps the YAGPT model in the lm-eval-harness LM interface so it can be
evaluated on standard benchmarks (HellaSwag, ARC, MMLU, etc.).

Requires: pip install lm-eval
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch
from torch import Tensor

from yagpt.models import GPT, GPTConfig
from yagpt.tokenizer import Tokenizer

try:
    from lm_eval.api.instance import Instance
    from lm_eval.api.model import LM

    HAS_LM_EVAL = True
except ImportError:
    HAS_LM_EVAL = False
    LM = object  # Fallback for type hints

    class Instance:  # type: ignore[no-redef]
        """Stub for type hints when lm-eval is not installed."""
        args: tuple = ()


class YAGPTWrapper(LM):
    """
    Wraps a YAGPT model for use with lm-eval-harness.

    Implements the two core methods: loglikelihood and generate_until.
    """

    def __init__(
        self,
        model: GPT,
        tokenizer: Tokenizer,
        device: str = "cuda",
        batch_size: int = 16,
        max_length: int | None = None,
    ):
        if not HAS_LM_EVAL:
            raise ImportError("lm-eval is required: pip install lm-eval")
        super().__init__()
        self.model = model
        self.model.eval()
        self.tokenizer = tokenizer
        self._device = torch.device(device)
        self._batch_size = batch_size
        self._max_length = max_length or model.config.max_seq_len

    @property
    def eot_token_id(self) -> int:
        return self.tokenizer.eot_token

    @property
    def max_length(self) -> int:
        return self._max_length

    @property
    def max_gen_toks(self) -> int:
        return 256

    @property
    def batch_size(self) -> int:
        return self._batch_size

    @property
    def device(self) -> torch.device:
        return self._device

    def tok_encode(self, string: str) -> list[int]:
        return self.tokenizer.encode(string)

    def tok_decode(self, tokens: list[int]) -> str:
        return self.tokenizer.decode(tokens)

    def loglikelihood(self, requests: list[Instance]) -> list[tuple[float, bool]]:
        """Compute log-likelihood of continuations given contexts."""
        results = []
        for req in requests:
            context, continuation = req.args
            ctx_tokens = self.tok_encode(context)
            cont_tokens = self.tok_encode(continuation)
            all_tokens = ctx_tokens + cont_tokens

            # Truncate from the left if too long
            if len(all_tokens) > self.max_length:
                all_tokens = all_tokens[-self.max_length:]
                cont_tokens = cont_tokens[-(len(all_tokens) - len(ctx_tokens)):]

            input_ids = torch.tensor([all_tokens[:-1]], device=self.device)
            target_ids = torch.tensor([all_tokens[1:]], device=self.device)

            with torch.no_grad(), torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                logits, _, _ = self.model(input_ids)

            log_probs = torch.nn.functional.log_softmax(logits, dim=-1)

            # Sum log-probs over continuation tokens
            cont_len = len(cont_tokens)
            cont_log_probs = log_probs[0, -cont_len:, :]
            cont_targets = target_ids[0, -cont_len:]
            token_log_probs = cont_log_probs.gather(1, cont_targets.unsqueeze(1)).squeeze(1)

            total_ll = token_log_probs.sum().item()
            is_greedy = (logits[0, -cont_len:].argmax(dim=-1) == cont_targets).all().item()

            results.append((total_ll, bool(is_greedy)))

        return results

    def loglikelihood_rolling(self, requests: list[Instance]) -> list[tuple[float, bool]]:
        """Compute rolling log-likelihood (for perplexity-style evals)."""
        results = []
        for req in requests:
            (text,) = req.args
            tokens = self.tok_encode(text)
            if len(tokens) > self.max_length:
                tokens = tokens[:self.max_length]

            input_ids = torch.tensor([tokens[:-1]], device=self.device)
            target_ids = torch.tensor([tokens[1:]], device=self.device)

            with torch.no_grad(), torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                logits, _, _ = self.model(input_ids)

            log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
            token_log_probs = log_probs.gather(2, target_ids.unsqueeze(2)).squeeze(2)
            total_ll = token_log_probs.sum().item()
            is_greedy = (logits.argmax(dim=-1) == target_ids).all().item()

            results.append((total_ll, bool(is_greedy)))

        return results

    def generate_until(self, requests: list[Instance]) -> list[str]:
        """Generate text until a stop sequence is encountered."""
        results = []
        for req in requests:
            context, gen_kwargs = req.args
            stop = gen_kwargs.get("until", [])
            max_tokens = gen_kwargs.get("max_gen_toks", self.max_gen_toks)

            tokens = self.tok_encode(context)
            if len(tokens) > self.max_length - max_tokens:
                tokens = tokens[-(self.max_length - max_tokens):]

            input_ids = torch.tensor([tokens], device=self.device)

            with torch.no_grad():
                output = self.model.generate(
                    input_ids, max_new_tokens=max_tokens, temperature=0.0, top_k=1,
                )

            generated = self.tok_decode(output[0, len(tokens):].tolist())

            # Truncate at first stop sequence
            for s in stop:
                idx = generated.find(s)
                if idx >= 0:
                    generated = generated[:idx]
                    break

            results.append(generated)

        return results


def evaluate_model(
    model: GPT,
    tokenizer: Tokenizer,
    tasks: list[str],
    device: str = "cuda",
    batch_size: int = 16,
    num_fewshot: int | None = None,
) -> dict:
    """
    Evaluate a model on lm-eval-harness tasks.

    Args:
        model: Trained GPT model
        tokenizer: Tokenizer
        tasks: List of task names (e.g., ["hellaswag", "arc_easy"])
        device: Device to run on
        batch_size: Evaluation batch size
        num_fewshot: Number of few-shot examples (None = task default)

    Returns:
        Results dictionary from lm-eval-harness
    """
    import lm_eval

    wrapper = YAGPTWrapper(model, tokenizer, device=device, batch_size=batch_size)

    results = lm_eval.simple_evaluate(
        model=wrapper,
        tasks=tasks,
        num_fewshot=num_fewshot,
        batch_size=batch_size,
    )

    return results
