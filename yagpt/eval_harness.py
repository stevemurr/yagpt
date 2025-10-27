"""
Evaluation wrapper for lm-evaluation-harness.

Provides a custom model wrapper that makes YAGPT compatible with
EleutherAI's lm-evaluation-harness for running standard benchmarks
like HellaSwag, MMLU, ARC, etc.
"""

import torch
import torch.nn.functional as F
from typing import List, Tuple, Optional
from lm_eval.api.model import LM
from lm_eval import evaluator

from yagpt.model import GPT
from yagpt.tokenizer import GPT4Tokenizer


class YAGPTEvalWrapper(LM):
    """
    Wrapper class to make YAGPT compatible with lm-evaluation-harness.

    This allows us to run standard benchmarks (HellaSwag, MMLU, etc.)
    without converting our model to HuggingFace format.
    """

    def __init__(
        self,
        model: GPT,
        tokenizer: GPT4Tokenizer,
        device: str = "cuda",
        batch_size: int = 1,
    ):
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer
        self._device = device
        self._batch_size = batch_size

        # Put model in eval mode
        self.model.eval()

        # Get vocab size
        self.vocab_size = tokenizer.vocab_size

        # EOT token (end of text) - tiktoken uses token 100257 for <|endoftext|>
        # For cl100k_base encoding (GPT-4), the EOT token is at index 100257
        self.eot_token_id = 100257

    @property
    def batch_size(self) -> int:
        """Maximum batch size for evaluation."""
        return self._batch_size

    @property
    def device(self) -> str:
        """Device for evaluation."""
        return self._device

    @property
    def eot_token(self) -> int:
        """End of text token ID."""
        return self.eot_token_id

    @property
    def max_length(self) -> int:
        """Maximum sequence length the model can handle."""
        return self.model.config.block_size

    def tok_encode(self, string: str, **kwargs) -> List[int]:
        """
        Encode a string into tokens.

        Args:
            string: Text to encode

        Returns:
            List of token IDs
        """
        return self.tokenizer.encode(string)

    def tok_decode(self, tokens: List[int], **kwargs) -> str:
        """
        Decode tokens into a string.

        Args:
            tokens: List of token IDs

        Returns:
            Decoded text
        """
        return self.tokenizer.decode(tokens)

    def loglikelihood(
        self,
        requests
    ) -> List[Tuple[float, bool]]:
        """
        Compute log-likelihood of continuations given contexts.

        This is the main method used by most benchmarks like HellaSwag.

        Args:
            requests: List of Instance objects with arguments=(context, continuation)

        Returns:
            List of (log_likelihood, is_greedy) tuples where:
            - log_likelihood: Log probability of the continuation
            - is_greedy: Whether continuation would be generated greedily
        """
        results = []

        # Process in batches
        for i in range(0, len(requests), self.batch_size):
            batch = requests[i:i + self.batch_size]
            batch_results = self._loglikelihood_batch(batch)
            results.extend(batch_results)

        return results

    def _loglikelihood_batch(
        self,
        batch
    ) -> List[Tuple[float, bool]]:
        """Process a single batch for loglikelihood computation."""
        # Encode contexts and continuations
        contexts = []
        continuations = []

        for instance in batch:
            # Extract context and continuation from Instance.arguments
            context, continuation = instance.args
            ctx_tokens = self.tok_encode(context)
            cont_tokens = self.tok_encode(continuation)

            contexts.append(ctx_tokens)
            continuations.append(cont_tokens)

        # Find max length in batch
        max_len = max(
            len(ctx) + len(cont)
            for ctx, cont in zip(contexts, continuations)
        )
        max_len = min(max_len, self.max_length)

        # Prepare batch tensors
        batch_size = len(batch)
        input_ids = torch.zeros((batch_size, max_len), dtype=torch.long, device=self.device)

        # Track where continuation starts for each sample
        cont_starts = []
        cont_ends = []

        for idx, (ctx, cont) in enumerate(zip(contexts, continuations)):
            # Concatenate context + continuation
            full_seq = ctx + cont

            # Truncate if needed (from the left/context)
            if len(full_seq) > max_len:
                # Keep the continuation, truncate context
                trunc_amount = len(full_seq) - max_len
                full_seq = full_seq[trunc_amount:]
                ctx = ctx[trunc_amount:]

            # Fill input tensor
            seq_len = len(full_seq)
            input_ids[idx, :seq_len] = torch.tensor(full_seq, dtype=torch.long)

            # Track continuation boundaries
            cont_start = len(ctx)
            cont_end = len(full_seq)
            cont_starts.append(cont_start)
            cont_ends.append(cont_end)

        # Forward pass to get logits
        with torch.no_grad():
            logits, _, _ = self.model(input_ids)  # [batch_size, seq_len, vocab_size]

        # Compute log-likelihoods for continuations only
        results = []

        for idx in range(batch_size):
            cont_start = cont_starts[idx]
            cont_end = cont_ends[idx]

            if cont_start >= cont_end:
                # Empty continuation
                results.append((0.0, True))
                continue

            # Get logits for continuation (shifted by 1 for next-token prediction)
            # We want P(token_i | tokens_0...i-1)
            # For each token at position i, we use logits from position i-1
            if cont_start == 0:
                # If continuation starts at position 0, we can't predict it
                results.append((0.0, True))
                continue

            cont_logits = logits[idx, cont_start-1:cont_end-1, :]  # [cont_len, vocab_size]
            cont_tokens = input_ids[idx, cont_start:cont_end]  # [cont_len]

            # Check for empty continuation after slicing
            if cont_logits.size(0) == 0 or len(cont_tokens) == 0:
                results.append((0.0, True))
                continue

            # Compute log probabilities
            log_probs = F.log_softmax(cont_logits, dim=-1)

            # Get log prob of actual tokens
            token_log_probs = log_probs[range(len(cont_tokens)), cont_tokens]

            # Sum log probs (product in prob space = sum in log space)
            total_log_prob = token_log_probs.sum().item()

            # Check if greedy (would model generate this?)
            greedy_tokens = cont_logits.argmax(dim=-1)
            is_greedy = torch.all(greedy_tokens == cont_tokens).item()

            results.append((total_log_prob, is_greedy))

        return results

    def loglikelihood_rolling(
        self,
        requests
    ) -> List[float]:
        """
        Compute rolling log-likelihood (for perplexity evaluation).

        Not commonly used for benchmarks, but required by interface.
        """
        raise NotImplementedError("loglikelihood_rolling not implemented")

    def generate_until(
        self,
        requests
    ) -> List[str]:
        """
        Generate text until stopping criteria.

        Used by some generative benchmarks.
        """
        results = []

        for instance in requests:
            context = instance.args[0]
            gen_kwargs = instance.args[1] if len(instance.args) > 1 else {}
            # Encode context
            tokens = self.tok_encode(context)
            input_ids = torch.tensor([tokens], dtype=torch.long, device=self.device)

            # Extract generation parameters
            max_tokens = gen_kwargs.get("max_gen_toks", 100)
            temperature = gen_kwargs.get("temperature", 0.0)  # Greedy by default

            # Generate
            with torch.no_grad():
                generated = self.model.generate(
                    input_ids,
                    max_new_tokens=max_tokens,
                    temperature=max(temperature, 0.01),  # Avoid 0 temp
                    use_cache=True,
                )

            # Decode only the generated part (exclude context)
            generated_tokens = generated[0, len(tokens):].tolist()
            generated_text = self.tok_decode(generated_tokens)

            results.append(generated_text)

        return results


def run_hellaswag_eval(
    model: GPT,
    tokenizer: GPT4Tokenizer,
    device: str = "cuda",
    batch_size: int = 8,
    limit: Optional[int] = None,
) -> dict:
    """
    Run HellaSwag evaluation on a YAGPT model.

    Args:
        model: YAGPT model to evaluate
        tokenizer: Tokenizer instance
        device: Device to run on
        batch_size: Batch size for evaluation
        limit: Limit number of examples (for testing)

    Returns:
        Dictionary with evaluation results:
        {
            'hellaswag_acc': float,  # Accuracy (0-1)
            'hellaswag_acc_norm': float,  # Length-normalized accuracy
        }
    """
    # Create wrapper
    eval_model = YAGPTEvalWrapper(
        model=model,
        tokenizer=tokenizer,
        device=device,
        batch_size=batch_size,
    )

    # Run evaluation
    print(f"\n{'='*70}")
    print("Running HellaSwag Evaluation")
    print(f"{'='*70}")
    print(f"Batch size: {batch_size}")
    if limit:
        print(f"Limit: {limit} examples (testing mode)")
    print()

    results = evaluator.simple_evaluate(
        model=eval_model,
        tasks=["hellaswag"],
        batch_size=batch_size,
        limit=limit,
        log_samples=False,  # Don't log individual samples
    )

    # Extract metrics
    hellaswag_results = results["results"]["hellaswag"]

    metrics = {
        "hellaswag_acc": hellaswag_results["acc,none"],
        "hellaswag_acc_norm": hellaswag_results["acc_norm,none"],
    }

    print(f"\n{'='*70}")
    print("HellaSwag Results:")
    print(f"  Accuracy: {metrics['hellaswag_acc']:.4f} ({metrics['hellaswag_acc']*100:.2f}%)")
    print(f"  Acc (normalized): {metrics['hellaswag_acc_norm']:.4f} ({metrics['hellaswag_acc_norm']*100:.2f}%)")
    print(f"{'='*70}\n")

    return metrics


if __name__ == "__main__":
    """Test the evaluation wrapper."""
    from yagpt.model import create_gpt_mini

    print("Testing HellaSwag evaluation wrapper...\n")

    # Determine device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}\n")

    # Create a small model for testing
    print("Creating test model...")
    model = create_gpt_mini(n_layer=2, n_head=2, n_embd=128, block_size=512)
    model = model.to(device)
    model.eval()

    # Create tokenizer
    print("Creating tokenizer...")
    tokenizer = GPT4Tokenizer()

    # Run evaluation on a small subset
    print("\nRunning evaluation on 10 examples...")
    metrics = run_hellaswag_eval(
        model=model,
        tokenizer=tokenizer,
        device=device,
        batch_size=2,
        limit=10,  # Just 10 examples for testing
    )

    print("\n✅ Evaluation wrapper test complete!")
