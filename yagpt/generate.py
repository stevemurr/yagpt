"""
Text generation script for GPT-4 Mini

Supports:
- Loading from checkpoint
- Interactive generation
- Batch generation
- Various sampling strategies (temperature, top-k, top-p)
"""

import torch
import argparse
from pathlib import Path

from yagpt.model import GPT, GPTConfig
from yagpt.tokenizer import GPT4Tokenizer


def load_model_from_checkpoint(checkpoint_path: str, device: str = "cuda") -> tuple[GPT, GPT4Tokenizer]:
    """
    Load a model from a checkpoint.

    Args:
        checkpoint_path: Path to checkpoint file
        device: Device to load model on

    Returns:
        Tuple of (model, tokenizer)
    """
    print(f"Loading checkpoint from {checkpoint_path}...")

    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Recreate config
    config_dict = checkpoint['config']
    model_config = GPTConfig(
        vocab_size=config_dict['vocab_size'],
        n_layer=config_dict['n_layer'],
        n_head=config_dict['n_head'],
        n_embd=config_dict['n_embd'],
        block_size=config_dict['block_size'],
        dropout=0.0,  # No dropout during inference
    )

    # Create model
    model = GPT(model_config)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()

    print(f"  Loaded from iteration {checkpoint['iteration']}")
    if 'val_loss' in checkpoint:
        print(f"  Validation loss: {checkpoint['val_loss']:.4f}")

    # Create tokenizer
    tokenizer = GPT4Tokenizer()

    return model, tokenizer


@torch.no_grad()
def generate_text(
    model: GPT,
    tokenizer: GPT4Tokenizer,
    prompt: str,
    max_new_tokens: int = 100,
    temperature: float = 0.8,
    top_k: int = 40,
    top_p: float = 0.9,
    device: str = "cuda",
) -> str:
    """
    Generate text from a prompt.

    Args:
        model: GPT model
        tokenizer: Tokenizer
        prompt: Input text prompt
        max_new_tokens: Maximum number of tokens to generate
        temperature: Sampling temperature (higher = more random)
        top_k: Top-k sampling (None to disable)
        top_p: Top-p (nucleus) sampling (None to disable)
        device: Device to run on

    Returns:
        Generated text
    """
    # Encode prompt
    tokens = tokenizer.encode(prompt)
    tokens = torch.tensor(tokens, dtype=torch.long, device=device).unsqueeze(0)  # (1, T)

    # Generate
    generated_tokens = generate_with_top_p(
        model=model,
        idx=tokens,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
    )

    # Decode
    generated_text = tokenizer.decode(generated_tokens[0].tolist())

    return generated_text


@torch.no_grad()
def generate_with_top_p(
    model: GPT,
    idx: torch.Tensor,
    max_new_tokens: int,
    temperature: float = 1.0,
    top_k: int = None,
    top_p: float = None,
) -> torch.Tensor:
    """
    Generate tokens with top-p (nucleus) sampling support.

    This is an enhanced version of the model's generate() method.
    """
    for _ in range(max_new_tokens):
        # Crop context if needed
        idx_cond = idx if idx.size(1) <= model.config.block_size else idx[:, -model.config.block_size:]

        # Forward pass
        logits, _, _ = model(idx_cond)
        logits = logits[:, -1, :] / temperature

        # Top-k filtering
        if top_k is not None:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = -float('Inf')

        # Top-p (nucleus) filtering
        if top_p is not None:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)

            # Remove tokens with cumulative probability above the threshold
            sorted_indices_to_remove = cumulative_probs > top_p
            # Shift the indices to the right to keep the first token above threshold
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0

            # Scatter sorted tensors back to original indexing
            indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
            logits[indices_to_remove] = -float('Inf')

        # Sample
        probs = torch.softmax(logits, dim=-1)
        idx_next = torch.multinomial(probs, num_samples=1)

        # Append
        idx = torch.cat((idx, idx_next), dim=1)

    return idx


def interactive_mode(
    model: GPT,
    tokenizer: GPT4Tokenizer,
    device: str = "cuda",
    max_new_tokens: int = 100,
    temperature: float = 0.8,
    top_k: int = 40,
    top_p: float = 0.9,
):
    """
    Interactive text generation mode.

    User can enter prompts and get generated continuations.
    """
    print("\n" + "="*70)
    print("Interactive Generation Mode")
    print("="*70)
    print("\nEnter prompts to generate text. Type 'quit' or 'exit' to stop.")
    print("Commands:")
    print("  /temp <value>   - Set temperature (e.g., /temp 0.8)")
    print("  /tokens <n>     - Set max tokens (e.g., /tokens 200)")
    print("  /topk <k>       - Set top-k (e.g., /topk 50)")
    print("  /topp <p>       - Set top-p (e.g., /topp 0.95)")
    print()

    while True:
        try:
            prompt = input(">>> ")

            if prompt.lower() in ['quit', 'exit', 'q']:
                break

            # Handle commands
            if prompt.startswith('/'):
                parts = prompt.split()
                cmd = parts[0].lower()

                if cmd == '/temp' and len(parts) > 1:
                    temperature = float(parts[1])
                    print(f"Temperature set to {temperature}")
                elif cmd == '/tokens' and len(parts) > 1:
                    max_new_tokens = int(parts[1])
                    print(f"Max tokens set to {max_new_tokens}")
                elif cmd == '/topk' and len(parts) > 1:
                    top_k = int(parts[1])
                    print(f"Top-k set to {top_k}")
                elif cmd == '/topp' and len(parts) > 1:
                    top_p = float(parts[1])
                    print(f"Top-p set to {top_p}")
                else:
                    print("Unknown command")
                continue

            if not prompt.strip():
                continue

            # Generate
            print("\nGenerating...\n")
            generated = generate_text(
                model=model,
                tokenizer=tokenizer,
                prompt=prompt,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                device=device,
            )

            print(generated)
            print()

        except KeyboardInterrupt:
            print("\n\nExiting...")
            break
        except Exception as e:
            print(f"Error: {e}")


def batch_generate(
    model: GPT,
    tokenizer: GPT4Tokenizer,
    prompts: list[str],
    device: str = "cuda",
    **kwargs
):
    """
    Generate text for multiple prompts.

    Args:
        model: GPT model
        tokenizer: Tokenizer
        prompts: List of prompt strings
        device: Device
        **kwargs: Generation parameters
    """
    print("\n" + "="*70)
    print("Batch Generation")
    print("="*70 + "\n")

    for i, prompt in enumerate(prompts):
        print(f"Prompt {i+1}/{len(prompts)}: {prompt[:50]}...")
        generated = generate_text(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            device=device,
            **kwargs
        )
        print(f"Generated:\n{generated}\n")
        print("-"*70 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Generate text using GPT-4 Mini")

    # Model loading
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to checkpoint file"
    )

    # Generation mode
    parser.add_argument(
        "--mode",
        type=str,
        choices=["interactive", "batch", "single"],
        default="interactive",
        help="Generation mode"
    )

    # Single generation
    parser.add_argument(
        "--prompt",
        type=str,
        help="Prompt for single generation mode"
    )

    # Batch generation
    parser.add_argument(
        "--prompts-file",
        type=str,
        help="File containing prompts (one per line) for batch mode"
    )

    # Generation parameters
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=100,
        help="Maximum number of tokens to generate"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.8,
        help="Sampling temperature (higher = more random)"
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=40,
        help="Top-k sampling parameter"
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=0.9,
        help="Top-p (nucleus) sampling parameter"
    )

    # System
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use (cuda/cpu)"
    )

    args = parser.parse_args()

    # Load model
    model, tokenizer = load_model_from_checkpoint(args.checkpoint, args.device)

    # Run generation
    if args.mode == "interactive":
        interactive_mode(
            model=model,
            tokenizer=tokenizer,
            device=args.device,
            max_new_tokens=args.max_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
        )

    elif args.mode == "single":
        if not args.prompt:
            print("Error: --prompt is required for single mode")
            return

        generated = generate_text(
            model=model,
            tokenizer=tokenizer,
            prompt=args.prompt,
            max_new_tokens=args.max_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            device=args.device,
        )

        print("\n" + "="*70)
        print("Generated Text")
        print("="*70)
        print(f"\nPrompt: {args.prompt}\n")
        print(generated)
        print()

    elif args.mode == "batch":
        if not args.prompts_file:
            print("Error: --prompts-file is required for batch mode")
            return

        with open(args.prompts_file, 'r') as f:
            prompts = [line.strip() for line in f if line.strip()]

        batch_generate(
            model=model,
            tokenizer=tokenizer,
            prompts=prompts,
            device=args.device,
            max_new_tokens=args.max_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
        )


if __name__ == "__main__":
    main()
