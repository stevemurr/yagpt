"""
Utility functions for GPT-4 Mini

Includes:
- Checkpoint inspection
- Model comparison
- Configuration management
- Data statistics
"""

import torch
import argparse
from pathlib import Path
from tabulate import tabulate


def inspect_checkpoint(checkpoint_path: str):
    """
    Inspect a checkpoint file and print detailed information.

    Args:
        checkpoint_path: Path to checkpoint file
    """
    checkpoint = torch.load(checkpoint_path, map_location='cpu')

    print("="*70)
    print(f"Checkpoint: {checkpoint_path}")
    print("="*70)

    # Basic info
    print("\n📊 Training Information:")
    print(f"  Iteration: {checkpoint['iteration']:,}")
    print(f"  Train Loss: {checkpoint['train_loss']:.4f}")
    if 'val_loss' in checkpoint and checkpoint['val_loss'] is not None:
        print(f"  Validation Loss: {checkpoint['val_loss']:.4f}")

    # Configuration
    print("\n⚙️  Model Configuration:")
    config = checkpoint['config']
    config_table = [
        ["Layers", config['n_layer']],
        ["Heads", config['n_head']],
        ["Embedding Dim", config['n_embd']],
        ["Context Length", config['block_size']],
        ["Vocabulary Size", f"{config['vocab_size']:,}"],
        ["Dropout", config['dropout']],
    ]
    print(tabulate(config_table, tablefmt="simple"))

    # Model size
    print("\n📦 Model Size:")
    total_params = sum(p.numel() for p in checkpoint['model_state_dict'].values())
    print(f"  Total Parameters: {total_params:,} ({total_params/1e6:.2f}M)")

    # Breakdown by component
    print("\n🔧 Parameter Breakdown:")
    component_sizes = {}
    for name, param in checkpoint['model_state_dict'].items():
        component = name.split('.')[0]  # Get top-level component
        if component not in component_sizes:
            component_sizes[component] = 0
        component_sizes[component] += param.numel()

    breakdown_table = [
        [comp, f"{size:,}", f"{size/1e6:.2f}M", f"{size/total_params*100:.1f}%"]
        for comp, size in sorted(component_sizes.items(), key=lambda x: x[1], reverse=True)
    ]
    print(tabulate(
        breakdown_table,
        headers=["Component", "Parameters", "Millions", "Percentage"],
        tablefmt="simple"
    ))

    # Optimizer state
    print("\n🎯 Optimizer State:")
    if 'optimizer_state_dict' in checkpoint:
        opt_state = checkpoint['optimizer_state_dict']
        print(f"  Param Groups: {len(opt_state['param_groups'])}")
        for i, group in enumerate(opt_state['param_groups']):
            print(f"  Group {i}:")
            print(f"    Learning Rate: {group['lr']:.2e}")
            print(f"    Weight Decay: {group['weight_decay']}")
            if 'betas' in group:
                print(f"    Betas: {group['betas']}")

    # File size
    file_size = Path(checkpoint_path).stat().st_size
    print(f"\n💾 File Size: {file_size / 1024**2:.2f} MB")


def list_checkpoints(checkpoint_dir: str = "./checkpoints"):
    """
    List all checkpoints in a directory with key information.

    Args:
        checkpoint_dir: Directory containing checkpoints
    """
    checkpoint_dir = Path(checkpoint_dir)

    if not checkpoint_dir.exists():
        print(f"Checkpoint directory not found: {checkpoint_dir}")
        return

    checkpoints = sorted(
        checkpoint_dir.glob("checkpoint_iter_*.pt"),
        key=lambda p: int(p.stem.split('_')[-1])
    )

    if not checkpoints:
        print(f"No checkpoints found in {checkpoint_dir}")
        return

    print("="*70)
    print(f"Checkpoints in {checkpoint_dir}")
    print("="*70 + "\n")

    table_data = []
    for ckpt_path in checkpoints:
        ckpt = torch.load(ckpt_path, map_location='cpu')

        iteration = ckpt['iteration']
        train_loss = ckpt['train_loss']
        val_loss = ckpt.get('val_loss', None)
        file_size = ckpt_path.stat().st_size / 1024**2  # MB

        table_data.append([
            ckpt_path.name,
            f"{iteration:,}",
            f"{train_loss:.4f}",
            f"{val_loss:.4f}" if val_loss is not None else "N/A",
            f"{file_size:.1f} MB"
        ])

    print(tabulate(
        table_data,
        headers=["Checkpoint", "Iteration", "Train Loss", "Val Loss", "Size"],
        tablefmt="simple"
    ))

    print(f"\nTotal checkpoints: {len(checkpoints)}")
    print(f"Latest: {checkpoints[-1].name}")


def compare_checkpoints(checkpoint_paths: list[str]):
    """
    Compare multiple checkpoints.

    Args:
        checkpoint_paths: List of checkpoint paths
    """
    print("="*70)
    print("Checkpoint Comparison")
    print("="*70 + "\n")

    table_data = []
    for ckpt_path in checkpoint_paths:
        ckpt = torch.load(ckpt_path, map_location='cpu')

        iteration = ckpt['iteration']
        train_loss = ckpt['train_loss']
        val_loss = ckpt.get('val_loss', None)

        config = ckpt['config']
        model_desc = f"{config['n_layer']}L-{config['n_head']}H-{config['n_embd']}D"

        table_data.append([
            Path(ckpt_path).name,
            f"{iteration:,}",
            model_desc,
            f"{train_loss:.4f}",
            f"{val_loss:.4f}" if val_loss is not None else "N/A",
        ])

    print(tabulate(
        table_data,
        headers=["Checkpoint", "Iteration", "Architecture", "Train Loss", "Val Loss"],
        tablefmt="simple"
    ))


def estimate_training_time(
    current_iter: int,
    target_iter: int,
    time_per_iter_ms: float,
):
    """
    Estimate remaining training time.

    Args:
        current_iter: Current iteration
        target_iter: Target iteration
        time_per_iter_ms: Average time per iteration in milliseconds
    """
    remaining_iters = target_iter - current_iter
    remaining_time_s = remaining_iters * time_per_iter_ms / 1000

    hours = int(remaining_time_s // 3600)
    minutes = int((remaining_time_s % 3600) // 60)
    seconds = int(remaining_time_s % 60)

    print(f"Current iteration: {current_iter:,}")
    print(f"Target iteration: {target_iter:,}")
    print(f"Remaining iterations: {remaining_iters:,}")
    print(f"Time per iteration: {time_per_iter_ms:.2f}ms")
    print(f"Estimated remaining time: {hours}h {minutes}m {seconds}s")


def calculate_model_flops(config_dict: dict, seq_length: int = 2048):
    """
    Calculate approximate FLOPs for a forward pass.

    Args:
        config_dict: Model configuration dictionary
        seq_length: Sequence length

    Returns:
        Approximate FLOPs
    """
    n_layer = config_dict['n_layer']
    n_embd = config_dict['n_embd']
    n_head = config_dict['n_head']
    vocab_size = config_dict['vocab_size']

    # Embeddings
    embed_flops = 0

    # Attention (per layer)
    # QKV projection: 3 * seq_len * n_embd * n_embd
    # Attention: seq_len * seq_len * n_embd
    # Output projection: seq_len * n_embd * n_embd
    attn_flops = (
        3 * seq_length * n_embd * n_embd +
        seq_length * seq_length * n_embd +
        seq_length * n_embd * n_embd
    )

    # MLP (per layer)
    # Up projection: seq_len * n_embd * (4 * n_embd)
    # Down projection: seq_len * (4 * n_embd) * n_embd
    mlp_flops = (
        seq_length * n_embd * (4 * n_embd) +
        seq_length * (4 * n_embd) * n_embd
    )

    # LM head
    lm_head_flops = seq_length * n_embd * vocab_size

    # Total
    total_flops = (
        embed_flops +
        n_layer * (attn_flops + mlp_flops) +
        lm_head_flops
    )

    # Multiply by 2 for backward pass
    total_flops *= 2

    return total_flops


def main():
    parser = argparse.ArgumentParser(description="GPT-4 Mini Utilities")
    subparsers = parser.add_subparsers(dest='command', help='Command to run')

    # Inspect checkpoint
    inspect_parser = subparsers.add_parser('inspect', help='Inspect a checkpoint')
    inspect_parser.add_argument('checkpoint', type=str, help='Path to checkpoint')

    # List checkpoints
    list_parser = subparsers.add_parser('list', help='List all checkpoints')
    list_parser.add_argument(
        '--dir',
        type=str,
        default='./checkpoints',
        help='Checkpoint directory'
    )

    # Compare checkpoints
    compare_parser = subparsers.add_parser('compare', help='Compare checkpoints')
    compare_parser.add_argument(
        'checkpoints',
        type=str,
        nargs='+',
        help='Checkpoint paths to compare'
    )

    # Estimate training time
    time_parser = subparsers.add_parser('time', help='Estimate training time')
    time_parser.add_argument('--current', type=int, required=True, help='Current iteration')
    time_parser.add_argument('--target', type=int, required=True, help='Target iteration')
    time_parser.add_argument('--ms-per-iter', type=float, required=True, help='MS per iteration')

    args = parser.parse_args()

    if args.command == 'inspect':
        inspect_checkpoint(args.checkpoint)

    elif args.command == 'list':
        list_checkpoints(args.dir)

    elif args.command == 'compare':
        compare_checkpoints(args.checkpoints)

    elif args.command == 'time':
        estimate_training_time(args.current, args.target, args.ms_per_iter)

    else:
        parser.print_help()


if __name__ == "__main__":
    try:
        import tabulate
    except ImportError:
        print("Installing tabulate for better formatting...")
        print("Run: pip install tabulate")
        print("\nFalling back to simple output...")

        # Simple implementations without tabulate
        def tabulate(data, headers=None, tablefmt=None):
            if headers:
                print(" | ".join(str(h) for h in headers))
                print("-" * 50)
            for row in data:
                print(" | ".join(str(cell) for cell in row))
            return ""

    main()
