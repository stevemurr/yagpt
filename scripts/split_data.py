#!/usr/bin/env python3
"""
Split FineWeb dataset shards into train and validation sets.

This script takes a directory of parquet shards and splits them into
separate train and validation directories based on a specified ratio.

Usage:
    python scripts/split_data.py --input ./datasets/fineweb --val-ratio 0.1
    python scripts/split_data.py --input ./datasets/fineweb --val-shards 5
"""

import argparse
import shutil
from pathlib import Path
import glob


def split_shards(
    input_dir: str,
    train_dir: str = None,
    val_dir: str = None,
    val_ratio: float = None,
    val_shards: int = None,
    dry_run: bool = False,
):
    """
    Split parquet shards into train and validation directories.

    Args:
        input_dir: Directory containing all shards
        train_dir: Output directory for training shards (default: input_dir/train)
        val_dir: Output directory for validation shards (default: input_dir/val)
        val_ratio: Fraction of shards to use for validation (e.g., 0.1 for 10%)
        val_shards: Absolute number of shards to use for validation
        dry_run: If True, only print what would be done without moving files
    """
    input_path = Path(input_dir)

    # Set default output directories
    if train_dir is None:
        train_dir = input_path / "train"
    else:
        train_dir = Path(train_dir)

    if val_dir is None:
        val_dir = input_path / "val"
    else:
        val_dir = Path(val_dir)

    # Find all parquet shards
    shard_paths = sorted(glob.glob(str(input_path / "*.parquet")))

    if not shard_paths:
        print(f"ERROR: No parquet files found in {input_dir}")
        return

    print(f"Found {len(shard_paths)} total shards in {input_dir}")

    # Determine number of validation shards
    if val_shards is not None:
        num_val = val_shards
    elif val_ratio is not None:
        num_val = max(1, int(len(shard_paths) * val_ratio))
    else:
        # Default: 10% for validation
        num_val = max(1, int(len(shard_paths) * 0.1))

    num_train = len(shard_paths) - num_val

    print(f"\nSplit plan:")
    print(f"  Training shards: {num_train}")
    print(f"  Validation shards: {num_val}")
    print(f"  Val ratio: {num_val / len(shard_paths):.1%}")

    # Split shards
    val_shard_paths = shard_paths[-num_val:]  # Last N shards for validation
    train_shard_paths = shard_paths[:-num_val]  # Rest for training

    if dry_run:
        print("\n[DRY RUN] Would create directories:")
        print(f"  {train_dir}")
        print(f"  {val_dir}")
        print(f"\n[DRY RUN] Would move {num_train} files to {train_dir}")
        print(f"[DRY RUN] Would move {num_val} files to {val_dir}")
        return

    # Create output directories
    train_dir.mkdir(parents=True, exist_ok=True)
    val_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nCreated directories:")
    print(f"  {train_dir}")
    print(f"  {val_dir}")

    # Move training shards
    print(f"\nMoving {len(train_shard_paths)} shards to training directory...")
    for shard_path in train_shard_paths:
        shard_name = Path(shard_path).name
        dest = train_dir / shard_name
        shutil.move(shard_path, dest)

    # Move validation shards
    print(f"Moving {len(val_shard_paths)} shards to validation directory...")
    for shard_path in val_shard_paths:
        shard_name = Path(shard_path).name
        dest = val_dir / shard_name
        shutil.move(shard_path, dest)

    print("\nDone!")
    print(f"\nTo use this split in training, update your config:")
    print(f"  train_data_dir: \"{train_dir}\"")
    print(f"  val_data_dir: \"{val_dir}\"")


def main():
    parser = argparse.ArgumentParser(
        description="Split FineWeb shards into train and validation sets"
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Input directory containing all parquet shards",
    )
    parser.add_argument(
        "--train-dir",
        type=str,
        default=None,
        help="Output directory for training shards (default: input/train)",
    )
    parser.add_argument(
        "--val-dir",
        type=str,
        default=None,
        help="Output directory for validation shards (default: input/val)",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=None,
        help="Fraction of shards for validation (e.g., 0.1 for 10%%)",
    )
    parser.add_argument(
        "--val-shards",
        type=int,
        default=None,
        help="Absolute number of shards for validation",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be done without actually moving files",
    )

    args = parser.parse_args()

    # Validate arguments
    if args.val_ratio is not None and args.val_shards is not None:
        parser.error("Cannot specify both --val-ratio and --val-shards")

    split_shards(
        input_dir=args.input,
        train_dir=args.train_dir,
        val_dir=args.val_dir,
        val_ratio=args.val_ratio,
        val_shards=args.val_shards,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
