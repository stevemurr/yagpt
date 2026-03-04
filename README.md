# YAGPT

Yet Another GPT - A clean, modular GPT implementation for learning and experimentation.

## Features

- **Clean Architecture**: Single-responsibility modules, ~200 lines per file
- **Modern Techniques**: RoPE, RMSNorm, SwiGLU, GQA, QK-Norm, Flash Attention
- **Dual Optimizer**: Muon for transformer blocks + AdamW for embeddings
- **LR Schedules**: Warmup-cosine, three-phase, WSD (Warmup-Stable-Decay)
- **Gradient Checkpointing**: Trade compute for memory on large models
- **Vocab Padding**: Pad vocab to multiples of 64 for GPU kernel efficiency
- **MFU Tracking**: Model FLOPs Utilization monitoring during training
- **Streaming Data**: Efficient parquet-based data loading with `yagpt-data` prep tool
- **Modular Callbacks**: Logging, checkpointing, eval without loop clutter
- **SFT**: Supervised fine-tuning with ChatML formatting and masked loss
- **LoRA / QLoRA**: Parameter-efficient fine-tuning with 4-bit NF4 quantization
- **Alignment**: DPO, SimPO, and GRPO preference optimization
- **Evaluation**: lm-eval-harness integration and proxy metrics (bits-per-byte)

## Quick Start

```bash
# Install
pip install -e .

# Train
yagpt train --config configs/default.yaml

# Generate
yagpt generate checkpoints/final.pt --prompt "Once upon a time"

# Supervised fine-tuning
yagpt sft -c configs/sft.yaml --checkpoint checkpoints/final.pt

# Evaluate on benchmarks
yagpt eval checkpoints/final.pt --tasks hellaswag,arc_easy

# Align with DPO
yagpt align checkpoints/sft.pt -d data/prefs.jsonl -m dpo
```

## Project Structure

```
yagpt/
├── models/           # Model components
│   ├── gpt.py        # Main GPT model (forward, generate)
│   ├── attention.py  # Causal attention with GQA, QK-Norm
│   ├── mlp.py        # SwiGLU MLP
│   ├── norm.py       # RMSNorm
│   └── rope.py       # Rotary position embeddings
├── optim/            # Optimizers
│   ├── muon.py       # Muon optimizer
│   └── lr_schedule.py # Cosine, three-phase, WSD schedules
├── training/         # Training infrastructure
│   ├── trainer.py    # Training loop with gradient accumulation
│   ├── config.py     # Flat TrainConfig dataclass
│   ├── callbacks.py  # Logging, checkpoints, eval, sampling
│   └── mfu.py        # Model FLOPs Utilization callback
├── data/             # Data loading
│   └── dataloader.py # Streaming parquet loader
├── sft/              # Supervised fine-tuning
│   ├── config.py     # SFT configuration
│   ├── trainer.py    # SFT training loop
│   ├── dataset.py    # ChatML dataset with masked labels
│   └── format.py     # Chat template formatting
├── lora/             # Parameter-efficient fine-tuning
│   ├── lora.py       # LoRA low-rank adaptation
│   └── qlora.py      # NF4 4-bit quantization
├── alignment/        # Preference optimization
│   ├── dpo.py        # Direct Preference Optimization
│   ├── simpo.py      # Simple Preference Optimization
│   ├── grpo.py       # Group Relative Policy Optimization
│   └── dataset.py    # Preference dataset and collation
├── eval/             # Evaluation
│   ├── harness.py    # lm-eval-harness wrapper
│   ├── callback.py   # Benchmark callback for training
│   └── proxy.py      # Bits-per-byte proxy metric
└── tokenizer.py      # Tiktoken wrapper

scripts/
├── cli.py            # CLI (train, generate, sft, align, eval, info, count)
└── prepare_data.py   # Data preparation tool (yagpt-data)

configs/
└── default.yaml      # Training configuration
```

## Pipeline Overview

```
Pre-train → SFT → LoRA/QLoRA → Align (DPO/GRPO/SimPO) → Eval
```

1. **Pre-train**: `yagpt train` - Train a base model on raw text
2. **SFT**: `yagpt sft` - Fine-tune on instruction/chat data with ChatML
3. **LoRA**: Apply parameter-efficient adapters (optionally with NF4 quantization)
4. **Align**: `yagpt align` - Preference optimization (DPO, SimPO, or GRPO)
5. **Eval**: `yagpt eval` - Benchmark with lm-eval-harness

## CLI Commands

```bash
yagpt train -c config.yaml           # Train model
yagpt generate ckpt.pt -p "Hi"       # Generate text
yagpt sft -c sft.yaml                # Supervised fine-tuning
yagpt align ckpt.pt -d prefs.jsonl   # Preference alignment
yagpt eval ckpt.pt -t hellaswag      # Benchmark evaluation
yagpt info ckpt.pt                   # Checkpoint info
yagpt count ./data                   # Count dataset tokens
yagpt-data                           # Data preparation
```

## Configuration

All settings in a single flat YAML file:

```yaml
# Model
n_layers: 12
n_heads: 12
dim: 768
qk_norm: true
pad_vocab_to: 64
gradient_checkpointing: false

# Training
batch_size: 32
total_batch_size: 524288
max_steps: 100000

# Optimizer
optimizer: "dual"      # muon + adamw
learning_rate: 3e-4
muon_lr: 0.02

# LR Schedule
lr_schedule: "warmup_cosine"  # warmup_cosine, cosine, three_phase, or wsd
decay_ratio: 0.2              # For WSD schedule
```

See `configs/default.yaml` for all options.

## Data Format

Expects parquet files with either:
- `text` column: Raw text (tokenized on-the-fly)
- `tokens` column: Pre-tokenized sequences (faster)

Use `yagpt-data` to prepare and shard datasets.

## Model Architecture

```
Input IDs
    ↓
Token Embedding (no position embedding - using RoPE)
    ↓
┌─────────────────────────────────────┐
│  Transformer Block (×n_layers)      │
│  ├─ RMSNorm                         │
│  ├─ Causal Attention (RoPE, GQA)    │
│  ├─ Residual                        │
│  ├─ RMSNorm                         │
│  ├─ SwiGLU MLP                      │
│  └─ Residual                        │
└─────────────────────────────────────┘
    ↓
RMSNorm
    ↓
LM Head (weight-tied with embeddings)
    ↓
Logits
```

## References

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) - Transformer architecture
- [RoFormer](https://arxiv.org/abs/2104.09864) - Rotary Position Embeddings
- [RMSNorm](https://arxiv.org/abs/1910.07467) - Root Mean Square Normalization
- [GLU Variants](https://arxiv.org/abs/2002.05202) - SwiGLU activation
- [GQA](https://arxiv.org/abs/2305.13245) - Grouped Query Attention
- [QK-Norm](https://arxiv.org/abs/2302.05442) - Query-Key Normalization
- [Muon](https://kellerjordan.github.io/posts/muon/) - Muon optimizer
- [DPO](https://arxiv.org/abs/2305.18290) - Direct Preference Optimization
- [SimPO](https://arxiv.org/abs/2405.14734) - Simple Preference Optimization
- [GRPO](https://arxiv.org/abs/2402.03300) - Group Relative Policy Optimization
- [LoRA](https://arxiv.org/abs/2106.09685) - Low-Rank Adaptation
- [QLoRA](https://arxiv.org/abs/2305.14314) - 4-bit NormalFloat Quantization

## License

MIT
