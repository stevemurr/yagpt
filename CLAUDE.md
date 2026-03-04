# CLAUDE.md

This file provides context for AI assistants working with this codebase.

## Project Overview

YAGPT (Yet Another GPT) is a clean, educational GPT implementation in PyTorch. The goal is maximum clarity with modern techniques - every file should be readable in one sitting.

## Architecture Principles

1. **Single Responsibility**: Each file does one thing, ~200 lines max
2. **Flat Configuration**: No nested config hierarchies, all in `TrainConfig`
3. **Composition over Inheritance**: Build models from simple components
4. **Explicit over Implicit**: Clear data flow, no magic

## Directory Structure

```
yagpt/
├── models/           # Model architecture (GPT, attention, MLP, etc.)
├── optim/            # Optimizers (Muon) and LR schedules
├── training/         # Trainer, config, callbacks, MFU tracking
├── data/             # Data loading
├── sft/              # Supervised fine-tuning (ChatML, masked loss)
├── lora/             # LoRA and QLoRA (NF4 quantization)
├── alignment/        # DPO, SimPO, GRPO preference optimization
├── eval/             # lm-eval-harness integration, proxy metrics
└── tokenizer.py      # Tiktoken wrapper

scripts/
├── cli.py            # Command-line interface
└── prepare_data.py   # Data preparation tool

tests/
└── test_*.py         # pytest tests

configs/
└── default.yaml      # Training configuration
```

## Key Files

- `yagpt/models/gpt.py` - Main GPT model class with forward() and generate()
- `yagpt/models/attention.py` - Causal self-attention with GQA and QK-Norm
- `yagpt/training/trainer.py` - Training loop with gradient accumulation
- `yagpt/training/config.py` - Flat TrainConfig dataclass
- `yagpt/training/callbacks.py` - Modular logging, checkpointing, eval
- `yagpt/training/mfu.py` - Model FLOPs Utilization callback
- `yagpt/sft/trainer.py` - SFT training loop
- `yagpt/sft/dataset.py` - ChatML dataset with masked labels
- `yagpt/lora/lora.py` - LoRA low-rank adaptation
- `yagpt/lora/qlora.py` - NF4 4-bit quantization
- `yagpt/alignment/dpo.py` - Direct Preference Optimization
- `yagpt/alignment/grpo.py` - Group Relative Policy Optimization
- `yagpt/alignment/simpo.py` - Simple Preference Optimization
- `yagpt/eval/harness.py` - lm-eval-harness wrapper

## Common Tasks

### Running tests
```bash
pytest tests/
```

### Training
```bash
yagpt train -c configs/default.yaml
```

### Supervised fine-tuning
```bash
yagpt sft -c configs/sft.yaml --checkpoint checkpoints/final.pt
```

### Alignment
```bash
yagpt align checkpoints/sft.pt -d data/prefs.jsonl -m dpo
```

### Evaluation
```bash
yagpt eval checkpoints/final.pt --tasks hellaswag
```

### Adding a new callback
1. Create class inheriting from `Callback` in `training/callbacks.py`
2. Implement desired hooks: `on_train_start`, `on_step_end`, `on_eval_end`, etc.
3. Add to trainer's callback list

### Applying LoRA / QLoRA
```python
from yagpt.lora import apply_lora, quantize_model_nf4

# QLoRA: quantize first, then apply LoRA
quantize_model_nf4(model)
apply_lora(model, rank=16)
```

### Modifying the model
- Core architecture: `models/gpt.py`
- Individual components: `models/attention.py`, `models/mlp.py`, etc.
- All use standard PyTorch patterns

## Code Style

- Type hints on all function signatures
- Google-style docstrings for public APIs
- No unnecessary abstractions
- Prefer explicit loops over clever comprehensions for clarity

## Dependencies

Core:
- torch >= 2.0
- tiktoken
- pyarrow
- typer, rich (CLI)
- pyyaml

Optional:
- wandb (logging)
- lm-eval (evaluation benchmarks)
- pytest (testing)

## Technical Notes

### Dual Optimizer
The `dual` optimizer mode uses:
- **Muon**: For transformer block parameters (attention, MLP weights)
- **AdamW**: For embeddings and layer norms

Muon uses Newton-Schulz orthogonalization for faster convergence on dense matrices.

### Model Forward Signature
```python
def forward(input_ids, targets=None, kv_cache=None, ignore_index=-1, return_logits=False):
    # Returns: (logits, loss, new_kv_cache)
    # - With targets: logits is (B, T, vocab) for all positions, loss computed
    # - With return_logits=True: logits is (B, T, vocab), no loss (for alignment)
    # - Without targets or return_logits: logits is (B, 1, vocab) for last token only
```

### KV Caching
Generation uses KV caching for efficiency:
1. First call processes full prompt, returns kv_cache
2. Subsequent calls process one token, using cached K/V

### Data Format
Expects sharded parquet files with:
- `tokens` column (pre-tokenized, faster), OR
- `text` column (tokenized on-the-fly)
