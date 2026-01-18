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
├── training/         # Trainer, config, callbacks
├── data/             # Data loading
└── tokenizer.py      # Tiktoken wrapper

scripts/
└── cli.py            # Command-line interface

tests/
└── test_*.py         # pytest tests

configs/
└── default.yaml      # Training configuration
```

## Key Files

- `yagpt/models/gpt.py` - Main GPT model class with forward() and generate()
- `yagpt/models/attention.py` - Causal self-attention with GQA support
- `yagpt/training/trainer.py` - Training loop with gradient accumulation
- `yagpt/training/config.py` - Flat TrainConfig dataclass
- `yagpt/training/callbacks.py` - Modular logging, checkpointing, eval

## Common Tasks

### Running tests
```bash
pytest tests/
```

### Training
```bash
yagpt train -c configs/default.yaml
```

### Adding a new callback
1. Create class inheriting from `Callback` in `training/callbacks.py`
2. Implement desired hooks: `on_train_start`, `on_step_end`, `on_eval_end`, etc.
3. Add to trainer's callback list

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
def forward(input_ids, targets=None, kv_cache=None):
    # Returns: (logits, loss, new_kv_cache)
    # - Without targets: logits is (B, 1, vocab) for last token only
    # - With targets: logits is (B, T, vocab) for all positions
```

### KV Caching
Generation uses KV caching for efficiency:
1. First call processes full prompt, returns kv_cache
2. Subsequent calls process one token, using cached K/V

### Data Format
Expects sharded parquet files with:
- `tokens` column (pre-tokenized, faster), OR
- `text` column (tokenized on-the-fly)
