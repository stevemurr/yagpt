# YAGPT

Yet Another GPT - A clean, modular GPT implementation for learning and experimentation.

## Features

- **Clean Architecture**: Single-responsibility modules, ~200 lines per file
- **Modern Techniques**: RoPE, RMSNorm, SwiGLU, GQA, Flash Attention
- **Dual Optimizer**: Muon for transformer blocks + AdamW for embeddings
- **Streaming Data**: Efficient parquet-based data loading
- **Modular Callbacks**: Logging, checkpointing, eval without loop clutter

## Quick Start

```bash
# Install
pip install -e .

# Train
yagpt train --config configs/default.yaml

# Generate
yagpt generate checkpoints/final.pt --prompt "Once upon a time"
```

## Project Structure

```
yagpt/
├── models/           # Model components
│   ├── gpt.py        # Main GPT model
│   ├── attention.py  # Causal attention with GQA
│   ├── mlp.py        # SwiGLU MLP
│   ├── norm.py       # RMSNorm
│   └── rope.py       # Rotary position embeddings
├── optim/            # Optimizers
│   ├── muon.py       # Muon optimizer
│   └── lr_schedule.py
├── training/         # Training infrastructure
│   ├── trainer.py    # Training loop
│   ├── config.py     # Configuration
│   └── callbacks.py  # Logging, checkpoints, eval
├── data/             # Data loading
│   └── dataloader.py # Streaming parquet loader
└── tokenizer.py      # Tiktoken wrapper
```

## Usage

### Python API

```python
from yagpt import GPT, GPTConfig, Tokenizer

# Create model
config = GPTConfig(
    vocab_size=50257,
    n_layers=12,
    n_heads=12,
    dim=768,
)
model = GPT(config)

# Generate text
tokenizer = Tokenizer("gpt2")
tokens = tokenizer.encode("Hello, world!")
input_ids = torch.tensor([tokens])

output = model.generate(input_ids, max_new_tokens=50)
print(tokenizer.decode(output[0].tolist()))
```

### Training

```python
from yagpt import Trainer, TrainConfig, create_dataloader

config = TrainConfig.from_yaml("configs/default.yaml")
train_loader = create_dataloader(config.train_data_dir, ...)

trainer = Trainer(config, train_loader)
trainer.train()
```

### CLI Commands

```bash
yagpt train -c config.yaml     # Train model
yagpt generate ckpt.pt -p "Hi" # Generate text
yagpt info ckpt.pt             # Checkpoint info
yagpt count ./data             # Count dataset tokens
```

## Configuration

All settings in a single flat YAML file:

```yaml
# Model
n_layers: 12
n_heads: 12
dim: 768

# Training
batch_size: 32
total_batch_size: 524288
max_steps: 100000

# Optimizer
optimizer: "dual"      # muon + adamw
learning_rate: 3e-4
muon_lr: 0.02
```

See `configs/default.yaml` for all options.

## Data Format

Expects parquet files with either:
- `text` column: Raw text (tokenized on-the-fly)
- `tokens` column: Pre-tokenized sequences (faster)

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
- [Muon](https://kellerjordan.github.io/posts/muon/) - Muon optimizer

## License

MIT
