# YAGPT (Yet Another GPT) - PyTorch Implementation

A GPT-style language model built from scratch in PyTorch.

## Features

- ✅ **Complete GPT architecture** - Decoder-only transformer with multi-head attention
- ✅ **GPT-4 tokenizer** - Using tiktoken's `cl100k_base` encoding (100k vocabulary)
- ✅ **Scalable** - Mini (~85M), Medium (~350M), Large (~774M) configurations
- ✅ **Efficient data loading** - Streaming from parquet shards (FineWeb dataset)
- ✅ **Flexible logging** - Console, CSV, W&B, TensorBoard support
- ✅ **Checkpointing** - Automatic checkpoint management (keeps last 5)
- ✅ **Resume training** - Continue from any checkpoint
- ✅ **Generation** - Interactive and batch text generation
- ✅ **Modern training** - Gradient accumulation, LR scheduling, AdamW optimizer

## Project Structure

```
yagpt/
├── yagpt/
│   ├── model.py           # GPT model architecture
│   ├── tokenizer.py       # GPT-4 tokenizer wrapper (tiktoken)
│   ├── dataloader.py      # FineWeb dataset loader
│   ├── logger.py          # Multi-backend logging system
│   ├── train.py           # Training script
│   ├── generate.py        # Text generation script
│   └── utils.py           # Checkpoint inspection utilities
├── scripts/
│   └── train.py           # CLI entry point
├── tests/                 # Unit tests
├── docs/                  # Documentation
├── configs/               # Training configurations
├── pyproject.toml         # Project metadata and dependencies
└── README.md              # This file
```

## Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/stevemurr/yagpt.git
   cd yagpt
   ```

2. **Create a virtual environment**:
   ```bash
   uv venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. **Install the package**:
   ```bash
   # Install in editable mode with all dependencies
   uv pip install -e ".[all]"

   # Or just core dependencies
   uv pip install -e .
   ```

## Training

### Basic Training

Start training from scratch:

```bash
yagpt train -c ./configs/config.yaml
```

This uses default configuration:
- Model: GPT-Mini (12 layers, 768 dim, ~85M params)
- Batch size: 8 (with gradient accumulation = effective 32)
- Context length: 2048 tokens
- Learning rate: 3e-4 with warmup and cosine decay

### Resume Training

Continue from a checkpoint:

```bash
yagpt resume -c ./configs/config.yaml
```

Or specify a specific checkpoint:

```python
from yagpt import TrainingConfig, train

config = TrainingConfig()
config.resume_from = "./checkpoints/checkpoint_iter_5000.pt"
train(config)
```

### Fine-tuning

To fine-tune on a different dataset:

1. Load a pre-trained checkpoint
2. Point to your new dataset directory
3. Optionally reduce learning rate

```python
config = TrainingConfig()
config.resume_from = "./checkpoints/checkpoint_iter_50000.pt"
config.data_dir = "./datasets/my_custom_data"
config.learning_rate = 1e-4  # Lower LR for fine-tuning
config.max_iters = 10000
```

## Text Generation

### Interactive Mode

Generate text interactively:

```bash
yagpt generate ./checkpoints/checkpoint_iter_10000.pt

# or
python -m yagpt.generate \
  --checkpoint ./checkpoints/checkpoint_iter_10000.pt \
  --mode interactive
```

Commands in interactive mode:
- `/temp 0.8` - Set temperature
- `/tokens 200` - Set max tokens
- `/topk 50` - Set top-k sampling
- `/topp 0.95` - Set top-p (nucleus) sampling
- `quit` or `exit` - Exit

### Single Generation

Generate from a single prompt:

```bash
yagpt generate \
  --checkpoint ./checkpoints/checkpoint_iter_10000.pt \
  --mode single \
  --prompt "Once upon a time" \
  --max-tokens 200 \
  --temperature 0.8
```

### Batch Generation

Generate from multiple prompts in a file:

```bash
# Create prompts file
cat > prompts.txt << EOF
The future of AI is
In a world where technology
Once upon a time in a land
EOF

python -m yagpt.generate \
  --checkpoint ./checkpoints/checkpoint_iter_10000.pt \
  --mode batch \
  --prompts-file prompts.txt \
  --max-tokens 100
```

### Generation Parameters

- **`--temperature`** (default: 0.8)
  - Controls randomness: lower = more deterministic, higher = more creative
  - Range: 0.0 (greedy) to 2.0+
  - Sweet spot: 0.7-1.0

- **`--top-k`** (default: 40)
  - Only sample from top k most likely tokens
  - Prevents sampling unlikely tokens
  - Common values: 20-50

- **`--top-p`** (default: 0.9)
  - Nucleus sampling: sample from tokens comprising top p probability mass
  - Range: 0.0-1.0
  - Common values: 0.9-0.95

## Checkpointing

### Automatic Checkpointing

Checkpoints are automatically saved every `checkpoint_interval` iterations (default: 1000).

Each checkpoint contains:
- Model weights
- Optimizer state
- Training iteration
- Configuration
- Loss values

### Checkpoint Management

Only the last N checkpoints are kept (default: 5) to save disk space.

Checkpoints are named: `checkpoint_iter_{iteration}.pt`

Example:
```
checkpoints/
├── checkpoint_iter_1000.pt
├── checkpoint_iter_2000.pt
├── checkpoint_iter_3000.pt
├── checkpoint_iter_4000.pt
└── checkpoint_iter_5000.pt  (latest)
```

### Manual Checkpoint Loading

```python
from yagpt import load_model_from_checkpoint

model, tokenizer = load_model_from_checkpoint(
    checkpoint_path="./checkpoints/checkpoint_iter_5000.pt",
    device="cuda"
)

# Now use the model for generation or fine-tuning
```

## Model Sizes

### Mini (Default)
```python
from yagpt import create_gpt_mini
model = create_gpt_mini()
```
- Parameters: ~85M
- Layers: 12, Heads: 12, Dim: 768
- Memory: ~1GB
- Good for: Experimentation, small datasets

### Custom Size
```python
from yagpt import GPT, GPTConfig

config = GPTConfig(
    n_layer=48,
    n_head=24,
    n_embd=2048,
    # ... other params
)
model = GPT(config)
```

## Training Tips

### Memory Optimization

If you run out of memory:

1. **Reduce batch size**:
   ```python
   config.batch_size = 4
   ```

2. **Increase gradient accumulation**:
   ```python
   config.gradient_accumulation_steps = 8
   # Effective batch size = 4 * 8 = 32
   ```

3. **Reduce sequence length**:
   ```python
   config.block_size = 1024  # Instead of 2048
   ```

4. **Enable gradient checkpointing** (implement in model.py):
   ```python
   torch.utils.checkpoint.checkpoint(block, x)
   ```

5. **Use mixed precision** (add to train.py):
   ```python
   from torch.cuda.amp import autocast, GradScaler
   ```

### Learning Rate Guidelines

- **From scratch**: 3e-4 to 6e-4
- **Fine-tuning**: 1e-4 to 3e-5
- **Warmup**: 2000-4000 iterations recommended

### Data Preparation

Your data should be:
- Sharded into parquet files
- Each parquet has a 'text' column with raw text
- Located in a directory (e.g., `./datasets/fineweb/`)

Example data structure:
```python
# Each parquet file contains:
{
    'text': [
        "First document text...",
        "Second document text...",
        ...
    ]
}
```

## Logging & Monitoring

YAGPT includes a flexible multi-backend logging system. See [LOGGING.md](docs/LOGGING.md) for complete documentation.

### Quick Setup

**Default (Console + CSV)** - No additional setup:
```python
config = TrainingConfig()
# Automatically logs to console and CSV files
train(config)
```

**Add Weights & Biases**:
```bash
wandb login
```
```python
config.log_backends = ['console', 'csv', 'wandb']
config.wandb_project = "my-gpt-project"
```

**Add TensorBoard**:
```python
config.log_backends = ['console', 'csv', 'tensorboard']
```
```bash
tensorboard --logdir=./runs
```

### Logged Metrics

- **Training**: loss, learning rate, step time
- **Validation**: loss, perplexity
- **System**: GPU memory, throughput
- **Config**: All hyperparameters

Example console output:
```
step    100 | loss=3.2451 | lr=2.85e-04 | step_time_ms=234.56
step    110 | loss=3.2123 | lr=2.86e-04 | step_time_ms=231.23
```

## License

This is an educational implementation. Use for learning and experimentation.

## References

1. **Attention Is All You Need** - Vaswani et al., 2017
2. **Improving Language Understanding by Generative Pre-Training** - Radford et al., 2018 (GPT-1)
3. **Language Models are Unsupervised Multitask Learners** - Radford et al., 2019 (GPT-2)
4. **Language Models are Few-Shot Learners** - Brown et al., 2020 (GPT-3)
5. **Training Compute-Optimal Large Language Models** - Hoffmann et al., 2022 (Chinchilla)

## Acknowledgments

- Based on the GPT architecture from OpenAI
- Inspired by Andrej Karpathy's nanoGPT
- Uses FineWeb dataset for training
