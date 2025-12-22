# Configuration Files

YAML configuration files for models and training.

## Model Configurations

| Config | Parameters | VRAM | Use Case |
|--------|------------|------|----------|
| [model/tiny.yaml](model/tiny.yaml) | ~10M | 1GB | Testing, CI/CD |
| [model/medium.yaml](model/medium.yaml) | ~150M | 4GB | Development |
| [model/large.yaml](model/large.yaml) | ~1.6B | 12GB | **Production** (Recommended) |
| [model/xlarge.yaml](model/xlarge.yaml) | ~7B | 24GB | Best quality |

## Training Configurations

| Config | Stage | Description |
|--------|-------|-------------|
| [training/language.yaml](training/language.yaml) | 1 | Language pretraining on TinyStories |
| [training/code.yaml](training/code.yaml) | 2 | Code fine-tuning |
| [training/tools.yaml](training/tools.yaml) | 3 | Tool calling training |
| [training/rlhf.yaml](training/rlhf.yaml) | 4 | RLHF alignment |

## Usage

```bash
# Using config files (when supported)
python scripts/train.py --config configs/training/code.yaml

# Or specify options directly
python scripts/train.py --stage code --model-size large
```

## Key Parameters

### Model Config
```yaml
d_model: 1600      # Hidden dimension
n_layers: 32       # Number of transformer blocks
n_heads: 25        # Attention heads
d_ff: 6400         # FFN dimension
max_seq_len: 4096  # Maximum sequence length
```

### Training Config
```yaml
batch_size: 2
learning_rate: 5e-6
num_epochs: 5
warmup_steps: 1000
```

See [docs/reference/CONFIG-REFERENCE.md](../docs/reference/CONFIG-REFERENCE.md) for all options.
