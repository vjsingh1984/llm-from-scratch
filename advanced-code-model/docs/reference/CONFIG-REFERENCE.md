# Configuration Reference

> All configuration options explained.

---

## Model Configuration

### Model Sizes

| Size | d_model | n_layers | n_heads | d_ff | max_seq_len | vocab_size | Parameters |
|------|---------|----------|---------|------|-------------|------------|------------|
| tiny | 256 | 4 | 4 | 1024 | 512 | 32000 | ~10M |
| medium | 768 | 12 | 12 | 3072 | 1024 | 32000 | ~150M |
| large | 1600 | 32 | 25 | 6400 | 4096 | 32000 | ~1.6B |
| xlarge | 2048 | 40 | 32 | 8192 | 8192 | 32000 | ~7B |

### Using Configurations

```python
from src.model.config import get_config

# Get predefined config
config = get_config('large')

# Customize
config.use_rmsnorm = True
config.use_rope = True
config.dropout = 0.1
```

---

## Model Architecture Options

### use_rmsnorm

Use RMSNorm instead of LayerNorm.

```python
config.use_rmsnorm = True
```

**Benefits:**
- Slightly faster training
- More stable gradients
- Used in modern models (LLaMA, Mistral)

---

### use_rope

Use Rotary Position Embeddings.

```python
config.use_rope = True
```

**Benefits:**
- Better length generalization
- Relative position encoding
- Extrapolates to longer sequences

---

### dropout

Dropout rate for regularization.

```python
config.dropout = 0.1  # 10% dropout
```

**Guidelines:**
- 0.0 for large datasets
- 0.1 for medium datasets
- 0.2+ for small datasets or overfitting

---

### max_seq_len

Maximum sequence length.

```python
config.max_seq_len = 4096
```

**Memory impact:**
- Attention is O(n²) in sequence length
- Doubling length quadruples memory

---

## Training Configuration

### Learning Rate

```python
# Command line
--learning-rate 1e-4

# Recommended values
# Stage 1 (Language): 1e-4
# Stage 2 (Code): 5e-6 to 1e-5
# Stage 3 (Tools): 3e-6
# Stage 4 (RLHF): 1e-6
```

---

### Warmup Steps

```python
--warmup-steps 2000
```

Gradual increase from 0 to target learning rate.

**Guidelines:**
- More steps = safer, slower start
- Typical: 5-10% of total steps

---

### Weight Decay

```python
--weight-decay 0.1
```

L2 regularization on weights.

**Guidelines:**
- 0.1 is standard
- 0.01 for fine-tuning
- 0.0 if you have dropout

---

### Gradient Clipping

```python
--grad-clip 1.0
```

Clips gradients to prevent explosion.

**Guidelines:**
- 1.0 is safe default
- Lower (0.5) for instability
- Higher (5.0) for slow training

---

### Batch Size

```python
--batch-size 2
--grad-accumulation-steps 4
# Effective batch size = 2 * 4 = 8
```

**Memory guidelines:**
| VRAM | Batch Size | Grad Accum |
|------|------------|------------|
| 16GB | 1 | 8 |
| 24GB | 2 | 4 |
| 48GB | 4 | 2 |
| 80GB | 8 | 1 |

---

## Optimization Options

### use_compile

Enable torch.compile for faster training.

```python
--use-compile
```

**Benefits:**
- 10-30% speedup
- Automatic kernel fusion
- Works best with PyTorch 2.0+

**Caveats:**
- First epoch is slower (compilation)
- May not work with all architectures

---

### use_amp

Enable Automatic Mixed Precision (FP16/BF16).

```python
--use-amp
```

**Benefits:**
- ~2x memory reduction
- ~1.5x speedup
- Minimal accuracy loss

**Requirements:**
- GPU with tensor cores (NVIDIA Ampere+)
- Or Apple Silicon (BFloat16)

---

### gradient_checkpointing

Trade compute for memory.

```python
--gradient-checkpointing
```

**Benefits:**
- ~40% memory reduction
- Allows larger batch sizes

**Caveats:**
- ~20% slower training

---

## Data Configuration

### Data Files

```python
--data-dir data/processed
--data-file code_train_large.npy
--val-file code_val_large.npy
```

**File format:**
- NumPy arrays of shape `(num_sequences, seq_length)`
- dtype: int32 or int64
- Token IDs from tokenizer

---

### Sequence Length

```python
--max-length 1024  # During data preparation
```

**Guidelines:**
- 512 for language pretraining
- 1024-2048 for code
- 4096+ for long context

---

## Tokenizer Configuration

### Vocabulary Size

```python
--vocab-size 32000
```

**Trade-offs:**
| Vocab Size | Tokens/Text | Memory |
|------------|-------------|--------|
| 8000 | More | Less |
| 32000 | Balanced | Medium |
| 100000 | Fewer | More |

---

### Special Tokens

```python
special_tokens = [
    "<pad>",      # Padding
    "<unk>",      # Unknown
    "<bos>",      # Beginning of sequence
    "<eos>",      # End of sequence
    "<sep>",      # Separator
    "### Language:",  # Language marker
    "### Code:",      # Code marker
    "### End",        # End marker
    "<tool_call>",    # Tool call start
    "</tool_call>",   # Tool call end
]
```

---

## Environment Variables

### Device Selection

```bash
# Force specific device
export CUDA_VISIBLE_DEVICES=0  # Use GPU 0
export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0  # Unlimited MPS memory
```

### Memory Settings

```bash
# PyTorch memory
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

# Apple Silicon
export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.7
```

### Logging

```bash
export LOG_LEVEL=INFO
export WANDB_PROJECT=code-model  # For wandb logging
```

---

## YAML Configuration (Optional)

You can use YAML files in `configs/`:

```yaml
# configs/model/large.yaml
name: large
vocab_size: 32000
d_model: 1600
n_layers: 32
n_heads: 25
d_ff: 6400
max_seq_len: 4096
dropout: 0.0
use_rmsnorm: true
use_rope: true
```

```yaml
# configs/training/code.yaml
stage: code
batch_size: 2
learning_rate: 5e-6
num_epochs: 5
warmup_steps: 1000
weight_decay: 0.1
grad_clip: 1.0
use_compile: true
use_amp: true
```

**Usage:**

```bash
python scripts/train.py --config configs/training/code.yaml
```

---

## Default Values Summary

| Parameter | Default | Range |
|-----------|---------|-------|
| model_size | large | tiny/medium/large/xlarge |
| batch_size | 2 | 1-32 |
| learning_rate | 1e-4 | 1e-6 to 1e-3 |
| warmup_steps | 2000 | 100-10000 |
| weight_decay | 0.1 | 0.0-0.3 |
| grad_clip | 1.0 | 0.1-10.0 |
| dropout | 0.0 | 0.0-0.5 |
| num_epochs | 3 | 1-100 |
| max_seq_len | 4096 | 512-32768 |
| vocab_size | 32000 | 8000-100000 |
