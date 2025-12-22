# Training Pipeline

> Complete guide to training your own code model from scratch.

---

![Training Pipeline](diagrams/training-pipeline.svg)

---

## Overview

Training is done in **stages**, each building on the previous checkpoint:

| Stage | Purpose | Input Data | Output |
|-------|---------|------------|--------|
| 0 | Tokenizer | Code + Text | `data/tokenizer/tokenizer.json` |
| 1 | Language | TinyStories | `models/language_model_best.pth` |
| 2 | Code | Bash, Python, etc. | `models/code_model_best.pth` |
| 3 | Tools | Function examples | `models/tool_model_best.pth` |
| 4 | RLHF | Preferences | `models/rlhf_model_best.pth` |
| 5 | Deploy | - | API server |

---

## Prerequisites

```bash
# Check you have the right environment
python --version  # Should be 3.10+
pip install -r requirements.txt

# Check device
python -c "import torch; print(torch.backends.mps.is_available())"  # True for Apple Silicon
python -c "import torch; print(torch.cuda.is_available())"          # True for CUDA
```

---

## Stage 0: Train Tokenizer

The tokenizer converts text to numbers. We train it on code to get good code tokens.

### Step 1: Download Data

```bash
# Download code corpus from HuggingFace
python scripts/download_code_corpus.py

# This creates:
# - data/raw/code_corpus.txt (75K+ examples)
```

### Step 2: Train Tokenizer

```bash
python scripts/train_code_tokenizer.py --vocab-size 32000
```

**Output:**
```
data/tokenizer/tokenizer.json    # Main tokenizer file
data/tokenizer/tokenizer.json.backup  # Backup of old tokenizer
```

### Verify Tokenizer

```bash
python -c "
from tokenizers import Tokenizer
t = Tokenizer.from_file('data/tokenizer/tokenizer.json')
print(t.encode('#!/bin/bash').tokens)
# Should output: ['#!/', 'bin', '/', 'bash'] or similar
"
```

---

## Stage 1: Language Model

Train basic language understanding using TinyStories.

### Step 1: Download TinyStories

```bash
python scripts/download_data.py
# Downloads ~500MB of children's stories
```

### Step 2: Prepare Data

```bash
python scripts/prepare_data.py
# Creates: data/processed/train_large.npy, val_large.npy
```

### Step 3: Train

```bash
python scripts/train.py \
    --stage language \
    --model-size large \
    --batch-size 2 \
    --num-epochs 3 \
    --learning-rate 1e-4 \
    --warmup-steps 2000 \
    --use-rmsnorm --use-rope --use-compile --use-amp
```

**Expected output:**
```
Epoch 1/3: train_loss=4.5, val_loss=4.2
Epoch 2/3: train_loss=3.8, val_loss=3.5
Epoch 3/3: train_loss=3.2, val_loss=3.0
Saved: models/language_model_best.pth
```

### Verify

```bash
python scripts/test_stage1.py
# Should generate coherent stories
```

---

## Stage 2: Code Model

Fine-tune on code to learn programming patterns.

### Step 1: Prepare Code Data

```bash
python scripts/prepare_code_data.py
# Creates: data/processed/code_train_large.npy, code_val_large.npy
```

### Step 2: Train

```bash
python scripts/train.py \
    --stage code \
    --checkpoint models/language_model_best.pth \
    --data-file code_train_large.npy \
    --model-size large \
    --batch-size 2 \
    --num-epochs 5 \
    --learning-rate 5e-6 \
    --warmup-steps 1000 \
    --use-rmsnorm --use-rope --use-compile --use-amp
```

**Key differences from Stage 1:**
- Lower learning rate (5e-6 vs 1e-4)
- Loads previous checkpoint
- Uses code dataset

### Verify

```bash
python scripts/test_stage2.py
# Should generate bash scripts, Python code
```

---

## Stage 3: Tool Calling

Teach the model to use function calling.

### Step 1: Prepare Tool Data

```bash
python scripts/prepare_tool_calling_data.py
# Creates synthetic tool calling examples
```

### Step 2: Train

```bash
python scripts/train.py \
    --stage tools \
    --checkpoint models/code_model_best.pth \
    --data-file tool_train.npy \
    --model-size large \
    --batch-size 2 \
    --num-epochs 3 \
    --learning-rate 3e-6 \
    --use-rmsnorm --use-rope --use-compile --use-amp
```

### Verify

```bash
python scripts/test_tool_calling.py
# Should output proper function calls
```

---

## Stage 4: RLHF (Optional)

Align the model with human preferences.

### Step 1: Prepare Preference Data

```bash
python scripts/prepare_rlhf_data.py
# Creates: data/processed/preferences.json
```

### Step 2: Train Reward Model

```bash
python scripts/train_reward_model.py \
    --checkpoint models/tool_model_best.pth \
    --data-file preferences.json
# Creates: models/reward_model.pth
```

### Step 3: Train with PPO

```bash
python scripts/train_rlhf.py \
    --policy-model models/tool_model_best.pth \
    --reward-model models/reward_model.pth \
    --num-epochs 1 \
    --ppo-epochs 4
# Creates: models/rlhf_model_best.pth
```

---

## Training Parameters Reference

### Model Sizes

| Size | d_model | n_layers | n_heads | d_ff | Params |
|------|---------|----------|---------|------|--------|
| tiny | 256 | 4 | 4 | 1024 | 10M |
| medium | 768 | 12 | 12 | 3072 | 150M |
| large | 1600 | 32 | 25 | 6400 | 1.6B |
| xlarge | 2048 | 40 | 32 | 8192 | 7B |

### Learning Rates by Stage

| Stage | Recommended LR | Notes |
|-------|----------------|-------|
| Language | 1e-4 | Training from scratch |
| Code | 5e-6 to 1e-5 | Fine-tuning |
| Tools | 3e-6 | Fine-tuning |
| RLHF | 1e-6 | Very careful updates |

### Batch Size vs VRAM

| VRAM | Recommended Batch Size | Gradient Accumulation |
|------|------------------------|----------------------|
| 16GB | 1 | 8 |
| 24GB | 2 | 4 |
| 48GB | 4 | 2 |
| 80GB | 8 | 1 |

---

## Training Tips

### Preventing Overfitting

```bash
# Signs of overfitting:
# - Train loss decreasing
# - Val loss stagnant or increasing

# Solutions:
# 1. More data
python scripts/download_code_corpus.py  # Get more data

# 2. Lower learning rate
--learning-rate 1e-6

# 3. More regularization
--weight-decay 0.1
--dropout 0.1
```

### Checkpointing

```bash
# Resume from checkpoint
python scripts/train.py \
    --checkpoint models/code_model_epoch_2.pth \
    --resume
```

### Monitoring Training

```bash
# Use TensorBoard
tensorboard --logdir logs/

# Or check loss directly
tail -f logs/training.log
```

---

## Troubleshooting

### Out of Memory

```bash
# Reduce batch size
--batch-size 1

# Enable gradient checkpointing
--gradient-checkpointing

# Use smaller model
--model-size medium
```

### Training Too Slow

```bash
# Enable compilation
--use-compile

# Enable mixed precision
--use-amp

# Reduce gradient accumulation
--grad-accumulation-steps 1
```

### Loss Not Decreasing

```bash
# Increase learning rate
--learning-rate 3e-4

# Check data quality
python scripts/validate_data.py

# Reduce warmup
--warmup-steps 500
```

---

## Full Training Script

Here's a complete script to train all stages:

```bash
#!/bin/bash
set -e

echo "=== Stage 0: Tokenizer ==="
python scripts/download_code_corpus.py
python scripts/train_code_tokenizer.py

echo "=== Stage 1: Language ==="
python scripts/download_data.py
python scripts/prepare_data.py
python scripts/train.py --stage language --model-size large \
    --batch-size 2 --num-epochs 3 --learning-rate 1e-4 \
    --use-rmsnorm --use-rope --use-compile --use-amp

echo "=== Stage 2: Code ==="
python scripts/prepare_code_data.py
python scripts/train.py --stage code \
    --checkpoint models/language_model_best.pth \
    --data-file code_train_large.npy \
    --batch-size 2 --num-epochs 5 --learning-rate 5e-6 \
    --use-rmsnorm --use-rope --use-compile --use-amp

echo "=== Stage 3: Tools ==="
python scripts/prepare_tool_calling_data.py
python scripts/train.py --stage tools \
    --checkpoint models/code_model_best.pth \
    --data-file tool_train.npy \
    --batch-size 2 --num-epochs 3 --learning-rate 3e-6 \
    --use-rmsnorm --use-rope --use-compile --use-amp

echo "=== Done! ==="
echo "Model saved to: models/tool_model_best.pth"
```

---

## Next Steps

- **Deploy your model**: [05-DEPLOYMENT.md](05-DEPLOYMENT.md)
- **Understand the architecture**: [03-ARCHITECTURE.md](03-ARCHITECTURE.md)
- **Script reference**: [reference/SCRIPT-REFERENCE.md](reference/SCRIPT-REFERENCE.md)
