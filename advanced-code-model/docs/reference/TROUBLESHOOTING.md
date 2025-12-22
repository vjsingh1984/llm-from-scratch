# Troubleshooting Guide

> Common issues and how to fix them.

---

## Quick Fixes

| Issue | Quick Fix |
|-------|-----------|
| Out of memory | `--batch-size 1 --gradient-checkpointing` |
| Slow training | `--use-compile --use-amp` |
| Loss not decreasing | Increase learning rate |
| Tokenizer error | `python scripts/train_code_tokenizer.py` |
| Checkpoint not found | Check file path |

---

## Training Issues

### Out of Memory (OOM)

**Symptoms:**
```
RuntimeError: MPS backend out of memory
RuntimeError: CUDA out of memory
```

**Solutions:**

1. **Reduce batch size**
   ```bash
   --batch-size 1
   ```

2. **Enable gradient checkpointing**
   ```bash
   --gradient-checkpointing
   ```

3. **Use smaller model**
   ```bash
   --model-size medium
   ```

4. **Clear memory cache**
   ```python
   import torch
   torch.mps.empty_cache()  # For Apple Silicon
   torch.cuda.empty_cache()  # For NVIDIA
   ```

5. **Reduce sequence length**
   ```bash
   --max-length 512
   ```

---

### Training Too Slow

**Symptoms:**
- Less than 1 step/second
- GPU utilization low

**Solutions:**

1. **Enable compilation**
   ```bash
   --use-compile
   ```

2. **Enable mixed precision**
   ```bash
   --use-amp
   ```

3. **Increase batch size** (if memory allows)
   ```bash
   --batch-size 4
   ```

4. **Check data loading**
   ```bash
   --num-workers 4
   ```

---

### Loss Not Decreasing

**Symptoms:**
- Loss stays flat
- Training loss very high (>10)

**Solutions:**

1. **Increase learning rate**
   ```bash
   --learning-rate 3e-4  # Try 3x higher
   ```

2. **Check data quality**
   ```bash
   python scripts/validate_data.py --data-file your_data.npy
   ```

3. **Reduce warmup**
   ```bash
   --warmup-steps 500
   ```

4. **Check model initialization**
   - Ensure checkpoint is loading correctly
   - Try training from scratch

---

### Overfitting (Val Loss Increasing)

**Symptoms:**
- Train loss decreasing
- Val loss flat or increasing

**Solutions:**

1. **More training data**
   ```bash
   python scripts/download_code_corpus.py
   ```

2. **Lower learning rate**
   ```bash
   --learning-rate 1e-6
   ```

3. **Add regularization**
   ```bash
   --dropout 0.1
   --weight-decay 0.1
   ```

4. **Early stopping** (save best checkpoint)

---

### NaN or Inf Loss

**Symptoms:**
```
Loss: nan
Loss: inf
```

**Solutions:**

1. **Lower learning rate**
   ```bash
   --learning-rate 1e-5
   ```

2. **Enable gradient clipping**
   ```bash
   --grad-clip 0.5
   ```

3. **Check for bad data**
   ```bash
   python -c "
   import numpy as np
   data = np.load('data/processed/train.npy')
   print('Contains NaN:', np.isnan(data).any())
   print('Contains Inf:', np.isinf(data).any())
   "
   ```

4. **Disable AMP temporarily**
   - Remove `--use-amp`

---

## Tokenizer Issues

### "Tokenizer not found"

**Symptoms:**
```
FileNotFoundError: data/tokenizer/tokenizer.json
```

**Solution:**
```bash
python scripts/train_code_tokenizer.py
```

---

### Wrong Token Spacing

**Symptoms:**
```
Output: "#!/ bin / bash"
Expected: "#!/bin/bash"
```

**Cause:** Tokenizer trained on natural language, not code.

**Solution:**
```bash
# Retrain tokenizer on code
python scripts/train_code_tokenizer.py

# Re-prepare all data
python scripts/prepare_data.py
python scripts/prepare_code_data.py

# Retrain model from scratch
python scripts/train.py --stage language
```

---

### Unknown Tokens

**Symptoms:**
- `<unk>` appearing in output
- Rare words not tokenized correctly

**Solution:**
- Increase vocabulary size
  ```bash
  python scripts/train_code_tokenizer.py --vocab-size 50000
  ```

---

## Checkpoint Issues

### "Checkpoint not found"

**Symptoms:**
```
FileNotFoundError: models/code_model_best.pth
```

**Solutions:**

1. **Check file exists**
   ```bash
   ls -la models/
   ```

2. **Check path is correct**
   ```bash
   --checkpoint models/language_model_best.pth
   ```

3. **Train previous stage first**
   ```bash
   python scripts/train.py --stage language
   ```

---

### "Size mismatch"

**Symptoms:**
```
RuntimeError: size mismatch for embedding.weight
```

**Cause:** Model config doesn't match checkpoint.

**Solutions:**

1. **Use correct model size**
   ```bash
   --model-size large  # Must match checkpoint
   ```

2. **Check tokenizer vocabulary**
   - Checkpoint vocab size must match tokenizer

3. **Retrain if vocabulary changed**
   - After retraining tokenizer, must retrain model

---

### "_orig_mod" Prefix Error

**Symptoms:**
```
RuntimeError: Error(s) in loading state_dict:
    Missing key(s): "embedding.weight"
    Unexpected key(s): "_orig_mod.embedding.weight"
```

**Cause:** Checkpoint saved from compiled model.

**Solution:** Strip prefix when loading:
```python
state_dict = checkpoint['model_state_dict']
state_dict = {
    key.replace('_orig_mod.', ''): value
    for key, value in state_dict.items()
}
model.load_state_dict(state_dict)
```

---

## Generation Issues

### Garbage Output

**Symptoms:**
- Random characters
- Repeated tokens
- Incoherent text

**Solutions:**

1. **Check model loaded correctly**
   ```python
   print(model.state_dict().keys())
   ```

2. **Check tokenizer matches**
   - Must use same tokenizer as training

3. **Adjust temperature**
   ```python
   temperature = 0.7  # Lower = more focused
   ```

4. **Use top-k/top-p sampling**
   ```python
   top_k = 50
   top_p = 0.95
   ```

---

### Repetitive Output

**Symptoms:**
- Same phrase repeated
- Gets stuck in loop

**Solutions:**

1. **Increase temperature**
   ```python
   temperature = 1.0
   ```

2. **Add repetition penalty**
   ```python
   repetition_penalty = 1.2
   ```

3. **Use top-p sampling**
   ```python
   top_p = 0.9
   ```

---

## Environment Issues

### MPS (Apple Silicon) Issues

**"MPS backend out of memory"**
```bash
export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0
```

**"MPS not available"**
```python
import torch
print(torch.backends.mps.is_available())
# Should be True on M1/M2/M3
```

---

### CUDA Issues

**"CUDA out of memory"**
```bash
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
```

**"CUDA not available"**
```python
import torch
print(torch.cuda.is_available())
# Check NVIDIA drivers and PyTorch CUDA version
```

---

### Import Errors

**"No module named 'src'"**
```bash
# Add to path
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# Or run from project root
cd advanced-code-model
python scripts/train.py
```

---

## Data Issues

### "Data file not found"

**Solution:**
```bash
# Check data exists
ls data/processed/

# Prepare data
python scripts/prepare_data.py
python scripts/prepare_code_data.py
```

---

### "Data shape mismatch"

**Symptoms:**
```
ValueError: cannot reshape array of size X into shape Y
```

**Solution:**
- Re-prepare data with correct sequence length
  ```bash
  python scripts/prepare_data.py --max-length 1024
  ```

---

### Empty or Corrupt Data

**Check data quality:**
```bash
python -c "
import numpy as np
data = np.load('data/processed/train.npy')
print('Shape:', data.shape)
print('Min:', data.min())
print('Max:', data.max())
print('Sample:', data[0, :10])
"
```

---

## Getting Help

1. **Check this guide** for common issues
2. **Check script help**: `python scripts/train.py --help`
3. **Check logs**: Look for error messages above the crash
4. **Minimal reproduction**: Try with tiny model first
5. **Report issue**: Include error message, command, and environment
