# Training Your LLM: From Data to Generation

## Overview

Now that we have a working model and tokenizer, let's train it! This guide covers the complete training pipeline, from data loading to text generation.

## Quick Start

```bash
# 1. Download data
python scripts/download_data.py --num-samples 10000

# 2. Train tokenizer
python scripts/train_tokenizer.py --vocab-size 8000

# 3. Train model
python scripts/train.py \
    --model-size tiny \
    --batch-size 16 \
    --max-epochs 10 \
    --learning-rate 3e-4

# 4. Generate text
python scripts/generate.py \
    --checkpoint checkpoints/best.npz \
    --prompt "Once upon a time" \
    --interactive
```

---

## 1. Data Preparation

### Loading Text Data

Our data pipeline handles:
1. Loading tokenized text
2. Creating overlapping sequences
3. Batching for efficient training

**Implementation** (`training/data_loader.py`):

```python
class TextDataset:
    def __init__(self, token_ids: List[int], seq_len: int):
        self.token_ids = token_ids
        self.seq_len = seq_len

    def __getitem__(self, idx):
        start = idx * self.seq_len
        # Input: tokens[start:start+seq_len]
        # Target: tokens[start+1:start+seq_len+1] (shifted by 1)
        return input_seq, target_seq
```

### Next-Token Prediction

Language models learn by predicting the next token:

```
Input:  "The cat sat on the"
Target: "cat sat on the mat"
```

Each position predicts the next token:
- Position 0: "The" → predict "cat"
- Position 1: "cat" → predict "sat"
- Position 2: "sat" → predict "on"
- ...

### Batching

Group sequences into batches for parallel processing:

```
Batch shape: [batch_size, seq_len]
Example: [16, 256] = 16 sequences of 256 tokens each
```

**Memory usage**: batch_size × seq_len × model_params × 4 bytes

For Tiny model (50M params) with batch size 16, seq len 256:
- Forward pass: ~500MB
- Training (with gradients): ~2GB

---

## 2. Training Loop

### Core Training Step

```python
def train_step(inputs, targets):
    # 1. Forward pass
    logits, loss = model(inputs, targets)

    # 2. Backward pass (compute gradients)
    grads = compute_gradients(loss)

    # 3. Clip gradients (prevent explosion)
    grads = clip_gradients(grads, max_norm=1.0)

    # 4. Update weights
    optimizer.update(model, grads)

    return loss
```

### Loss Function: Cross-Entropy

Measures how well the model predicts the next token:

```python
# Model outputs: probability distribution over vocabulary
logits = model(inputs)  # [batch, seq_len, vocab_size]

# Targets: actual next tokens
targets = ...  # [batch, seq_len]

# Cross-entropy loss
loss = -log(P(target_token | context))
```

**Lower loss** = better predictions

**Typical loss values**:
- Random initialization: ~ln(vocab_size) ≈ 9.0 for vocab_size=8000
- After training: 2.0-4.0 (good), <2.0 (excellent)

### Perplexity

Another common metric: `perplexity = exp(loss)`

- Perplexity of 20: Model is "confused" between ~20 tokens on average
- Lower is better

---

## 3. Optimization with AdamW

### Why AdamW?

**AdamW** (Adam with Weight Decay) is the standard optimizer for transformers:
- Adaptive learning rates per parameter
- Momentum for smooth convergence
- Weight decay for regularization

```python
optimizer = AdamW(
    learning_rate=3e-4,
    betas=(0.9, 0.95),      # Momentum parameters
    weight_decay=0.1,        # L2 regularization
    eps=1e-8                 # Numerical stability
)
```

### Hyperparameters Explained

**Learning Rate** (3e-4 is standard for transformers):
- Too high: Training unstable, loss explodes
- Too low: Training too slow, might not converge
- Range to try: 1e-4 to 6e-4

**Betas** (0.9, 0.95):
- β1=0.9: Short-term momentum
- β2=0.95: Long-term momentum (higher for transformers vs CNNs)

**Weight Decay** (0.1):
- Regularization to prevent overfitting
- Applied to all weights except biases and layer norms

---

## 4. Learning Rate Schedule

Learning rate should change during training for best results.

### Warmup + Cosine Decay

**Standard schedule for transformers**:

```
         max_lr ----___
               /        \___
              /             \___
             /                  \___
    0 -----/                        \___ min_lr
         warmup                 cosine decay
```

**Warmup** (first 2000 steps typically):
- Gradually increase LR from 0 to max_lr
- Prevents instability at start

**Cosine Decay**:
- Smoothly decrease LR using cosine function
- Allows fine-tuning at end

**Implementation**:
```python
def cosine_schedule(step):
    if step < warmup_steps:
        return max_lr * step / warmup_steps

    progress = (step - warmup_steps) / (total_steps - warmup_steps)
    return min_lr + (max_lr - min_lr) * 0.5 * (1 + cos(π * progress))
```

### Choosing Warmup Steps

- **Small models** (<100M): 500-2000 steps
- **Large models** (>1B): 2000-10000 steps
- Rule of thumb: ~1% of total training steps

---

## 5. Gradient Clipping

Prevents gradient explosion (especially early in training):

```python
def clip_gradients(grads, max_norm=1.0):
    # Compute global gradient norm
    total_norm = sqrt(sum(||grad||² for grad in grads))

    # Clip if too large
    if total_norm > max_norm:
        scale = max_norm / total_norm
        grads = {k: v * scale for k, v in grads.items()}

    return grads
```

**max_norm=1.0** is standard for transformers.

---

## 6. Gradient Accumulation

Train with larger effective batch size on limited memory:

```python
# Effective batch size = batch_size × accumulation_steps
# Example: 8 × 4 = 32 effective batch size

for i in range(accumulation_steps):
    loss = train_step(mini_batch)
    accumulate_gradients()

# Update weights once after accumulating
optimizer.step()
reset_gradients()
```

**When to use**:
- M1 Max with 32GB: Use for models >350M params
- Allows training larger models or longer sequences
- No quality loss vs larger batch size

---

## 7. Monitoring Training

### Metrics to Track

**1. Training Loss**:
- Should steadily decrease
- If plateaus early: increase LR or model capacity
- If explodes: decrease LR or check data

**2. Validation Loss**:
- Most important metric
- Prevents overfitting
- If train loss ↓ but val loss ↑: you're overfitting

**3. Learning Rate**:
- Verify warmup and decay working correctly

**4. Gradient Norm**:
- Should be < max_clip_norm (1.0)
- If constantly clipping: might need to adjust LR

**5. Tokens/Second**:
- Training throughput
- M1 Max typical: 5000-50000 tokens/sec depending on model size

### Text Generation During Training

Sample generation helps monitor quality:

```python
# Every 500 steps, generate samples
prompts = ["Once upon a time", "The cat", "In space"]
for prompt in prompts:
    generated = model.generate(prompt, max_tokens=50)
    print(generated)
```

**What to look for**:
- Early: gibberish or repetition
- Mid: some structure, vocabulary improving
- Late: coherent sentences, proper grammar

---

## 8. Checkpointing

Save model regularly to prevent data loss:

```python
# Save every 1000 steps
if step % 1000 == 0:
    model.save_weights(f"checkpoint_step_{step}.npz")

# Save best model (lowest validation loss)
if val_loss < best_val_loss:
    model.save_weights("best_model.npz")
```

**What to save**:
- Model weights (.npz file)
- Training state (step, epoch, best loss)
- Optimizer state (for resuming training)

---

## 9. Training Schedule for M1 Max

### Tiny Model (50M params) - Learning & Debugging

```bash
python scripts/train.py \
    --model-size tiny \
    --batch-size 32 \
    --seq-len 256 \
    --max-steps 10000 \
    --learning-rate 3e-4 \
    --warmup-steps 500
```

**Expected**:
- Time: ~30-60 minutes
- Memory: ~2-3GB
- Validation loss: 3.0-4.0 on TinyStories
- Good for learning and experimentation

### GPT-2 Small (124M params) - Quality Results

```bash
python scripts/train.py \
    --model-size gpt2-small \
    --batch-size 16 \
    --seq-len 512 \
    --max-steps 50000 \
    --learning-rate 3e-4 \
    --warmup-steps 2000
```

**Expected**:
- Time: 4-8 hours
- Memory: ~8-12GB
- Validation loss: 2.0-3.0 on TinyStories
- Coherent, grammatical text

### GPT-2 Medium (350M params) - High Quality

```bash
python scripts/train.py \
    --model-size gpt2-medium \
    --batch-size 8 \
    --seq-len 512 \
    --max-steps 100000 \
    --learning-rate 2e-4 \
    --warmup-steps 4000 \
    --grad-accumulation-steps 2  # Effective batch 16
```

**Expected**:
- Time: 12-24 hours
- Memory: ~16-20GB
- Validation loss: 1.5-2.5
- Very high quality text

---

## 10. Troubleshooting

### Loss is NaN or Exploding

**Causes**:
- Learning rate too high
- Numerical instability

**Solutions**:
- Reduce LR (try 1e-4)
- Increase warmup steps
- Check for bad data (very long sequences)
- Use gradient clipping (already default)

### Loss Not Decreasing

**Causes**:
- LR too low
- Model too small
- Data issues

**Solutions**:
- Increase LR (try 6e-4)
- Use larger model
- Check data quality
- Train longer

### Overfitting (Val Loss Increasing)

**Causes**:
- Training too long
- Model too large for data
- Not enough regularization

**Solutions**:
- Stop training (use best checkpoint)
- Increase weight decay (try 0.2)
- Add more training data
- Use smaller model

### Out of Memory

**Solutions**:
- Reduce batch size
- Reduce sequence length
- Use gradient accumulation
- Use smaller model

---

## 11. Complete Training Example

Here's a complete workflow:

```bash
# Step 1: Download TinyStories dataset
python scripts/download_data.py --num-samples 10000

# Step 2: Train tokenizer
python scripts/train_tokenizer.py \
    --data-file data/tinystories_10000.txt \
    --vocab-size 8000 \
    --output-dir tokenizer_model

# Step 3: Train model
python scripts/train.py \
    --model-size tiny \
    --vocab-size 8000 \
    --seq-len 256 \
    --batch-size 16 \
    --max-steps 10000 \
    --learning-rate 3e-4 \
    --warmup-steps 500 \
    --eval-interval 500 \
    --save-interval 1000 \
    --checkpoint-dir checkpoints

# Step 4: Generate text
python scripts/generate.py \
    --checkpoint checkpoints/best.npz \
    --tokenizer-path tokenizer_model \
    --model-size tiny \
    --prompt "Once upon a time, there was a" \
    --max-tokens 100 \
    --temperature 0.8 \
    --num-samples 3
```

---

## 12. Hyperparameter Tuning Guide

### Priority Order

1. **Learning Rate** (most important)
   - Start: 3e-4
   - Range: 1e-4 to 6e-4
   - Too high if: loss explodes or NaN
   - Too low if: loss barely decreases

2. **Batch Size**
   - Larger = more stable, faster wall-clock time
   - Smaller = better generalization (sometimes)
   - Limited by memory
   - Effective batch (with accumulation): 16-64

3. **Sequence Length**
   - Longer = better context, but more memory
   - 256: fast training
   - 512: good balance
   - 1024+: best quality, slow

4. **Warmup Steps**
   - Typically 1-5% of total steps
   - More for larger models
   - Prevents early instability

5. **Weight Decay**
   - Start: 0.1
   - Increase if overfitting: 0.2
   - Decrease if underfitting: 0.05

### Grid Search Example

```python
for lr in [1e-4, 3e-4, 6e-4]:
    for batch_size in [8, 16, 32]:
        train_model(lr=lr, batch_size=batch_size)
        # Compare validation losses
```

---

## 13. Advanced: Mixed Precision Training

MLX handles mixed precision automatically, but you can optimize further:

```python
# MLX uses float32 by default
# For faster training with minimal quality loss:
mx.set_default_device(mx.gpu)  # Use Metal GPU

# Model will automatically use optimal precision
# for each operation (float32, float16, or bfloat16)
```

**Benefits on M1 Max**:
- Up to 2x faster training
- Reduced memory usage
- No code changes needed (automatic)

---

## Key Takeaways

1. **Start small**: Use Tiny model first to learn the pipeline
2. **Monitor validation loss**: It's your north star
3. **LR schedule matters**: Warmup + cosine decay is standard
4. **Checkpoint often**: Save every 1000 steps minimum
5. **Generate samples**: Qualitative eval during training
6. **M1 Max is powerful**: Can train up to 3B params with care

---

## Next Steps

1. **Train your first model**: Start with Tiny on 10K stories
2. **Experiment**: Try different hyperparameters
3. **Scale up**: Move to GPT-2 Small for quality
4. **Advanced topics**: Read `04-MOE.md` for Mixture of Experts

The training infrastructure is complete. Time to train your LLM!
