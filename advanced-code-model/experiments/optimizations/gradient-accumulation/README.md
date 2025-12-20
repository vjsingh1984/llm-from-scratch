# Gradient Accumulation

## Overview
Simulate larger batch sizes by accumulating gradients over multiple forward passes before updating weights.

## Problem
- Want batch_size=16 for better gradients
- Can only fit batch_size=4 in memory

## Solution
- Do 4 forward passes with batch_size=4
- Accumulate gradients
- Update once (equivalent to batch_size=16)

## Implementation

```python
# In train_epoch()
accumulation_steps = 4
optimizer.zero_grad()

for step in range(num_steps):
    x, y = get_batch(train_data, batch_size, seq_len, device)

    # Forward pass
    loss = compute_loss(model, x, y)

    # Scale loss by accumulation steps
    loss = loss / accumulation_steps

    # Backward pass (accumulate gradients)
    loss.backward()

    # Update every N steps
    if (step + 1) % accumulation_steps == 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
```

## Benefits
- **Effective batch size**: batch_size × accumulation_steps
- **Same memory**: No additional memory needed
- **Better gradients**: Less noisy, more stable training
- **Same convergence**: Equivalent to larger batch size

## Trade-offs
- **Slightly slower**: More forward/backward passes
- **Different dynamics**: Batch normalization behaves differently
- **Logging complexity**: Need to track accumulated loss

## Parameters
- `accumulation_steps=4`: Most common (4x effective batch size)
- `accumulation_steps=8`: Very large effective batches
- `accumulation_steps=2`: Modest improvement

## When to Use
- Limited GPU memory
- Want larger effective batch size
- Training is unstable with small batches

## Expected Results
With Medium model:
- Current: batch_size=4
- With accumulation_steps=4: effective batch_size=16
- Expected: 10-15% better final loss, smoother training

## Status
- [ ] Implemented
- [ ] Tested
- [ ] Benchmarked
- [ ] Documented

## Files
- `gradient_accumulation.py` - Implementation
- `benchmark_results.md` - Performance comparison
- `plots/` - Training curves
