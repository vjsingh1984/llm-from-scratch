# Architecture Comparison Guide

**Compare Dense, Mamba, MoE, and Hybrid architectures on the same task**

This guide helps you train and compare different architectures to understand their trade-offs in speed, memory, and quality.

---

## Table of Contents

1. [Quick Start](#quick-start)
2. [Architecture Overview](#architecture-overview)
3. [Training Commands](#training-commands)
4. [Performance Comparison](#performance-comparison)
5. [When to Use Each Architecture](#when-to-use-each-architecture)
6. [Benchmarking Protocol](#benchmarking-protocol)

---

## Quick Start

Train all four architectures on the same task and compare:

```bash
# 1. Dense Transformer (baseline)
python3 scripts/train.py \
  --architecture dense \
  --stage language \
  --model-size tiny \
  --num-epochs 1 \
  --steps-per-epoch 100

# 2. Mamba (SSM)
python3 scripts/train.py \
  --architecture mamba \
  --stage language \
  --model-size tiny \
  --state-size 16 \
  --num-epochs 1 \
  --steps-per-epoch 100

# 3. MoE (Sparse)
python3 scripts/train.py \
  --architecture moe \
  --stage language \
  --model-size tiny \
  --num-experts 8 \
  --expert-capacity 2 \
  --num-epochs 1 \
  --steps-per-epoch 100

# 4. Hybrid (Mamba + Local Attention)
python3 scripts/train.py \
  --architecture hybrid \
  --stage language \
  --model-size tiny \
  --state-size 16 \
  --hybrid-local-window 256 \
  --num-epochs 1 \
  --steps-per-epoch 100
```

---

## Architecture Overview

### 1. Dense Transformer (Baseline)

**What it is**: Standard transformer with multi-head self-attention.

**Architecture**:
- Full self-attention: every token attends to every other token
- O(n²) complexity in sequence length
- All parameters active for every token

**Characteristics**:
- ✅ **Mature**: Most tested and understood
- ✅ **Quality**: Excellent for most tasks
- ✅ **Stable**: Proven training dynamics
- ❌ **Slow**: O(n²) for long sequences
- ❌ **Memory**: O(n²) attention matrix

**Best for**:
- Baseline comparisons
- Tasks with <2048 token sequences
- When stability matters more than speed

**Key Parameters**:
- `--architecture dense`
- `--use-rmsnorm` (optional, faster normalization)
- `--use-rope` (optional, better positions)

---

### 2. Mamba (State Space Model)

**What it is**: Linear-complexity sequence model with selective state spaces.

**Architecture**:
- Selective SSM instead of attention
- O(n) complexity in sequence length
- No position embeddings needed (implicit in SSM)
- Constant memory regardless of sequence length

**Characteristics**:
- ✅ **Fast**: O(n) vs O(n²) attention
- ✅ **Memory**: Constant, not quadratic
- ✅ **Long sequences**: Can handle 10K+ tokens
- ⚠️ **Emerging**: Less mature than transformers
- ⚠️ **Different**: New training dynamics to learn

**Best for**:
- Very long sequences (>4K tokens)
- Memory-constrained environments
- Research into attention alternatives

**Key Parameters**:
- `--architecture mamba`
- `--state-size 16` (state expansion factor, higher = more capacity)
- `--conv-size 4` (convolution kernel size for local patterns)

---

### 3. MoE (Mixture of Experts)

**What it is**: Sparse transformer where different "experts" specialize in different inputs.

**Architecture**:
- Standard attention layers
- FFN replaced with multiple expert networks
- Router decides which experts process each token
- Only top-K experts active per token (sparse)

**Characteristics**:
- ✅ **Scalable**: 10x params with 2x compute
- ✅ **Specialized**: Different experts for different patterns
- ✅ **Efficient**: Sparse activation saves compute
- ⚠️ **Complex**: Router training + load balancing
- ⚠️ **Memory**: More total params (but fewer active)

**Best for**:
- Scaling to large parameter counts efficiently
- Diverse tasks (experts can specialize)
- When you have sufficient data (to train all experts)

**Key Parameters**:
- `--architecture moe`
- `--num-experts 8` (number of expert networks)
- `--expert-capacity 2` (top-K experts per token)

---

### 4. Hybrid (Mamba + Local Attention)

**What it is**: Combines Mamba for global context with local attention for fine-grained patterns.

**Architecture**:
- Mamba SSM for long-range dependencies (O(n))
- Local windowed attention for short-range precision (O(n*w))
- Combined complexity: O(n + n*w) where w << n

**Characteristics**:
- ✅ **Best of both**: Global + local modeling
- ✅ **Efficient**: Near-linear complexity
- ✅ **Flexible**: Tune global/local balance
- ⚠️ **Experimental**: Newest architecture
- ⚠️ **Hyperparams**: More knobs to tune

**Best for**:
- Long sequences that need precise local attention
- Research into hybrid architectures
- When pure Mamba isn't expressive enough

**Key Parameters**:
- `--architecture hybrid`
- `--state-size 16` (Mamba state size)
- `--hybrid-local-window 256` (attention window size)

---

## Training Commands

### Full Training (Medium Model, 48GB VRAM)

#### Dense Transformer
```bash
python3 scripts/train.py \
  --architecture dense \
  --stage language \
  --model-size medium \
  --batch-size 2 \
  --gradient-accumulation-steps 4 \
  --use-compile \
  --use-rmsnorm \
  --use-rope \
  --use-amp \
  --use-gradient-checkpointing \
  --num-epochs 3 \
  --steps-per-epoch 1000 \
  --learning-rate 5e-5 \
  --warmup-steps 200
```

**Expected**: ~10-12 hours, 6-8GB memory, Loss ~4.0

#### Mamba
```bash
python3 scripts/train.py \
  --architecture mamba \
  --stage language \
  --model-size medium \
  --batch-size 4 \
  --gradient-accumulation-steps 2 \
  --use-compile \
  --use-rmsnorm \
  --use-amp \
  --use-gradient-checkpointing \
  --state-size 16 \
  --conv-size 4 \
  --num-epochs 3 \
  --steps-per-epoch 1000 \
  --learning-rate 5e-5 \
  --warmup-steps 200
```

**Expected**: ~8-10 hours (faster!), 5-6GB memory (less!), Loss ~4.0-4.5

#### MoE
```bash
python3 scripts/train.py \
  --architecture moe \
  --stage language \
  --model-size medium \
  --batch-size 2 \
  --gradient-accumulation-steps 4 \
  --use-compile \
  --use-rmsnorm \
  --use-rope \
  --use-amp \
  --use-gradient-checkpointing \
  --num-experts 8 \
  --expert-capacity 2 \
  --num-epochs 3 \
  --steps-per-epoch 1000 \
  --learning-rate 5e-5 \
  --warmup-steps 200
```

**Expected**: ~12-14 hours, 8-10GB memory, Loss ~3.8-4.0 (better with more params!)

#### Hybrid
```bash
python3 scripts/train.py \
  --architecture hybrid \
  --stage language \
  --model-size medium \
  --batch-size 3 \
  --gradient-accumulation-steps 3 \
  --use-compile \
  --use-rmsnorm \
  --use-amp \
  --use-gradient-checkpointing \
  --state-size 16 \
  --hybrid-local-window 256 \
  --num-epochs 3 \
  --steps-per-epoch 1000 \
  --learning-rate 5e-5 \
  --warmup-steps 200
```

**Expected**: ~9-11 hours, 6-7GB memory, Loss ~3.9-4.1

---

## Performance Comparison

### Theoretical Complexity

| Architecture | Time Complexity | Space Complexity | Active Params |
|-------------|----------------|------------------|---------------|
| **Dense** | O(n² × d) | O(n² + nd) | 100% |
| **Mamba** | O(n × d) | O(n × d) | 100% |
| **MoE** | O(n² × d / E × K) | O(n² + nd × E) | K/E (~25% for 2/8) |
| **Hybrid** | O(n × d + n × w × d) | O(n × w + nd) | 100% |

Where: n = sequence length, d = model dimension, E = num experts, K = expert capacity, w = local window

### Expected Results (Medium Model, 3 Epochs)

| Architecture | Training Time | Memory Usage | Final Loss | Tokens/sec |
|-------------|--------------|--------------|------------|------------|
| **Dense** | 10-12h | 6-8 GB | ~4.0 | 450-550 |
| **Mamba** | 8-10h ⚡ | 5-6 GB 💾 | ~4.0-4.5 | 550-650 ⚡ |
| **MoE** | 12-14h | 8-10 GB | ~3.8-4.0 ⭐ | 400-500 |
| **Hybrid** | 9-11h | 6-7 GB | ~3.9-4.1 | 500-600 |

**Legend**: ⚡ Fastest, 💾 Most memory-efficient, ⭐ Best quality

---

## When to Use Each Architecture

### Dense Transformer ✅

**Choose when**:
- This is your first LLM project (safest choice)
- Sequence length < 2048 tokens
- You want maximum stability and reproducibility
- Baseline for comparison

**Avoid when**:
- Very long sequences (>4K tokens)
- Severely memory-constrained
- Need maximum efficiency

---

### Mamba (SSM) ⚡

**Choose when**:
- Long sequences (>4K tokens)
- Memory is limited
- Speed is critical
- Willing to experiment with newer architecture

**Avoid when**:
- Need proven production stability
- Sequence length already short (<1K tokens)
- Fine-grained token interactions are critical

---

### MoE (Sparse) 🎯

**Choose when**:
- Want to scale to many parameters efficiently
- Have diverse tasks/data (experts can specialize)
- Can afford higher total memory for params
- Quality is more important than speed

**Avoid when**:
- Small datasets (experts won't specialize)
- Memory-constrained (MoE has more total params)
- Training complexity is a concern

---

### Hybrid (Mamba + Attention) 🔬

**Choose when**:
- Long sequences but need precise local attention
- Want to experiment with cutting-edge architectures
- Mamba alone isn't expressive enough
- Research-oriented project

**Avoid when**:
- Need production-ready stability
- Don't want to tune additional hyperparameters
- Pure Mamba or Dense already meets your needs

---

## Benchmarking Protocol

To fairly compare architectures, use this protocol:

### 1. Same Task, Same Data

```bash
# All architectures use same:
--stage language
--num-epochs 3
--steps-per-epoch 1000
```

### 2. Same Model Size (Parameters)

```bash
# All use medium config (371M params baseline)
--model-size medium

# Adjust for MoE: total params higher, but active params similar
# MoE: 8 experts × medium ≈ 3B total, ~450M active (2/8)
```

### 3. Same Effective Batch Size

```bash
# Dense: batch 2, accum 4 = effective 8
--batch-size 2 --gradient-accumulation-steps 4

# Mamba: batch 4, accum 2 = effective 8
--batch-size 4 --gradient-accumulation-steps 2

# Adjust based on memory usage
```

### 4. Same Optimizations

```bash
# All architectures use:
--use-compile \
--use-rmsnorm \
--use-amp \
--use-gradient-checkpointing
```

### 5. Measure Consistently

**Track for each architecture**:
- ✅ Training time (total hours)
- ✅ Peak memory usage (GB)
- ✅ Final validation loss
- ✅ Tokens/second throughput
- ✅ Training stability (NaN count)

### 6. Comparison Template

```markdown
## Benchmark Results

**Hardware**: M1 Max 48GB
**Task**: Language pretraining (TinyStories)
**Date**: 2024-XX-XX

| Architecture | Time | Memory | Final Loss | Tokens/sec | Notes |
|-------------|------|--------|------------|------------|-------|
| Dense | 10.5h | 7.2 GB | 4.02 | 520 | Stable baseline |
| Mamba | 8.8h | 5.5 GB | 4.18 | 610 | 20% faster, 24% less memory |
| MoE | 13.2h | 9.1 GB | 3.87 | 460 | Best loss, slower |
| Hybrid | 9.7h | 6.3 GB | 4.05 | 570 | Good balance |

**Winner**: Mamba for efficiency, MoE for quality, Dense for stability
```

---

## Advanced Comparisons

### Scaling to Large Model

Test how architectures scale with model size:

```bash
# Tiny (137M)
for arch in dense mamba moe hybrid; do
  python3 scripts/train.py --architecture $arch --model-size tiny ...
done

# Medium (371M)
for arch in dense mamba moe hybrid; do
  python3 scripts/train.py --architecture $arch --model-size medium ...
done

# Large (1.1B)
for arch in dense mamba moe hybrid; do
  python3 scripts/train.py --architecture $arch --model-size large ...
done
```

### Varying Sequence Length

Test how architectures handle different sequence lengths:

```bash
# Short sequences (512) - modify prepare_datasets.py
# Dense should excel

# Medium sequences (1024) - current
# All should be competitive

# Long sequences (4096) - modify prepare_datasets.py
# Mamba and Hybrid should excel
```

### Architecture-Specific Hyperparameter Tuning

**Mamba**:
- Try `--state-size [8, 16, 32]`
- Try `--conv-size [2, 4, 8]`

**MoE**:
- Try `--num-experts [4, 8, 16]`
- Try `--expert-capacity [1, 2, 4]`

**Hybrid**:
- Try `--hybrid-local-window [128, 256, 512]`
- Try `--state-size [8, 16, 32]`

---

## Troubleshooting

### Architecture-Specific Issues

**Mamba**:
- If loss is higher: increase `--state-size`
- If too slow: reduce `--conv-size`
- No position embeddings needed (implicit)

**MoE**:
- If experts not used evenly: check load balancing loss
- If unstable: reduce learning rate or increase warmup
- Higher memory: reduce `--num-experts`

**Hybrid**:
- If too slow: reduce `--hybrid-local-window`
- If quality lower: increase window or state size
- Balance SSM vs attention components

---

## Next Steps

1. **Quick test** all architectures with tiny model (1 hour each)
2. **Choose winner** based on your priorities (speed/memory/quality)
3. **Full training** with chosen architecture
4. **Fine-tune** architecture-specific hyperparameters
5. **Stage 2** code training with best architecture

---

## References

**Dense Transformer**:
- "Attention Is All You Need" (Vaswani et al., 2017)

**Mamba**:
- "Mamba: Linear-Time Sequence Modeling" (Gu & Dao, 2023)

**MoE**:
- "Switch Transformers" (Fedus et al., 2021)
- "Mixtral 8x7B" (Mistral AI, 2023)

**Hybrid**:
- Experimental combination

---

**Ready to compare architectures!** 🚀

Start with quick tests on tiny models, then scale up your winner.
