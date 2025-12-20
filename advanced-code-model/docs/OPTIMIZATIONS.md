# LLM Training Optimizations: Current & Future

This document catalogs all optimizations in our training pipeline, from basic to cutting-edge.

## Table of Contents
1. [Current Optimizations](#current-optimizations)
2. [Memory Optimizations](#memory-optimizations)
3. [Speed Optimizations](#speed-optimizations)
4. [Architecture Optimizations](#architecture-optimizations)
5. [Emerging Research](#emerging-research)

---

## Current Optimizations

### ✅ Implemented

#### 1. **Learning Rate Warmup**
- **Location**: `scripts/train.py:340-347`
- **What**: Gradually increase LR from 0 to target over N steps
- **Why**: Prevents instability in early training
- **Impact**: Stable training start, prevents early divergence
- **Code**:
```python
def lr_lambda(current_step):
    if current_step < warmup_steps:
        return float(current_step) / float(max(1, warmup_steps))
    # Cosine decay after warmup
    progress = (current_step - warmup_steps) / (total_steps - warmup_steps)
    return max(0.1, 0.5 * (1.0 + cos(progress * pi)))
```

#### 2. **Cosine Learning Rate Decay**
- **Location**: `scripts/train.py:344-347`
- **What**: Smooth LR reduction following cosine curve
- **Why**: Better convergence than step decay
- **Impact**: ~5% better final loss
- **Formula**: `lr = lr_max * 0.5 * (1 + cos(π * t/T))`

#### 3. **Gradient Clipping**
- **Location**: `scripts/train.py:221`
- **What**: Clip gradient norms to max value (1.0)
- **Why**: Prevents exploding gradients
- **Impact**: Critical for training stability
- **Code**:
```python
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

#### 4. **NaN Detection & Skip**
- **Location**: `scripts/train.py:206-208`
- **What**: Check for NaN loss, skip update if found
- **Why**: Prevents training crash from numerical issues
- **Impact**: Robust training even with unstable batches

#### 5. **Loss Clamping**
- **Location**: `scripts/train.py:140`
- **What**: Clamp loss to [0.0, 20.0] range
- **Why**: Prevents extreme loss values
- **Impact**: More stable training

#### 6. **AdamW Optimizer**
- **Location**: `scripts/train.py:330-335`
- **What**: Adam with decoupled weight decay
- **Why**: Better generalization than Adam
- **Impact**: ~2-3% better validation loss
- **Params**: β₁=0.9, β₂=0.999, weight_decay=0.01

#### 7. **Weight Tying**
- **Location**: `src/model/transformer.py:204`
- **What**: Share weights between input/output embeddings
- **Why**: Reduces parameters, improves quality
- **Impact**: Saves vocab_size × d_model parameters

#### 8. **Pre-Layer Normalization**
- **Location**: `src/model/transformer.py:173,176`
- **What**: LayerNorm before attention/FFN (not after)
- **Why**: More stable gradients, better training
- **Impact**: Essential for deep models (24+ layers)

#### 9. **Dropout Regularization**
- **Location**: `src/model/config.py:28-30`
- **What**: Random neuron dropout during training
- **Why**: Prevents overfitting
- **Impact**: Better generalization
- **Rates**: attention=0.1, residual=0.1, embedding=0.1

#### 10. **Efficient Data Loading**
- **Location**: `scripts/train.py:45-76`
- **What**: Keep data on CPU, move batches to GPU
- **Why**: Saves GPU memory
- **Impact**: Allows larger models/batches

#### 11. **MPS Cache Clearing**
- **Location**: `scripts/train.py:232-233`
- **What**: Clear MPS cache after each step
- **Why**: Prevent memory accumulation
- **Impact**: Stable memory usage on Apple Silicon

---

## Memory Optimizations

### ✅ Implemented

#### 1. **Gradient Accumulation** ⭐⭐ ✅ IMPLEMENTED
- **Status**: Live in `scripts/train.py`
- **Usage**: `--gradient-accumulation-steps 4`
- **Impact**: Simulate 4x larger batch size with same memory
- **Trade-off**: No speed loss, just more steps per update
- **Type**: **Training-only** (safe for Stage 1 & 2)
- **Implementation Difficulty**: Easy
- **Best For**: Effective batch size > GPU can fit
- **When to Use**: Want batch_size=16 but can only fit batch_size=4
- **Expected Result**: Better gradients, smoother training

#### 3. **Mixed Precision (AMP)** ⭐⭐ ✅ IMPLEMENTED
- **Status**: Live in `scripts/train.py`
- **Usage**: `--use-amp`
- **Impact**: 30-40% memory reduction, 2x faster
- **Trade-off**: Slight numerical precision loss (usually negligible)
- **Type**: **Training-only** (safe for Stage 1 & 2)
- **Implementation Difficulty**: Easy (PyTorch built-in)
- **Best For**: Modern GPUs with tensor cores
- **When to Use**: Any modern training (standard practice)
- **Expected Result**: 2x faster training, 30% less memory
- **Note**: MPS support for AMP is limited, works better on CUDA

#### 4. **Gradient Checkpointing** ⭐⭐⭐ ✅ IMPLEMENTED
- **Status**: Live in `scripts/train.py`
- **Usage**: `--use-gradient-checkpointing`
- **Impact**: 40-50% memory reduction
- **Trade-off**: 20% slower training (recompute activations)
- **Type**: **Training-only** (safe for Stage 1 & 2)
- **Implementation Difficulty**: Easy (PyTorch built-in)
- **Best For**: Fitting larger models or batch sizes
- **When to Use**: When hitting memory limits
- **Expected Result**: Medium model → 4-5GB instead of 6-8GB

---

## Speed Optimizations

### ✅ Implemented

#### 1. **Compiled Model (torch.compile)** ⭐ ✅ IMPLEMENTED
- **Status**: Live in `scripts/train.py`
- **Usage**: `--use-compile`
- **Impact**: 20-30% speedup
- **Trade-off**: Longer startup time
- **Type**: **Training-only** (safe for Stage 1 & 2)
- **Implementation Difficulty**: Easy (PyTorch 2.0+)
- **When to Use**: PyTorch 2.0+, long training runs
- **Expected Result**: 20-30% faster training after compilation warmup

### 🚧 Not Yet Implemented

#### 1. **Flash Attention** ⭐⭐⭐
- **Potential Impact**: 2-4x faster attention, 5-20x less memory
- **Trade-off**: None (pure improvement!)
- **Implementation Difficulty**: Medium (requires flash-attn package)
- **Best For**: Long sequences (>512 tokens)
- **Description**: Fused attention kernel with I/O optimization
- **Code Preview**:
```python
from flash_attn import flash_attn_func

# Instead of manual attention computation
attn_output = flash_attn_func(q, k, v, causal=True)
```
- **When to Use**: Sequences >512 tokens, available on CUDA
- **Expected Result**: 2-3x faster training for our 1024-token sequences
- **Limitation**: Not available on MPS (CUDA only)

#### 2. **Fused Kernels**
- **Examples**: Fused LayerNorm, fused activation functions
- **Impact**: 10-20% speedup
- **Difficulty**: Medium
- **When to Use**: Production training on CUDA

---

## Architecture Optimizations

### ⚠️ CRITICAL: Architecture vs Training Optimizations

**Architecture-Changing** (Stage 1 ONLY - changes model structure):
- ❌ **RMSNorm** - Incompatible with LayerNorm checkpoints
- ❌ **RoPE** - Incompatible with learned position checkpoints

**Training-Only** (Safe for Stage 1 & 2 - no model changes):
- ✅ **torch.compile**, **AMP**, **Gradient Checkpointing**, **Gradient Accumulation**

**Rule**: For fine-tuning (Stage 2), ONLY use training-only optimizations to match checkpoint architecture.

### ✅ Implemented

#### 1. **Rotary Position Embeddings (RoPE)** ⭐⭐⭐ ✅ IMPLEMENTED
- **Status**: Live in `src/model/transformer.py`
- **Usage**: `--use-rope` (Stage 1 ONLY)
- **Impact**: Better extrapolation to longer sequences
- **Trade-off**: None
- **Type**: **Architecture-changing** (Stage 1 ONLY)
- **Implementation Difficulty**: Medium
- **Used In**: LLaMA, GPT-NeoX, PaLM
- **Why Better**: Relative positions, better length generalization
- **When to Use**: Any new model training from scratch
- **Expected Result**: Better performance on varying sequence lengths
- **⚠️ Warning**: Do NOT use when loading checkpoints trained without RoPE

#### 2. **RMSNorm** ⭐ ✅ IMPLEMENTED
- **Status**: Live in `src/model/transformer.py`
- **Usage**: `--use-rmsnorm` (Stage 1 ONLY)
- **Impact**: 10-15% faster normalization
- **Trade-off**: Slightly different behavior than LayerNorm
- **Type**: **Architecture-changing** (Stage 1 ONLY)
- **Implementation Difficulty**: Easy
- **Used In**: LLaMA, T5
- **When to Use**: Speed-critical training from scratch
- **Expected Result**: 5-10% faster training
- **⚠️ Warning**: Do NOT use when loading checkpoints trained with LayerNorm

### 🚧 Not Yet Implemented

#### 2. **Grouped Query Attention (GQA)** ⭐⭐
- **Potential Impact**: 30-40% less memory for KV cache
- **Trade-off**: Slightly lower quality than full MQA
- **Implementation Difficulty**: Medium
- **Used In**: LLaMA 2, Mistral
- **Description**: Share K/V across multiple Q heads
- **When to Use**: Large models, inference optimization
- **Expected Result**: Faster generation, less memory

#### 3. **SwiGLU Activation** ⭐⭐
- **Potential Impact**: 2-3% better quality
- **Trade-off**: 33% more parameters in FFN
- **Implementation Difficulty**: Easy
- **Used In**: LLaMA, PaLM
- **Code Preview**:
```python
class SwiGLU(nn.Module):
    def forward(self, x):
        x, gate = x.chunk(2, dim=-1)
        return F.silu(gate) * x
```
- **When to Use**: Any new model (modern standard)
- **Expected Result**: Better quality for same compute

#### 3. **Multi-Query Attention (MQA)**
- **Impact**: Faster inference, less KV cache
- **Used In**: PaLM, Falcon
- **Trade-off**: Lower quality than standard attention

---

## Emerging Research

### 🔬 Alternative Architectures

#### 1. **Mamba (State Space Models)** 🔥
- **Status**: Cutting-edge (2023-2024)
- **What**: Linear-time alternative to attention
- **Complexity**: O(n) instead of O(n²)
- **Advantages**:
  - Constant memory regardless of sequence length
  - Linear scaling with sequence length
  - Can handle 1M+ token contexts
- **Disadvantages**:
  - Different training dynamics
  - Less mature ecosystem
  - Different sweet spots than transformers
- **Papers**:
  - "Mamba: Linear-Time Sequence Modeling" (Gu & Dao, 2023)
  - "Hungry Hungry Hippos" (Fu et al., 2023)
- **Directory**: `experiments/emerging-research/mamba-ssm/`

#### 2. **Mixture of Experts (MoE)** 🔥
- **Status**: Production-ready (GPT-4, Mixtral)
- **What**: Sparse activation - only use subset of parameters per token
- **Advantages**:
  - 10x parameters with 2x compute
  - Specialist sub-models
  - Better parameter efficiency
- **Disadvantages**:
  - Complex training (load balancing)
  - Harder to deploy
  - More memory during training
- **Papers**:
  - "Switch Transformers" (Fedus et al., 2021)
  - "Mixtral 8x7B" (Mistral AI, 2023)
- **Directory**: `experiments/emerging-research/mixture-of-experts/`

#### 3. **Retrieval Augmented Generation (RAG)**
- **What**: Combine LLM with external knowledge retrieval
- **Advantages**:
  - Up-to-date information
  - Factual grounding
  - Smaller models possible
- **Use Cases**: Question answering, code completion
- **Directory**: `experiments/emerging-research/retrieval-augmented/`

#### 4. **Test-Time Compute Scaling**
- **What**: Use more compute at inference for better results
- **Examples**:
  - Chain-of-thought prompting
  - Self-consistency
  - Tree-of-thought
  - Iterative refinement
- **Papers**: "Let's think step by step", "Tree of Thoughts"
- **Directory**: `experiments/emerging-research/test-time-compute/`

#### 5. **Long Context Methods**
- **Challenge**: Transformers scale O(n²) with sequence length
- **Solutions**:
  - Sliding window attention (Mistral)
  - Sparse attention patterns
  - State space models (Mamba)
  - Hierarchical approaches
- **Directory**: `experiments/emerging-research/long-context/`

#### 6. **Efficient Architectures**
- **RetNet**: Retention-based architecture
- **RWKV**: RNN with transformer performance
- **Hyena**: Subquadratic attention alternative
- **Directory**: `experiments/emerging-research/efficient-architectures/`

---

## Implementation Priority

### Phase 1: Quick Wins ✅ COMPLETE
1. ✅ **Gradient Accumulation** - IMPLEMENTED (`--gradient-accumulation-steps`)
2. ✅ **Compiled Model** - IMPLEMENTED (`--use-compile`)
3. ✅ **RMSNorm** - IMPLEMENTED (`--use-rmsnorm`, Stage 1 only)
4. ✅ **Mixed Precision (AMP)** - IMPLEMENTED (`--use-amp`)
5. ✅ **Gradient Checkpointing** - IMPLEMENTED (`--use-gradient-checkpointing`)
6. ✅ **RoPE** - IMPLEMENTED (`--use-rope`, Stage 1 only)

### Phase 2: Architecture Improvements
1. 🔄 **SwiGLU** - Better activation
2. 🔄 **GQA** - Efficient attention

### Phase 3: Advanced (Platform-Dependent)
1. 🔄 **Flash Attention** - Much faster (CUDA only, not available on MPS)

### Phase 4: Research Exploration
1. 🔬 **Mamba** - Alternative architecture
2. 🔬 **MoE** - Sparse models
3. 🔬 **RAG** - External knowledge

---

## Benchmarking Plan

For each optimization, we'll measure:
1. **Memory usage** (peak GB)
2. **Training speed** (tokens/sec)
3. **Final loss** (validation)
4. **Implementation complexity** (hours to implement)
5. **Maintenance burden** (ongoing cost)

### Benchmark Template

```markdown
## Optimization: [Name]

### Implementation
- Time: X hours
- Lines of code: Y
- Dependencies: [list]

### Results
| Metric | Baseline | Optimized | Improvement |
|--------|----------|-----------|-------------|
| Memory | XGB | YGB | Z% |
| Speed | X tok/s | Y tok/s | Z% |
| Loss | X.XX | Y.YY | Z% |

### Recommendation
- Use when: [conditions]
- Skip when: [conditions]
- Trade-offs: [list]
```

---

## Educational Value

Each optimization teaches:
1. **Gradient Checkpointing**: Memory-compute trade-offs
2. **Flash Attention**: Hardware-aware algorithms
3. **RoPE**: Position encoding theory
4. **MoE**: Sparse computation
5. **Mamba**: Alternative to attention
6. **Mixed Precision**: Numerical precision
7. **Gradient Accumulation**: Effective batch size

---

## Resources

### Papers
- "Attention Is All You Need" (Vaswani et al., 2017)
- "GPT-3" (Brown et al., 2020)
- "LLaMA" (Touvron et al., 2023)
- "Mamba" (Gu & Dao, 2023)
- "Flash Attention" (Dao et al., 2022)

### Code References
- Hugging Face Transformers
- PyTorch examples
- nanoGPT (Karpathy)
- LLaMA implementation

---

## Next Steps

1. ✅ Start Medium model training
2. 🔄 While training, implement quick wins:
   - Gradient accumulation
   - torch.compile
   - RMSNorm
3. 🔄 After training, benchmark improvements
4. 🔄 Explore emerging research (Mamba, MoE)
5. 🔄 Document findings for educational purposes

---

**Status**: Living document - updated as we implement each optimization

**Last Updated**: 2024
