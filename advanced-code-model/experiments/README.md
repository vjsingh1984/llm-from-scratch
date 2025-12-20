# LLM Training Experiments & Optimizations

This directory contains isolated experiments exploring optimizations and emerging research in LLM training.

## Directory Structure

```
experiments/
├── optimizations/              # Performance improvements
│   ├── gradient-checkpointing/ # Memory-compute trade-off
│   ├── mixed-precision/        # FP16 training
│   ├── flash-attention/        # Efficient attention
│   ├── rope-embeddings/        # Better position encoding
│   ├── swiglu/                 # Better activation
│   ├── rmsnorm/                # Faster normalization
│   └── gradient-accumulation/  # Larger effective batch size
│
└── emerging-research/          # Cutting-edge approaches
    ├── mamba-ssm/             # Linear-time alternative to attention
    ├── mixture-of-experts/     # Sparse models
    ├── retrieval-augmented/    # External knowledge
    ├── test-time-compute/      # Inference-time optimization
    ├── long-context/           # Beyond transformer limits
    └── efficient-architectures/ # RetNet, RWKV, Hyena

```

## Quick Reference

### 🎯 By Impact (High → Low)

**Memory Savings**:
1. 🥇 Gradient Checkpointing (40-50% less memory)
2. 🥈 Mixed Precision (30-40% less memory)
3. 🥉 Flash Attention (5-20x less memory for long sequences)

**Speed Improvements**:
1. 🥇 Flash Attention (2-4x faster)
2. 🥈 torch.compile (20-30% faster)
3. 🥉 Mixed Precision (2x faster)

**Quality Improvements**:
1. 🥇 RoPE (better position encoding)
2. 🥈 SwiGLU (better activation)
3. 🥉 Gradient Accumulation (larger effective batch size)

### 📊 By Difficulty (Easy → Hard)

**Easy** (< 1 hour):
- torch.compile
- Gradient Accumulation
- RMSNorm

**Medium** (2-4 hours):
- Gradient Checkpointing
- Mixed Precision (AMP)
- RoPE
- SwiGLU

**Hard** (1-2 days):
- Flash Attention
- Grouped Query Attention
- Mixture of Experts
- Mamba

### 🎓 Educational Value

**Fundamentals**:
- Gradient Accumulation → Batch size concepts
- Mixed Precision → Numerical precision
- Gradient Checkpointing → Memory-compute trade-offs

**Advanced**:
- Flash Attention → Hardware-aware algorithms
- RoPE → Position encoding theory
- Mamba → Alternatives to attention

**Research Frontiers**:
- MoE → Sparse computation
- RAG → External knowledge integration
- Test-time Compute → Inference optimization

## Implementation Order

### Phase 1: Quick Wins (While Model Training)
**Time**: 2-3 hours
**Goal**: Immediate improvements without stopping training

1. ✅ **Document current optimizations**
   - Status: Complete
   - See: `docs/OPTIMIZATIONS.md`

2. ✅ **Add Gradient Accumulation**
   - Status: Complete
   - Time: 30 minutes
   - Impact: Better gradients, effective larger batch size
   - Usage: `--gradient-accumulation-steps 4`
   - See: `optimizations/gradient-accumulation/`

3. ✅ **Add torch.compile**
   - Status: Complete
   - Time: 5 minutes (one line!)
   - Impact: 20-30% speedup
   - Usage: `--use-compile`
   - Code: `model = torch.compile(model)`

4. ✅ **Implement RMSNorm**
   - Status: Complete
   - Time: 1 hour
   - Impact: 10-15% faster normalization
   - Usage: `--use-rmsnorm`
   - See: `optimizations/rmsnorm/`

### Phase 2: Architecture Improvements ✅ COMPLETE
**Status**: Implemented
**Goal**: Retrain with architectural improvements

5. ✅ **Add RoPE**
   - Status: Complete
   - Impact: Better position encoding
   - Usage: `--use-rope` (Stage 1 ONLY)
   - See: `optimizations/rope-embeddings/`

6. ✅ **Implement Gradient Checkpointing**
   - Status: Complete
   - Impact: Fit larger models (40-50% less memory)
   - Usage: `--use-gradient-checkpointing`
   - See: `optimizations/gradient-checkpointing/`

7. ✅ **Add Mixed Precision**
   - Status: Complete
   - Impact: Faster training (2x speedup + memory savings)
   - Usage: `--use-amp`
   - See: `optimizations/mixed-precision/`

8. 🔄 **Add SwiGLU**
   - Status: Planned
   - Impact: Better activation function
   - See: `optimizations/swiglu/`

### Phase 3: Research Exploration
**Time**: 1-2 weeks
**Goal**: Understand emerging approaches

9. 🔬 **Explore Mamba**
   - Learn: Linear-time sequence modeling
   - See: `emerging-research/mamba-ssm/`

10. 🔬 **Explore MoE**
    - Learn: Sparse computation
    - See: `emerging-research/mixture-of-experts/`

11. 🔬 **Explore RAG**
    - Learn: External knowledge integration
    - See: `emerging-research/retrieval-augmented/`

## Current Baseline

### Model: Medium (371M params)
- **Layers**: 24
- **Hidden**: 1024
- **Sequence**: 1024 tokens
- **Memory**: ~6-8GB
- **Speed**: TBD (training in progress)
- **Quality**: TBD (training in progress)

### Optimizations Active
- ✅ Learning rate warmup
- ✅ Cosine LR decay
- ✅ Gradient clipping (1.0)
- ✅ NaN detection
- ✅ Loss clamping
- ✅ AdamW optimizer
- ✅ Weight tying
- ✅ Pre-LayerNorm
- ✅ Dropout (0.1)
- ✅ CPU data loading
- ✅ MPS cache clearing

## Benchmarking Template

For each experiment:

### 1. Before Implementation
```markdown
## Baseline Metrics
- Memory: X GB
- Speed: Y tokens/sec
- Loss: Z.ZZ
- Training time: H hours
```

### 2. After Implementation
```markdown
## Results
| Metric | Before | After | Improvement |
|--------|--------|-------|-------------|
| Memory | XGB | YGB | +/-Z% |
| Speed | X tok/s | Y tok/s | +Z% |
| Loss | X.XX | Y.YY | +/-Z% |
| Time | Xh | Yh | +/-Z% |
```

### 3. Analysis
```markdown
## Findings
- What worked: [details]
- What didn't: [details]
- Surprises: [details]
- Recommendations: [when to use, when to skip]
```

## Learning Objectives

By the end of these experiments, you'll understand:

### Technical
1. **Memory-Compute Trade-offs** (Gradient Checkpointing)
2. **Hardware-Aware Algorithms** (Flash Attention)
3. **Numerical Precision** (Mixed Precision)
4. **Position Encoding** (RoPE vs Learned)
5. **Activation Functions** (GELU vs SwiGLU)
6. **Normalization** (LayerNorm vs RMSNorm)
7. **Attention Alternatives** (Mamba, SSMs)
8. **Sparse Models** (MoE)

### Conceptual
1. **No free lunch** - Every optimization has trade-offs
2. **Measure everything** - Intuition can be wrong
3. **Context matters** - What works for GPT-4 may not work for your model
4. **Emerging is risky** - Cutting-edge has rough edges
5. **Fundamentals matter** - Good data beats fancy architecture

## Resources

### Papers (Essential)
- "Attention Is All You Need" (Vaswani et al., 2017) - The foundation
- "GPT-3" (Brown et al., 2020) - Scaling laws
- "LLaMA" (Touvron et al., 2023) - Modern architecture choices
- "Flash Attention" (Dao et al., 2022) - Efficient attention
- "Mamba" (Gu & Dao, 2023) - Attention alternative

### Code References
- Hugging Face Transformers - Production implementations
- nanoGPT (Karpathy) - Educational Transformer
- PyTorch Examples - Official tutorials
- flash-attn - Efficient attention kernels

### Courses
- Stanford CS224N - NLP with Deep Learning
- Fast.ai - Practical Deep Learning
- Andrej Karpathy - Neural Networks: Zero to Hero

## Contributing

Each experiment should include:
1. **README.md** - Overview, theory, implementation plan
2. **Code** - Working implementation
3. **Results** - Benchmark data, plots
4. **Analysis** - What we learned

## Status Tracker

| Experiment | Status | Priority | Difficulty | Impact | Stage Compatibility |
|-----------|--------|----------|------------|--------|-------------------|
| Gradient Accumulation | ✅ Complete | High | Easy | Medium | Stage 1 & 2 |
| torch.compile | ✅ Complete | High | Easy | High | Stage 1 & 2 |
| Mixed Precision (AMP) | ✅ Complete | High | Medium | High | Stage 1 & 2 |
| Gradient Checkpointing | ✅ Complete | High | Medium | High | Stage 1 & 2 |
| RMSNorm | ✅ Complete | Medium | Easy | Low | **Stage 1 ONLY** |
| RoPE | ✅ Complete | Medium | Medium | Medium | **Stage 1 ONLY** |
| SwiGLU | 📝 Planned | Medium | Medium | Medium | **Stage 1 ONLY** |
| Flash Attention | 📝 Planned | Low | Hard | High* | Stage 1 & 2 |
| Mamba | 📝 Planned | Low | Hard | Educational | Stage 1 & 2 |
| MoE | 📝 Planned | Low | Hard | Educational | Stage 1 & 2 |

*High impact but CUDA-only (not available on MPS)

### Key Notes:
- **Stage 1 ONLY**: Architecture-changing optimizations (RMSNorm, RoPE, SwiGLU) can't be used when loading checkpoints
- **Stage 1 & 2**: Training-only optimizations (compile, AMP, gradient checkpointing, gradient accumulation) work for both stages

## Questions to Explore

1. **Do modern optimizations help small models?** Most research focuses on billion+ parameter models. Do these techniques scale down?

2. **Is attention really necessary?** Mamba suggests no. When does it break down?

3. **What's the optimal sequence length?** We use 1024. Is that right for our use case?

4. **How much does architecture matter vs data?** If we improve the model 10%, is that as good as 10% more data?

5. **What optimizations stack?** Can we combine gradient checkpointing + mixed precision + flash attention?

---

**Remember**: The goal isn't just to train a model, but to deeply understand how modern LLMs work and where the field is heading. Each experiment is a learning opportunity!

**Last Updated**: 2024
