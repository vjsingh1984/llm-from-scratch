# LLM From Scratch - Learning Journey

Build and understand Large Language Models from the ground up, optimized for Apple Silicon (M1 Max).

## Project Goals

1. **Learn by Building**: Implement every component of an LLM from scratch
2. **Understand Fundamentals**: Tokenization, attention mechanisms, transformer architecture
3. **Practical Training**: Train models on M1 Max with manageable parameter counts
4. **Progressive Complexity**: Start simple (dense transformer) → advance to MoE
5. **Document Everything**: Step-by-step guides for each component

## Technology Stack

- **Framework**: MLX (Apple's ML framework optimized for M1/M2/M3)
- **Language**: Python 3.9+
- **Target Hardware**: M1 Max (32-64GB unified memory)
- **Model Sizes**: 50M → 124M → 500M → 1-3B parameters

## Project Structure

```
llm-from-scratch/
├── docs/               # Step-by-step documentation
│   ├── 00-OVERVIEW.md
│   ├── 01-TOKENIZATION.md
│   ├── 02-ARCHITECTURE.md
│   ├── 03-TRAINING.md
│   └── 04-MOE.md
├── data/              # Sample datasets (TinyStories, OpenWebText subset)
├── tokenizer/         # Tokenization implementation (BPE)
│   ├── bpe.py
│   └── vocab.py
├── model/             # Transformer architecture
│   ├── attention.py
│   ├── transformer.py
│   ├── embedding.py
│   └── moe.py
├── training/          # Training infrastructure
│   ├── trainer.py
│   ├── optimizer.py
│   └── data_loader.py
├── evaluation/        # Inference and evaluation
│   ├── generate.py
│   └── metrics.py
└── experiments/       # Jupyter notebooks for exploration
```

## Learning Roadmap

### Phase 1: Foundations ✅
- [x] Project setup
- [x] Understanding tokenization
- [x] Implementing BPE tokenizer
- [x] Building vocabulary from sample data

### Phase 2: Model Architecture ✅
- [x] Embeddings and positional encoding (learned, sinusoidal, RoPE)
- [x] Multi-head self-attention with causal masking
- [x] Transformer blocks with Pre-LayerNorm
- [x] Complete GPT-style model (50M-3B params)
- [x] Grouped Query Attention support

### Phase 3: Training ✅
- [x] Data preprocessing pipeline
- [x] Training loop with AdamW optimizer
- [x] Learning rate scheduling (warmup + cosine decay)
- [x] Gradient clipping and accumulation
- [x] Automatic checkpointing
- [x] MLX mixed precision support

### Phase 4: Scaling & Evaluation ✅
- [x] Text generation with sampling strategies
- [x] Generation script with interactive mode
- [x] Training scripts for all model sizes
- [x] M1 Max optimization guidelines

### Phase 5: Advanced Topics (Next)
- [ ] Mixture of Experts (MoE) architecture
- [ ] Sparse routing mechanisms
- [ ] Comparing dense vs MoE performance
- [ ] Scaling experiments

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Download data
python scripts/download_data.py --num-samples 10000

# 3. Train tokenizer
python scripts/train_tokenizer.py --vocab-size 8000

# 4. Train model
python scripts/train.py \
    --model-size tiny \
    --batch-size 16 \
    --max-steps 10000

# 5. Generate text
python scripts/generate.py \
    --checkpoint checkpoints/best.npz \
    --prompt "Once upon a time" \
    --interactive

# Or test the model architecture
python scripts/test_model.py
```

## Resources

- [MLX Documentation](https://ml-explore.github.io/mlx/)
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) (Original Transformer paper)
- [GPT-2 Paper](https://d4mucfpksywv.cloudfront.net/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)
- [MoE Paper](https://arxiv.org/abs/2101.03961)

## M1 Max Optimization Tips

- MLX uses unified memory architecture - no explicit device transfers needed
- Batch size: Start with 8-32, adjust based on model size
- Sequence length: 256-512 for learning, up to 2048 for production
- Mixed precision (float16) can double throughput
- Monitor memory with Activity Monitor while training

---

**Next Step**: Read `docs/00-OVERVIEW.md` to understand the architecture fundamentals.
