# Overview: Understanding Language Models from First Principles

## What is a Language Model?

A language model is a probability distribution over sequences of tokens (words, subwords, or characters). Given a sequence of tokens, it predicts what comes next.

**Mathematical Definition**:
```
P(w₁, w₂, ..., wₙ) = P(w₁) × P(w₂|w₁) × P(w₃|w₁,w₂) × ... × P(wₙ|w₁,...,wₙ₋₁)
```

Modern LLMs like GPT use neural networks to learn these conditional probabilities from massive amounts of text.

## Core Components

### 1. Tokenization
**Purpose**: Convert text into numerical representations that the model can process.

**Why not just use words?**
- Limited vocabulary (can't handle new/rare words)
- Large vocabulary = huge embedding matrix
- No handling of morphology (run, running, runs = separate entries)

**Solution**: Subword tokenization (BPE, WordPiece)
- Vocabulary size: typically 30K-50K tokens
- Balances efficiency and flexibility
- Example: "unhappiness" → ["un", "happiness"] or ["un", "happy", "ness"]

**You'll implement**: Byte Pair Encoding (BPE) tokenizer

---

### 2. Embeddings
**Purpose**: Convert discrete tokens into continuous vector representations.

**Two types**:
1. **Token Embeddings**: Map each token ID to a d-dimensional vector
   - Shape: `[vocab_size, d_model]`
   - Learned during training

2. **Positional Encodings**: Add position information to embeddings
   - Transformers have no inherent notion of sequence order
   - Options: Sinusoidal (fixed), Learned, RoPE (rotary), ALiBi
   - Shape: `[max_seq_len, d_model]`

**Combined**: `embedding = token_embedding + positional_encoding`

---

### 3. Transformer Architecture

The heart of modern LLMs. Key innovation: **Self-Attention Mechanism**.

#### Multi-Head Self-Attention

**Intuition**: Allow each token to "look at" all previous tokens and decide what's relevant.

**Mechanism**:
1. Each token creates three vectors: Query (Q), Key (K), Value (V)
2. Compute attention scores: how much should each token attend to others?
3. Use scores to create weighted sum of values

**Math**:
```
Attention(Q, K, V) = softmax(QK^T / √d_k) V
```

**Multi-Head**: Run multiple attention mechanisms in parallel
- Each "head" learns different relationships
- 8-16 heads is typical
- Concatenate outputs and project back

**Code Teaser** (what you'll implement):
```python
def attention(q, k, v, mask=None):
    d_k = q.shape[-1]
    scores = (q @ k.transpose(-2, -1)) / mx.sqrt(d_k)
    if mask is not None:
        scores = mx.where(mask, scores, float('-inf'))
    attn = mx.softmax(scores, axis=-1)
    return attn @ v
```

#### Transformer Block

Each transformer block contains:
1. **Multi-Head Self-Attention** (with residual connection + layer norm)
2. **Feed-Forward Network** (with residual connection + layer norm)

**Feed-Forward Network**:
- Two linear layers with activation in between
- Expands dimension by 4x typically
- `FFN(x) = ReLU(xW₁ + b₁)W₂ + b₂`

**Layer Normalization**:
- Stabilizes training
- Applied before or after each sub-layer (Pre-LN vs Post-LN)

**Residual Connections**:
- Help gradients flow during backpropagation
- `output = LayerNorm(x + Attention(x))`

#### Full GPT-Style Model

Stack multiple transformer blocks:
```
Input Text
    ↓
Tokenization → [token_ids]
    ↓
Token Embedding + Positional Encoding
    ↓
Transformer Block 1
    ↓
Transformer Block 2
    ↓
...
    ↓
Transformer Block N
    ↓
Layer Norm
    ↓
Output Projection → [vocab_size]
    ↓
Softmax → Probability Distribution
```

---

### 4. Training

**Objective**: Next Token Prediction (Causal Language Modeling)

**Loss Function**: Cross-Entropy Loss
```python
loss = -log P(actual_next_token | previous_tokens)
```

**Key Concepts**:

1. **Causal Masking**: Prevent tokens from "seeing" future tokens
   - Use triangular mask during attention

2. **Optimizer**: AdamW (Adam with weight decay)
   - Learning rate: typically 1e-4 to 6e-4
   - Weight decay: 0.1

3. **Learning Rate Schedule**: Warmup + Cosine Decay
   - Warmup: Gradually increase LR for first ~2000 steps
   - Decay: Slowly decrease to 10% of peak

4. **Gradient Accumulation**: Train with larger effective batch size
   - M1 Max memory limited? Accumulate gradients over multiple mini-batches
   - Effective batch size = mini_batch_size × accumulation_steps

5. **Mixed Precision**: Use float16 for speed, float32 for stability
   - MLX handles this automatically
   - Can 2x training speed

---

## Model Sizes for M1 Max

### Starter: 50M Parameters
- **Layers**: 6
- **Hidden size**: 512
- **Attention heads**: 8
- **FFN size**: 2048
- **Vocab size**: 32K
- **Context length**: 512
- **Memory**: ~2GB
- **Training time**: Minutes on sample data

### GPT-2 Small: 124M Parameters
- **Layers**: 12
- **Hidden size**: 768
- **Attention heads**: 12
- **FFN size**: 3072
- **Vocab size**: 50K
- **Context length**: 1024
- **Memory**: ~5GB
- **Training time**: Hours on TinyStories

### Medium: 350M Parameters
- **Layers**: 24
- **Hidden size**: 1024
- **Attention heads**: 16
- **FFN size**: 4096
- **Memory**: ~12GB

### Your M1 Max Can Handle: Up to 3B Parameters
- With 32GB RAM: comfortably train 1B
- With 64GB RAM: can push to 3B
- Requires efficient implementation and mixed precision

---

## Dense vs Mixture of Experts (MoE)

### Dense Models (What You'll Build First)
- All parameters active for every input
- Simple, well-understood
- Example: GPT-2, GPT-3

### Mixture of Experts (Advanced Topic)
- Only a subset of parameters active for each input
- **Sparse Routing**: Route each token to 1-2 "expert" FFNs
- **Benefit**: More parameters without proportional compute increase
- **Example**: 8 experts, top-2 routing → only 25% of FFN parameters active

**MoE Architecture**:
```
Input
    ↓
Router Network → [scores for each expert]
    ↓
Top-K Selection → [select best 2 experts]
    ↓
Expert 1 (FFN) ──┐
Expert 2 (FFN) ──┼→ Weighted Combination
...             │
Expert 8 (FFN) ──┘
    ↓
Output
```

**You'll explore**: Build dense model first, then add MoE to FFN layers

---

## Development Workflow

### Phase 1: Build Small, Debug Fast
1. Implement tokenizer on tiny dataset (1000 sentences)
2. Build 50M model with 6 layers
3. Train on small corpus (10MB text)
4. Verify loss decreases, model can overfit small dataset
5. Test text generation

### Phase 2: Scale Up
1. Train 124M model on larger dataset (TinyStories ~100MB)
2. Implement proper evaluation (validation loss, perplexity)
3. Experiment with hyperparameters
4. Compare results to known baselines

### Phase 3: Advanced
1. Scale to 350M-1B parameters
2. Implement MoE architecture
3. Compare dense vs MoE efficiency
4. Optimize for M1 Max

---

## Success Metrics

**Technical**:
- ✓ Loss decreases during training
- ✓ Validation loss improves
- ✓ Perplexity < 50 on TinyStories (good), < 20 (great)
- ✓ Model can generate coherent text after training

**Learning**:
- ✓ Understand how attention works mathematically
- ✓ Can explain difference between Q, K, V
- ✓ Know why layer norm and residual connections matter
- ✓ Can tune hyperparameters (LR, batch size, warmup)
- ✓ Understand trade-offs (model size vs compute vs memory)

---

## Next Steps

1. Read `01-TOKENIZATION.md` to understand and implement BPE
2. Collect sample dataset (we'll use TinyStories - stories written for children)
3. Build your first tokenizer

**Key Insight**: You're not just training a model - you're building deep intuition for how language models work. Every component teaches you something fundamental about modern AI.

---

## Recommended Reading Order

1. ✓ This document (foundations)
2. `01-TOKENIZATION.md` (how text becomes numbers)
3. `02-ARCHITECTURE.md` (building the transformer)
4. `03-TRAINING.md` (making it learn)
5. `04-MOE.md` (advanced: sparse models)

Let's start building!
