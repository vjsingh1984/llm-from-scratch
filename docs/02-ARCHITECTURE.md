# Transformer Architecture: Building GPT from Scratch

## Overview

We've implemented a complete GPT-style transformer model with all the key components. This document explains each piece in detail.

## Architecture Summary

```
Input Token IDs [batch_size, seq_len]
    ↓
Token Embedding [batch_size, seq_len, d_model]
    ↓
Positional Encoding (add position information)
    ↓
Dropout
    ↓
┌─────────────────────────────────────┐
│ Transformer Block 1                 │
│  ┌─────────────────────────────┐    │
│  │ Layer Norm                  │    │
│  │ Multi-Head Self-Attention   │    │
│  │ Residual Connection         │    │
│  └─────────────────────────────┘    │
│  ┌─────────────────────────────┐    │
│  │ Layer Norm                  │    │
│  │ Feed-Forward Network        │    │
│  │ Residual Connection         │    │
│  └─────────────────────────────┘    │
└─────────────────────────────────────┘
    ↓
Transformer Block 2 (same structure)
    ↓
... (repeat for n_layers)
    ↓
Final Layer Norm
    ↓
Output Projection [batch_size, seq_len, vocab_size]
    ↓
Softmax → Token Probabilities
```

---

## 1. Embeddings (`model/embedding.py`)

### Token Embeddings

**Purpose**: Convert discrete token IDs to continuous vectors.

**Implementation**:
```python
class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size: int, d_model: int):
        self.embedding = nn.Embedding(vocab_size, d_model)

    def __call__(self, x: mx.array) -> mx.array:
        return self.embedding(x) * math.sqrt(self.d_model)
```

**Key Points**:
- Maps each token ID (0 to vocab_size) to a d_model-dimensional vector
- Scaling by √d_model helps with gradient flow (from original Transformer paper)
- Embeddings are learned during training

**Example**:
- Input: `[1, 42, 123]` (token IDs)
- Output: 3 vectors of dimension d_model (e.g., 768)

### Positional Encodings

Transformers have no inherent notion of sequence order. Positional encodings add this information.

#### Option 1: Sinusoidal (Original Transformer)

**Formula**:
```
PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
```

**Advantages**:
- Fixed (no parameters to learn)
- Can extrapolate to longer sequences
- Mathematical properties for relative positions

#### Option 2: Learned (GPT-2 style)

**Implementation**:
```python
class LearnedPositionalEmbedding(nn.Module):
    def __init__(self, max_seq_len: int, d_model: int):
        self.position_embedding = nn.Embedding(max_seq_len, d_model)

    def __call__(self, x: mx.array) -> mx.array:
        positions = mx.arange(x.shape[1])
        return x + self.position_embedding(positions)
```

**Advantages**:
- Model learns optimal position representations
- Often works better in practice
- Used by GPT-2, GPT-3

**Trade-off**: Limited to sequences ≤ max_seq_len seen during training.

#### Option 3: Rotary Position Embedding (RoPE)

Used by modern LLMs (LLaMA, PaLM). Applies rotation to Q and K based on position.

**Advantages**:
- Relative position information
- Better extrapolation to longer sequences
- No explicit position embeddings needed

---

## 2. Multi-Head Self-Attention (`model/attention.py`)

The heart of the transformer. Allows each token to attend to all previous tokens.

### The Attention Mechanism

**Intuition**:
- Query (Q): "What am I looking for?"
- Key (K): "What information do I have?"
- Value (V): "What do I actually represent?"

**Formula**:
```
Attention(Q, K, V) = softmax(Q @ K^T / √d_k) @ V
```

**Step by Step**:

1. **Compute attention scores**: How much should each token attend to others?
   ```
   scores = Q @ K^T / √d_k
   ```
   - Shape: [batch, n_heads, seq_len, seq_len]
   - Divided by √d_k for stable gradients

2. **Apply causal mask**: Prevent attending to future tokens
   ```
   scores = where(mask, scores, -inf)
   ```

3. **Softmax**: Convert scores to probabilities
   ```
   attention_weights = softmax(scores, axis=-1)
   ```

4. **Apply to values**: Weighted sum of value vectors
   ```
   output = attention_weights @ V
   ```

### Multi-Head Attention

Instead of one attention mechanism, run multiple in parallel:

```python
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int):
        self.n_heads = n_heads
        self.d_head = d_model // n_heads

        # Single projection for Q, K, V (more efficient)
        self.qkv_proj = nn.Linear(d_model, 3 * d_model)
        self.out_proj = nn.Linear(d_model, d_model)
```

**Why multiple heads?**
- Each head can learn different relationships
- Head 1 might learn syntax, Head 2 semantics, etc.
- Increases model capacity without proportional compute increase

**Example with 12 heads, d_model=768**:
- Each head has dimension 768 / 12 = 64
- 12 independent attention mechanisms run in parallel
- Outputs are concatenated and projected back to 768 dimensions

### Causal Masking

For language modeling, we must prevent tokens from "seeing" the future:

```python
def create_causal_mask(seq_len: int):
    # Lower triangular matrix
    # [[1, 0, 0],
    #  [1, 1, 0],
    #  [1, 1, 1]]
    return mx.tril(mx.ones((seq_len, seq_len)))
```

Token at position i can only attend to positions 0 through i.

### Grouped Query Attention (GQA)

Modern variant used by LLaMA 2, Mistral:

**Idea**: Share K and V across groups of query heads.

- **Multi-Head Attention**: 12 Q heads, 12 K heads, 12 V heads
- **Grouped Query Attention**: 12 Q heads, 4 K heads, 4 V heads (3:1 ratio)
- **Multi-Query Attention**: 12 Q heads, 1 K head, 1 V head

**Benefits**:
- Reduces memory usage (fewer K/V pairs to store)
- Faster inference with KV caching
- Minimal quality degradation

**Implementation** in `model/attention.py`:
```python
class GroupedQueryAttention(nn.Module):
    def __init__(self, d_model, n_heads, n_kv_heads):
        self.n_groups = n_heads // n_kv_heads
        # Repeat K, V for each group during forward pass
```

---

## 3. Feed-Forward Network

Simple 2-layer MLP applied to each position independently:

```python
class FeedForward(nn.Module):
    def __call__(self, x):
        x = self.fc1(x)           # [d_model] → [d_ff]
        x = self.activation(x)    # GELU, ReLU, or Swish
        x = self.dropout(x)
        x = self.fc2(x)           # [d_ff] → [d_model]
        x = self.dropout(x)
        return x
```

**Typical dimensions**:
- d_ff = 4 × d_model
- Example: 768 → 3072 → 768

**Why expand 4x?**
- Gives model more "space" to process information
- Each position processed independently (no cross-position interaction)
- Attention handles cross-position, FFN handles per-position complexity

**Activation Functions**:
- **GELU** (Gaussian Error Linear Unit): Used by GPT-2, BERT. Smooth, differentiable everywhere
- **ReLU**: Simple, fast, but can cause dead neurons
- **Swish/SiLU**: Similar to GELU, used by some modern models

---

## 4. Transformer Block

Combines attention and FFN with normalization and residual connections:

```python
class TransformerBlock(nn.Module):
    def __call__(self, x):
        # Pre-LayerNorm architecture (GPT-2 style)
        x = x + self.attn(self.ln1(x))  # Attention + residual
        x = x + self.ff(self.ln2(x))    # FFN + residual
        return x
```

### Layer Normalization

Normalizes activations across the feature dimension:

```python
# For each position, normalize across d_model dimension
mean = x.mean(axis=-1, keepdims=True)
std = x.std(axis=-1, keepdims=True)
x_norm = (x - mean) / (std + eps)
x_out = gamma * x_norm + beta  # Learnable scale and shift
```

**Why Layer Norm?**
- Stabilizes training (prevents gradients from exploding/vanishing)
- Allows higher learning rates
- Works better than Batch Norm for sequences

### Residual Connections

Skip connections that add the input to the output:

```
output = input + Transform(input)
```

**Why residuals?**
- Allows gradients to flow directly through the network
- Enables training very deep models (100+ layers)
- Helps with optimization

### Pre-LN vs Post-LN

**Post-LN (Original Transformer)**:
```python
x = LayerNorm(x + Attention(x))
x = LayerNorm(x + FFN(x))
```

**Pre-LN (GPT-2, modern transformers)**:
```python
x = x + Attention(LayerNorm(x))
x = x + FFN(LayerNorm(x))
```

**Pre-LN advantages**:
- More stable training
- Can train deeper models without warmup
- Used by most modern LLMs

---

## 5. Complete GPT Model (`model/transformer.py`)

### Model Configuration

We use a dataclass for easy configuration:

```python
@dataclass
class GPTConfig:
    vocab_size: int = 50257
    n_layers: int = 12
    d_model: int = 768
    n_heads: int = 12
    d_ff: int = 3072
    max_seq_len: int = 1024
    dropout: float = 0.1
    # ... more options
```

**Pre-configured sizes**:
- **Tiny** (50M params): 6 layers, 512 hidden, 8 heads - for quick experiments
- **GPT-2 Small** (124M): 12 layers, 768 hidden, 12 heads
- **GPT-2 Medium** (350M): 24 layers, 1024 hidden, 16 heads
- **GPT-2 Large** (774M): 36 layers, 1280 hidden, 20 heads

### Full Model Architecture

```python
class GPTModel(nn.Module):
    def __init__(self, config):
        # Embeddings
        self.token_embedding = TokenEmbedding(...)
        self.pos_embedding = LearnedPositionalEmbedding(...)

        # Transformer blocks
        self.blocks = [TransformerBlock(config) for _ in range(n_layers)]

        # Output
        self.ln_f = LayerNorm(d_model)  # Final layer norm
        self.lm_head = Linear(d_model, vocab_size)  # Project to vocabulary

    def __call__(self, input_ids, targets=None):
        x = self.token_embedding(input_ids)
        x = self.pos_embedding(x)
        x = self.dropout(x)

        for block in self.blocks:
            x = block(x)

        x = self.ln_f(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            loss = cross_entropy(logits, targets)

        return logits, loss
```

### Parameter Count

**Formula for approximate parameter count**:

```
Embedding params = vocab_size × d_model + max_seq_len × d_model
Attention params = 4 × d_model² (Q, K, V, output projections)
FFN params = 2 × d_model × d_ff
Layer params = Attention + FFN ≈ 4d² + 8d²  = 12d² (for d_ff = 4d)
Total ≈ vocab_size × d + n_layers × 12d²
```

**Example (GPT-2 Small)**:
- Embeddings: 50,257 × 768 + 1024 × 768 ≈ 39M
- 12 layers × 12 × 768² ≈ 85M
- Total: ~124M parameters

---

## 6. Text Generation

Autoregressive generation: predict one token at a time.

```python
def generate(self, input_ids, max_new_tokens, temperature=1.0):
    for _ in range(max_new_tokens):
        # Get predictions
        logits, _ = self(input_ids)

        # Get logits for last token
        next_token_logits = logits[:, -1, :] / temperature

        # Sample next token
        probs = softmax(next_token_logits)
        next_token = sample(probs)

        # Append to sequence
        input_ids = cat([input_ids, next_token], dim=1)

    return input_ids
```

**Sampling Strategies**:

1. **Greedy**: Always pick highest probability token
   - Deterministic but repetitive

2. **Temperature Sampling**: Scale logits before softmax
   - temperature < 1: More conservative (peaked distribution)
   - temperature > 1: More random (flatter distribution)

3. **Top-K**: Sample from top K tokens
   - Prevents sampling very unlikely tokens

4. **Top-P (Nucleus)**: Sample from smallest set with cumulative prob ≥ P
   - Adaptive based on confidence

---

## Parameter Counts for M1 Max

### What Your M1 Max Can Handle

**Memory formula (inference)**:
```
Memory ≈ 2 bytes/param × num_params (float16)
       ≈ 4 bytes/param × num_params (float32)
```

**Training requires ~4x more memory** (gradients, optimizer states).

| Model Size | Parameters | FP16 Memory | FP32 Memory | Training (FP32) | Fits M1 Max? |
|------------|------------|-------------|-------------|-----------------|--------------|
| Tiny | 50M | 100 MB | 200 MB | 800 MB | ✓ (32GB) |
| GPT-2 Small | 124M | 250 MB | 500 MB | 2 GB | ✓ (32GB) |
| GPT-2 Medium | 350M | 700 MB | 1.4 GB | 5.6 GB | ✓ (32GB) |
| GPT-2 Large | 774M | 1.5 GB | 3 GB | 12 GB | ✓ (32GB) |
| 1B params | 1B | 2 GB | 4 GB | 16 GB | ✓ (32GB) |
| 3B params | 3B | 6 GB | 12 GB | 48 GB | ✓ (64GB) |

**Recommendations**:
- **Start with Tiny (50M)**: Learn quickly, iterate fast
- **Scale to Small (124M)**: Good quality, still fast on M1 Max
- **Medium (350M)**: Excellent quality-to-compute ratio
- **1-3B**: Requires careful memory management, gradient accumulation

---

## Testing Your Model

Run the test script to verify everything works:

```bash
python scripts/test_model.py
```

This tests:
- Forward pass
- Loss computation
- Text generation
- Different model sizes
- Attention variants

---

## Key Takeaways

1. **Embeddings**: Convert discrete tokens to continuous vectors + position info
2. **Attention**: Learn relationships between tokens (Q, K, V mechanism)
3. **Multi-Head**: Run multiple attention operations in parallel
4. **FFN**: Per-position processing with expansion
5. **Normalization + Residuals**: Stable training of deep networks
6. **Stacking**: Repeat transformer blocks for deeper understanding

---

## Next Steps

Now that you have a working model:

1. **Test the model**: Run `python scripts/test_model.py`
2. **Read `03-TRAINING.md`**: Learn how to train the model
3. **Experiment**: Try different configurations, visualize attention weights
4. **Train a small model**: Start with Tiny on TinyStories dataset

The architecture is complete and ready for training!

---

## Further Reading

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) - Original Transformer
- [Language Models are Unsupervised Multitask Learners](https://d4mucfpksywv.cloudfront.net/better-language-models/language_models_are_unsupervised_multitask_learners.pdf) - GPT-2
- [GQA: Training Generalized Multi-Query Transformer Models](https://arxiv.org/abs/2305.13245)
- [RoFormer: Enhanced Transformer with Rotary Position Embedding](https://arxiv.org/abs/2104.09864)
