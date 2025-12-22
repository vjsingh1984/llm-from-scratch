# Model Architectures

> Understanding the 4 model architectures available in this project.

---

![Architecture Comparison](diagrams/architecture-comparison.svg)

---

## Overview

This project supports **4 different model architectures**:

| Architecture | Complexity | Best For |
|--------------|------------|----------|
| Dense Transformer | O(n²) | Learning, standard tasks |
| Mamba (SSM) | O(n) | Long context, fast inference |
| MoE (Sparse) | O(n²) | Large scale, multi-domain |
| Hybrid | O(n) - O(n²) | Complex reasoning + long context |

---

## 1. Dense Transformer

The classic Transformer architecture used in GPT, Claude, and most LLMs.

### Architecture

```
Input → Embedding → [Attention + FFN] × N → Output
```

### How It Works

Every layer has:
1. **Self-Attention**: All tokens attend to all other tokens
2. **FFN**: Process each position independently

```python
class TransformerBlock:
    def forward(self, x):
        x = x + self.attention(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x
```

### Pros & Cons

| Pros | Cons |
|------|------|
| Simple, well-understood | O(n²) memory for attention |
| Excellent results | Slow for long sequences |
| Many optimizations available | All parameters active |
| Easy to train | - |

### When to Use

- Learning and experimentation
- Short to medium context (< 4K tokens)
- When you want proven, reliable results

### Configuration

```python
config = get_config('large')
model = create_model(config, device='mps')  # Dense by default
```

---

## 2. Mamba (State Space Model)

A linear-complexity alternative to attention, based on state space models.

### Architecture

```
Input → Embedding → [Mamba Block] × N → Output
```

### How It Works

Instead of attention, Mamba uses:
1. **Selective State Spaces**: Learns what to remember/forget
2. **Linear Recurrence**: O(n) complexity
3. **Hardware-aware Design**: Optimized for GPUs

```python
class MambaBlock:
    def forward(self, x):
        # Project to higher dimension
        xz = self.in_proj(x)
        x, z = xz.chunk(2, dim=-1)

        # Apply SSM
        x = self.ssm(self.conv1d(x))

        # Gate and project back
        return self.out_proj(x * F.silu(z))
```

### Pros & Cons

| Pros | Cons |
|------|------|
| O(n) complexity | Newer, less proven |
| Fast inference | Different optimization dynamics |
| Great for long context | May underperform on some tasks |
| Constant memory during generation | - |

### When to Use

- Very long documents (32K+ tokens)
- Streaming applications
- Low-latency inference requirements

### Configuration

```python
from src.model.mamba import create_mamba_model
from src.model.config import get_mamba_config

config = get_mamba_config('large')
model = create_mamba_model(config, device='mps')
```

---

## 3. MoE (Mixture of Experts)

Sparse architecture that activates only a subset of parameters per token.

### Architecture

```
Input → Embedding → [Attention + Router → Experts] × N → Output
```

### How It Works

1. **Router**: Decides which experts to use for each token
2. **Experts**: Multiple FFN blocks, only top-K activated
3. **Combine**: Weighted sum of expert outputs

```python
class MoELayer:
    def forward(self, x):
        # Route tokens to experts
        router_logits = self.router(x)
        weights, indices = topk(router_logits, k=2)

        # Only compute selected experts
        output = sum(
            weight * self.experts[idx](x)
            for weight, idx in zip(weights, indices)
        )
        return output
```

### Pros & Cons

| Pros | Cons |
|------|------|
| More params, same compute | Load balancing is tricky |
| Specialized experts | More complex training |
| Better scaling efficiency | Higher memory footprint |
| Multi-domain capability | - |

### When to Use

- Very large models (10B+ params)
- Multi-domain tasks (code + text + math)
- When you want more capacity without more compute

### Configuration

```python
from src.model.moe import create_moe_model
from src.model.config import get_moe_config

config = get_moe_config('large')
config.num_experts = 8
config.top_k = 2
model = create_moe_model(config, device='mps')
```

---

## 4. Hybrid (Attention + Mamba)

Combines attention layers with Mamba layers for the best of both worlds.

### Architecture

```
Input → Embedding → [Attention, Mamba, Attention, Mamba, ...] × N/2 → Output
```

### How It Works

Alternates between:
1. **Attention layers**: Capture precise token-to-token relationships
2. **Mamba layers**: Efficient sequence modeling

```python
class HybridBlock:
    def __init__(self, layer_idx):
        if layer_idx % 2 == 0:
            self.block = AttentionBlock()
        else:
            self.block = MambaBlock()

    def forward(self, x):
        return self.block(x)
```

### Pros & Cons

| Pros | Cons |
|------|------|
| Best of both architectures | More complex |
| Strong reasoning (attention) | May be overkill for simple tasks |
| Long context (Mamba) | Newer approach |
| Flexible design | - |

### When to Use

- Complex reasoning over long documents
- Code generation with large context
- When you need both precision and efficiency

### Configuration

```python
from src.model.hybrid import create_hybrid_model
from src.model.config import get_hybrid_config

config = get_hybrid_config('large')
config.attention_layers = [0, 2, 4, 6]  # Attention at these layers
config.mamba_layers = [1, 3, 5, 7]      # Mamba at these layers
model = create_hybrid_model(config, device='mps')
```

---

## Performance Comparison

### Inference Speed

| Architecture | Tokens/sec (1K context) | Tokens/sec (8K context) |
|--------------|-------------------------|-------------------------|
| Dense | 100 | 25 |
| Mamba | 150 | 145 |
| MoE | 80 | 20 |
| Hybrid | 120 | 80 |

*Approximate, depends on hardware*

### Memory Usage

| Architecture | 1K context | 8K context | 32K context |
|--------------|------------|------------|-------------|
| Dense | 4GB | 8GB | 32GB |
| Mamba | 4GB | 4.5GB | 5GB |
| MoE | 6GB | 12GB | 48GB |
| Hybrid | 4GB | 6GB | 12GB |

*For large model size*

### Training Speed

| Architecture | Steps/sec | Memory/batch |
|--------------|-----------|--------------|
| Dense | 2.5 | 8GB |
| Mamba | 3.0 | 6GB |
| MoE | 2.0 | 12GB |
| Hybrid | 2.2 | 7GB |

---

## Decision Guide

```
                        ┌─────────────────────────────────┐
                        │   What is your main use case?   │
                        └───────────────┬─────────────────┘
                                        │
              ┌─────────────────────────┼─────────────────────────┐
              │                         │                         │
              ▼                         ▼                         ▼
      ┌───────────────┐        ┌───────────────┐        ┌───────────────┐
      │  Learning /   │        │    Long       │        │   Multi-      │
      │   Standard    │        │   Context     │        │   Domain      │
      └───────┬───────┘        └───────┬───────┘        └───────┬───────┘
              │                         │                         │
              ▼                         ▼                         ▼
      ┌───────────────┐        ┌───────────────┐        ┌───────────────┐
      │    Dense      │        │    Mamba      │        │     MoE       │
      │  Transformer  │        │     or        │        │   (Sparse)    │
      │               │        │   Hybrid      │        │               │
      └───────────────┘        └───────────────┘        └───────────────┘
```

---

## Source Files

| Architecture | Source File |
|--------------|-------------|
| Dense | `src/model/transformer.py` |
| Mamba | `src/model/mamba.py` |
| MoE | `src/model/moe.py` |
| Hybrid | `src/model/hybrid.py` |
| Config | `src/model/config.py` |

---

## Next Steps

- **Train a model**: [04-TRAINING-PIPELINE.md](04-TRAINING-PIPELINE.md)
- **Understand concepts**: [02-CONCEPTS.md](02-CONCEPTS.md)
- **Deploy to production**: [05-DEPLOYMENT.md](05-DEPLOYMENT.md)
