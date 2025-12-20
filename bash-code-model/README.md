# Bash Code Generation Model

A small transformer model for generating bash scripts, trained with PyTorch and optimized for Apple M1 Max using Metal Performance Shaders (MPS).

## Why a Separate Coding Model?

**Domain-Specific Advantages:**
- **Smaller & Faster**: 10-50M params vs 100M+ for general LLMs
- **Code-Aware**: Specialized tokenizer preserves syntax
- **Quick Training**: Minutes instead of hours on specific task
- **Better Quality**: Focused training = better bash code

## Model Architecture

**Tiny Coder** (~12M parameters)
```
Layers: 6
Hidden size: 384
Attention heads: 6
FFN size: 1536
Context: 512 tokens
Vocabulary: 8K tokens (code-optimized)
```

**Small Coder** (~50M parameters)
```
Layers: 12
Hidden size: 768
Attention heads: 12
FFN size: 3072
Context: 1024 tokens
```

## PyTorch vs MLX

| Feature | PyTorch (this project) | MLX (previous) |
|---------|----------------------|----------------|
| **Adoption** | Industry standard | Apple-specific |
| **Resources** | Massive ecosystem | Limited but growing |
| **Frameworks** | HuggingFace, Lightning | Native only |
| **M1 Optimization** | MPS backend (good) | Native (better) |
| **Debugging** | Excellent tools | Basic |
| **Transfer** | Works everywhere | Apple Silicon only |

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Verify MPS (Metal) is available
python -c "import torch; print(f'MPS available: {torch.backends.mps.is_available()}')"

# Collect bash scripts
python scripts/collect_bash_data.py

# Train model
python scripts/train_coder.py --model-size tiny --max-steps 5000

# Generate bash code
python scripts/generate_code.py --prompt "# Script to list files"
```

## Dataset

**Sources:**
1. GitHub bash repositories (public, permissive licenses)
2. Shell script collections (awesome-shell)
3. Your own bash scripts
4. Documentation examples

**Size**: ~10-50MB of quality bash scripts

## Code Tokenization

Unlike natural language, code has special requirements:

**Character-level** (simplest):
- Pro: No unknown tokens, perfect reconstruction
- Con: Very long sequences
- Use: Quick prototypes

**BPE with code bias** (recommended):
- Pro: Balances efficiency and syntax preservation
- Con: Needs careful training
- Use: Production models

**Examples:**
```bash
# Character tokenization
"ls -la" → ['l', 's', ' ', '-', 'l', 'a']

# BPE code tokenization
"ls -la" → ['ls', ' -', 'la']  # Preserves common patterns
```

## Training on M1 Max

**Metal Performance Shaders (MPS)**:
```python
import torch

# Check MPS availability
assert torch.backends.mps.is_available()
device = torch.device("mps")

# Model to MPS
model = model.to(device)

# Training automatically uses Metal GPU
```

**Performance:**
- **Tiny**: ~30K tokens/sec
- **Small**: ~15K tokens/sec
- **Memory**: Unified architecture = efficient

## Project Structure

```
bash-code-model/
├── data/               # Bash scripts dataset
├── tokenizer/          # Code tokenizer
│   └── code_tokenizer.py
├── model/              # PyTorch transformer
│   ├── transformer.py
│   └── config.py
├── training/           # Training loop
│   ├── trainer.py
│   └── data_loader.py
├── scripts/            # Utilities
│   ├── collect_bash_data.py
│   ├── train_coder.py
│   └── generate_code.py
└── examples/           # Sample generated code
```

## Learning Objectives

1. **PyTorch fundamentals**: Tensors, autograd, MPS backend
2. **Code tokenization**: Different from natural language
3. **Domain specialization**: Why smaller is better for specific tasks
4. **MPS optimization**: Leveraging M1 Max GPU
5. **Code generation**: Temperature, top-k, syntax validation

## Next Steps

1. Install dependencies
2. Collect bash dataset
3. Train tiny model (10M params, ~10 minutes)
4. Generate your first bash script!

---

**Comparison with LLM Project:**
- LLM: General text, 50M-3B params, MLX
- This: Bash code, 10-50M params, PyTorch + MPS
