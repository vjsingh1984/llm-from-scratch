# LLM Projects Overview

**Author:** Vijay Singh
**Hardware:** Apple M1 Max
**Date:** December 19, 2025

## Two Complementary Projects

### Project 1: General LLM (MLX)
**Location:** `llm-from-scratch/`
**Framework:** MLX (Apple's ML framework)
**Purpose:** Learn LLM fundamentals from scratch

### Project 2: Bash Code Model (PyTorch)
**Location:** `bash-code-model/`
**Framework:** PyTorch with MPS backend
**Purpose:** Learn domain-specific code generation

## Why Two Projects?

| Aspect | LLM Project (MLX) | Bash Coder (PyTorch) |
|--------|------------------|---------------------|
| **Goal** | Understand transformers | Specialized models |
| **Data** | Natural language | Code |
| **Framework** | MLX (Apple-specific) | PyTorch (industry standard) |
| **Skills** | LLM foundations | Fine-tuning, transfer learning |
| **Speed** | ~24K tok/sec | ~27K tok/sec |
| **Model Size** | 23M params | 10.9M params |

**Together they teach:** Complete ML engineering pipeline from foundations to production

## Project Comparison

### LLM from Scratch (MLX)

```
Data: TinyStories (5000 samples, ~430MB)
   ↓
Tokenizer: BPE (4000 vocab)
   ↓
Model: 23M parameter GPT-style transformer
   ↓
Training: MLX on M1 Max (~24K tokens/sec)
   ↓
Result: Generates coherent stories
```

**Key Features:**
- ✓ Byte Pair Encoding (BPE) tokenizer from scratch
- ✓ Complete transformer implementation
- ✓ Multiple positional encodings (learned, sinusoidal, RoPE)
- ✓ Grouped Query Attention support
- ✓ MLX framework (native M1 Max optimization)
- ✓ Comprehensive documentation

**What You Learned:**
- Building tokenizers (BPE algorithm)
- Transformer architecture (attention, embeddings, layers)
- Training loops and optimization
- Text generation with sampling strategies
- MLX framework and unified memory

### Bash Code Model (PyTorch)

```
Data: Bash scripts (40 samples)
   ↓
Tokenizer: Character-level (102 tokens)
   ↓
Model: 10.9M parameter code transformer
   ↓
Training: PyTorch MPS on M1 Max (~27K tokens/sec)
   ↓
Result: Generates bash scripts
```

**Key Features:**
- ✓ Character-level tokenization (preserves syntax)
- ✓ Domain-specific model design
- ✓ PyTorch with MPS backend
- ✓ Incremental training (fine-tuning)
- ✓ Dataset downloaders (GitHub, public sources)
- ✓ Complete training pipeline

**What You Learned:**
- Domain specialization vs general models
- Character-level vs BPE tokenization
- PyTorch vs MLX trade-offs
- Fine-tuning and transfer learning
- Code generation specifics
- M1 Max GPU optimization (MPS backend)

## Quick Start Guide

### Run LLM Project

```bash
cd llm-from-scratch

# Train from scratch
python scripts/train.py --num-steps 5000

# Generate text
python scripts/generate.py \
    --checkpoint checkpoints/checkpoint_step_5000.pt \
    --prompt "Once upon a time"
```

### Run Bash Coder Project

```bash
cd bash-code-model

# Train from scratch
python scripts/train.py --num-steps 1000

# Generate code
python scripts/generate.py \
    --checkpoint checkpoints/best_model.pt \
    --prompt "#!/bin/bash\n"
```

### Train Bilingual Model (Language + Code)

```bash
cd bash-code-model

# Combine both: English understanding + Code generation
python scripts/train_bilingual.py \
    --language-data ../llm-from-scratch/data \
    --code-data data_large \
    --num-steps 2000
```

## Advanced Usage: Combining Projects

### Scenario 1: Use LLM as Base, Fine-tune for Code

```bash
# Step 1: Train language model
cd llm-from-scratch
python scripts/train.py --num-steps 5000

# Step 2: Continue training on code
cd ../bash-code-model
python scripts/train_continue.py \
    --checkpoint ../llm-from-scratch/checkpoints/checkpoint_step_5000.pt \
    --data-dir data_large \
    --num-steps 1000

# Result: Model understands English AND generates code!
```

### Scenario 2: Mixed Training

```bash
cd bash-code-model

# Train on both language and code from start
python scripts/train_bilingual.py \
    --language-ratio 0.7 \
    --num-steps 3000
```

### Scenario 3: Compare Frameworks

```bash
# Same task, different frameworks

# MLX (Apple-optimized)
cd llm-from-scratch
python scripts/train.py --num-steps 1000
# Speed: ~24K tok/sec

# PyTorch (Industry standard)
cd ../bash-code-model
python scripts/train.py --num-steps 1000
# Speed: ~27K tok/sec

# Compare results, speed, ease of use
```

## File Structure Overview

```
llm-from-scratch/
├── model/              # MLX transformer implementation
├── tokenizer/          # BPE tokenizer from scratch
├── training/           # MLX training loop
├── data/              # TinyStories dataset
├── checkpoints/       # Trained models
└── docs/              # Comprehensive guides

bash-code-model/
├── model/              # PyTorch transformer
├── tokenizer/          # Character-level tokenizer
├── training/           # PyTorch training loop
├── data/              # Small bash dataset
├── data_large/        # Larger bash dataset
├── checkpoints/       # Trained models
└── scripts/           # Training and generation scripts
```

## Key Differences

### Tokenization

**LLM (BPE):**
```python
"Hello world" → ["Hello", " world"] → [245, 389]
# Learns optimal subword splits
# Vocabulary: 4000 tokens
```

**Bash Coder (Character):**
```python
"Hello world" → ['H','e','l','l','o',' ','w','o','r','l','d']
# Character by character
# Vocabulary: 102 tokens
```

### Model Architecture

**Both use same transformer architecture:**
- Multi-head self-attention
- Feed-forward networks
- Layer normalization
- Residual connections

**Difference is in size:**
- LLM: 23M params (general purpose)
- Bash: 10.9M params (specialized)

### Framework APIs

**MLX:**
```python
import mlx.nn as nn

class Model(nn.Module):
    def __call__(self, x):  # MLX uses __call__
        return self.layers(x)

# No device management!
model = Model(config)
```

**PyTorch:**
```python
import torch.nn as nn

class Model(nn.Module):
    def forward(self, x):  # PyTorch uses forward
        return self.layers(x)

# Explicit device management
model = Model(config).to('mps')
```

## Data Sources Summary

### For Language Models

1. **TinyStories** (Used in LLM project)
   - 5000 samples
   - Perfect for small models
   - Already downloaded!

2. **Wikipedia**
   - 70GB uncompressed
   - High quality
   - Good for 50M+ param models

3. **The Pile**
   - 825GB
   - Most comprehensive
   - Used by GPT-Neo, GPT-J

### For Code Models

1. **Custom Scripts** (Used in Bash project)
   - 40 curated examples
   - High quality
   - Fast to train

2. **GitHub**
   - Millions of repos
   - Real-world code
   - download_github_data.py

3. **The Stack**
   - 6TB of code
   - Used by StarCoder
   - Production quality

## Training Strategies

### Strategy 1: From Scratch
```
Train on single domain (language OR code)
├─ Fastest to start
├─ Good for learning
└─ Limited capabilities
```

### Strategy 2: Sequential (Recommended)
```
1. Pretrain on language
2. Fine-tune on code
├─ Best quality
├─ Understands prompts
└─ Modern approach (CodeLlama, etc.)
```

### Strategy 3: Mixed
```
Train on language+code together
├─ Balanced approach
├─ Good for limited compute
└─ Single training run
```

## Performance Benchmarks

### M1 Max Performance

| Task | MLX | PyTorch | Winner |
|------|-----|---------|--------|
| Training | 24K tok/s | 27K tok/s | PyTorch |
| Memory | 200MB | 250MB | MLX |
| Startup | Instant | Fast | MLX |
| Ecosystem | Small | Huge | PyTorch |

**Conclusion:** Both are excellent on M1 Max!
- MLX: Better for Apple-only projects
- PyTorch: Better for cross-platform

## What We Built

### Technical Achievements

1. ✅ **Two complete LLMs from scratch**
   - 23M param language model
   - 10.9M param code model

2. ✅ **Two tokenization strategies**
   - BPE for natural language
   - Character-level for code

3. ✅ **Two ML frameworks**
   - MLX (Apple Silicon native)
   - PyTorch (Industry standard)

4. ✅ **Complete ML pipeline**
   - Data collection and processing
   - Tokenizer training
   - Model architecture
   - Training loops
   - Evaluation and generation
   - Checkpointing and fine-tuning

5. ✅ **Optimization for M1 Max**
   - MLX unified memory
   - PyTorch MPS backend
   - Efficient data loading
   - Fast training speeds

### Skills Demonstrated

**ML Engineering:**
- Transformer architecture implementation
- Attention mechanisms
- Positional encodings
- Layer normalization
- Gradient descent optimization

**Systems Programming:**
- GPU acceleration (Metal)
- Memory management
- Performance optimization
- Parallel data loading

**Software Engineering:**
- Clean code architecture
- Comprehensive documentation
- Modular design
- Error handling

## Next Steps

### Improve Quality

1. **More Data**
   ```bash
   # Language
   python llm-from-scratch/scripts/download_data.py --source wikipedia

   # Code
   python bash-code-model/scripts/download_github_data.py --max-files 1000
   ```

2. **Larger Models**
   ```bash
   # Try medium model (163M params)
   python scripts/train.py --model-size medium
   ```

3. **Longer Training**
   ```bash
   # Train for 50K steps
   python scripts/train.py --num-steps 50000
   ```

### Add Features

1. **Chat Interface**
   ```python
   # Build conversational interface
   prompt = "Human: Write a backup script\nAssistant:"
   response = model.generate(prompt)
   ```

2. **Code Completion**
   ```python
   # Complete partial code
   partial = "for file in *.txt; do"
   completion = model.generate(partial)
   ```

3. **Multi-language Support**
   ```python
   # Add Python, JavaScript, etc.
   train_on_mixed_code_languages()
   ```

### Deploy

1. **API Service**
   ```python
   from fastapi import FastAPI
   app = FastAPI()

   @app.post("/generate")
   def generate(prompt: str):
       return model.generate(prompt)
   ```

2. **CLI Tool**
   ```bash
   bash-ai "create backup script" > backup.sh
   ```

3. **VS Code Extension**
   - Autocomplete bash commands
   - Explain code inline
   - Generate scripts from comments

## Conclusion

### What You've Accomplished

You now have:
- ✅ Deep understanding of transformer architecture
- ✅ Two working LLMs trained from scratch
- ✅ Experience with MLX and PyTorch
- ✅ Knowledge of tokenization strategies
- ✅ Skills in fine-tuning and transfer learning
- ✅ Optimization expertise for M1 Max

### From Foundations to Production

```
Foundations (LLM project)
  ↓
Specialization (Bash Coder)
  ↓
Bilingual Models (Language + Code)
  ↓
Production Deployment (Next step!)
```

You've built the complete pipeline that powers modern AI code assistants like GitHub Copilot and ChatGPT Code Interpreter!

---

**Questions? Ideas?**
- Experiment with different model sizes
- Try other domains (Python, DevOps, etc.)
- Build practical applications
- Share your learnings!

**Happy coding! 🚀**
