# Pretrained Code Model

**Production-Quality Code LLM using Modern Best Practices**

## Overview

This project builds a code generation model using the same approach as CodeLlama, StarCoder, and GitHub Copilot:

```
Stage 1: Pretrain on Natural Language (BookCorpus)
    ↓
    Model learns English, reasoning, logic

Stage 2: Fine-tune on High-Quality Code (Bash scripts)
    ↓
    Model learns programming while retaining language understanding

Result: Model that understands English prompts AND generates code
```

## Why This Approach?

### Code-Only Model
```
Input: "write a script to backup files"
Output: [random bash tokens - doesn't understand]
```

### Pretrained + Fine-tuned Model
```
Input: "write a script to backup files"
Output:
#!/bin/bash
# Backup files to /backup directory
tar -czf /backup/backup_$(date +%Y%m%d).tar.gz /data
echo "Backup complete"
```

## Quick Start

### 1. Download Language Data (BookCorpus)

```bash
# Install dependencies
pip install datasets torch tqdm

# Download BookCorpus (5GB, ~20 min first time)
python scripts/download_language_data.py

# Or use smaller dataset for testing
python scripts/download_language_data.py --use-tinystories
```

### 2. Download Code Data (Bash Scripts)

```bash
# Download high-quality bash scripts
python scripts/download_code_data.py --max-files 500

# Or use existing curated examples
python scripts/download_code_data.py --use-existing
```

### 3. Train Language Model (Stage 1)

```bash
# Pretrain on BookCorpus (~2-4 hours on M1 Max)
python scripts/train_language.py \
    --data-dir data_language \
    --num-steps 10000 \
    --batch-size 16 \
    --model-size small
```

### 4. Fine-tune on Code (Stage 2)

```bash
# Continue training on bash scripts (~30 min)
python scripts/train_code.py \
    --language-checkpoint checkpoints_language/best_model.pt \
    --data-dir data_code \
    --num-steps 2000 \
    --batch-size 8
```

### 5. Generate Code

```bash
# Test the bilingual model
python scripts/generate.py \
    --checkpoint checkpoints_final/best_model.pt \
    --prompt "Create a script to list all large files"
```

## Project Structure

```
pretrained-code-model/
├── model/                   # PyTorch transformer (symlinked)
├── tokenizer/              # BPE tokenizer (symlinked)
├── training/               # Training infrastructure (symlinked)
├── scripts/
│   ├── download_language_data.py   # Download BookCorpus
│   ├── download_code_data.py       # Download bash scripts
│   ├── train_language.py           # Stage 1: Language pretraining
│   ├── train_code.py               # Stage 2: Code fine-tuning
│   └── generate.py                 # Generate code
├── data_language/          # BookCorpus or TinyStories
├── data_code/             # High-quality bash scripts
├── checkpoints_language/   # Language model checkpoints
├── checkpoints_code/      # Code model checkpoints
└── checkpoints_final/     # Final bilingual model
```

## Model Sizes

| Size | Params | Layers | Hidden | Best For |
|------|--------|--------|--------|----------|
| Tiny | 10.9M | 6 | 384 | Quick experiments |
| Small | 48.7M | 12 | 768 | Good quality, fast |
| Medium | 163M | 16 | 1024 | Best quality (slower) |

**Recommended:** Small (48.7M) for balance of quality and speed on M1 Max

## Training Timeline (M1 Max)

### Stage 1: Language Pretraining
- **Data:** BookCorpus (11K books) or TinyStories (5K stories)
- **Steps:** 10,000
- **Time:** 2-4 hours (BookCorpus), 30 min (TinyStories)
- **Result:** Model understands English

### Stage 2: Code Fine-tuning
- **Data:** High-quality bash scripts (500-1000 scripts)
- **Steps:** 2,000
- **Time:** 30 minutes
- **Result:** Model generates code from English prompts

## Features

- ✅ **Modern Architecture:** Same approach as CodeLlama/StarCoder
- ✅ **Bilingual:** Understands English + Generates Code
- ✅ **PyTorch + MPS:** Optimized for M1 Max
- ✅ **High Quality Data:** BookCorpus + curated bash scripts
- ✅ **Production Ready:** Complete training pipeline
- ✅ **Incremental Training:** Can continue improving model

## Example Outputs

### After Language Pretraining Only
```
Input: "Once upon a time"
Output: "Once upon a time there was a little girl who loved to play..."

Input: "#!/bin/bash"
Output: [poor quality - model doesn't understand code yet]
```

### After Code Fine-tuning
```
Input: "Once upon a time"
Output: "Once upon a time there was a little girl who loved to play..."

Input: "#!/bin/bash"
Output:
#!/bin/bash
# Simple script
echo "Hello, World!"
```

### English Prompt → Code Generation
```
Input: "Create a backup script"
Output:
#!/bin/bash
# Backup script
BACKUP_DIR="/backup"
SOURCE="/data"
tar -czf "$BACKUP_DIR/backup_$(date +%Y%m%d).tar.gz" "$SOURCE"
echo "Backup complete"
```

## Technical Details

### Tokenization
- **BPE (Byte Pair Encoding)** with 8000 vocab
- Trained on combined English + Code corpus
- Handles both natural language and code efficiently

### Architecture
- **Transformer decoder** (GPT-style)
- **Multi-head attention** with causal masking
- **Pre-LayerNorm** for training stability
- **Learned positional embeddings**

### Training
- **Optimizer:** AdamW (lr=3e-4 for language, 1e-4 for code)
- **LR Schedule:** Cosine with warmup
- **Gradient Clipping:** max_norm=1.0
- **Device:** MPS (Metal) backend for M1 Max

## Comparison with Other Approaches

| Approach | This Project | bash-code-model | llm-from-scratch |
|----------|--------------|-----------------|------------------|
| Framework | PyTorch | PyTorch | MLX |
| Pretraining | BookCorpus | None | TinyStories |
| Fine-tuning | Bash code | Bash code | None |
| Tokenizer | BPE | Character | BPE |
| English Prompts | ✓ Yes | ✗ No | ✓ Yes |
| Code Generation | ✓ Yes | ✓ Yes | ✗ No |
| Use Case | Production | Learning | Learning |

## Next Steps

After training:

1. **Test Capabilities**
   ```bash
   python scripts/test_bilingual.py
   ```

2. **Build Applications**
   - CLI tool for code generation
   - VS Code extension
   - API service

3. **Improve Quality**
   - Add more code languages (Python, JavaScript)
   - Increase model size
   - Train longer

4. **Deploy**
   - Export to ONNX
   - Optimize for inference
   - Build API

## Resources

- **BookCorpus:** 11,000 books, high-quality English
- **The Stack:** 6TB of code (for scaling up)
- **CodeLlama Paper:** https://arxiv.org/abs/2308.12950
- **StarCoder Paper:** https://arxiv.org/abs/2305.06161

---

**Author:** Vijay Singh
**Framework:** PyTorch with MPS
**Hardware:** Apple M1 Max
