# Quick Start Guide

Get up and running in **5 minutes**!

## Installation (1 minute)

```bash
# Clone
git clone <your-repo-url>
cd code-llm-from-scratch

# Install
pip install -r requirements.txt

# Verify
python -c "import torch; print('✓ PyTorch installed')"
```

## Training (3-5 hours total)

### Stage 1: Language Pretraining (2-4 hours)

```bash
python scripts/train_language.py
```

**What happens**:
- Trains BPE tokenizer on TinyStories
- Creates language model (learns English)
- Saves to `models/language/`

**Output**:
```
Epoch 1/10: loss=3.8
Epoch 5/10: loss=2.5
Epoch 10/10: loss=2.3
✓ Model saved
```

### Stage 2: Code Fine-Tuning (30-60 minutes)

```bash
python scripts/train_code.py
```

**What happens**:
- Loads pretrained language model
- Fine-tunes on 100+ bash scripts
- Saves to `models/code/`

**Output**:
```
Epoch 1/20: loss=2.1
Epoch 10/20: loss=1.2
Epoch 20/20: loss=1.0
✓ Model saved
```

## Generation (Instant!)

### Basic Usage

```bash
python scripts/generate.py \
    --prompt "#!/bin/bash\n# Create a backup script"
```

### With Options

```bash
python scripts/generate.py \
    --prompt "#!/bin/bash\n# Monitor system resources" \
    --max-length 200 \
    --temperature 0.8 \
    --top-k 50
```

### In Python

```python
from examples.basic_usage import load_model_and_tokenizer, generate_code

model, tokenizer, device = load_model_and_tokenizer()
code = generate_code(model, tokenizer, "Create a deployment script", device)
print(code)
```

## Common Tasks

### Generate Multiple Scripts

```bash
for prompt in "backup" "deploy" "monitor"; do
    python scripts/generate.py \
        --prompt "#!/bin/bash\n# ${prompt} script" \
        --output "${prompt}.sh"
done
```

### Fine-Tune on Your Own Data

```python
# 1. Prepare data
my_scripts = ["#!/bin/bash\n...", "#!/bin/bash\n...", ...]

# 2. Save as JSON
import json
with open('my_data.json', 'w') as f:
    json.dump({'scripts': my_scripts}, f)

# 3. Update data path in scripts/train_code.py
CODE_DATA_DIR = "path/to/my_data.json"

# 4. Train
python scripts/train_code.py --num-epochs 50
```

### Adjust Model Size

Edit `scripts/train_language.py`:
```python
MODEL_SIZE = "tiny"    # 10.9M params (fast experiments)
MODEL_SIZE = "small"   # 48.7M params (recommended)
MODEL_SIZE = "medium"  # 163M params (best quality)
```

## Troubleshooting

**Out of memory?**
```bash
# Use smaller batch size
python scripts/train_language.py --batch-size 8

# Or use tiny model
python scripts/train_language.py --model-size tiny
```

**Slow training?**
```bash
# Check if using GPU/MPS
python -c "import torch; print(f'MPS: {torch.backends.mps.is_available()}')"
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"
```

**Bad generation quality?**
```bash
# Train longer
python scripts/train_language.py --num-epochs 20
python scripts/train_code.py --num-epochs 40

# Or adjust temperature
python scripts/generate.py --prompt "..." --temperature 0.5
```

## File Locations

```
models/
├── language/
│   ├── language_model_final.pt      # Pretrained model
│   └── language_tokenizer.json      # Tokenizer
└── code/
    ├── code_model_final.pt          # Fine-tuned model
    └── generation_config.json       # Config

data/
├── tinystories_5000.txt             # Language data
└── code/
    ├── bash_scripts/                # 100+ scripts
    └── bash_scripts.json            # JSON format
```

## Quick Commands Cheat Sheet

```bash
# Full training pipeline
python scripts/train_language.py && python scripts/train_code.py

# Generate with defaults
python scripts/generate.py --prompt "#!/bin/bash\n# Your prompt"

# Interactive Python
python examples/basic_usage.py

# Check model size
ls -lh models/code/code_model_final.pt

# Check training data stats
cat data/code/stats.json

# View sample bash script
head -50 data/code/bash_scripts/script_001.sh
```

## Next Steps

1. **Read** `GETTING_STARTED.md` for detailed tutorial
2. **Explore** `docs/ARCHITECTURE.md` for technical details
3. **Review** `presentation/PRESENTATION_GUIDE.md` for demo ideas
4. **Experiment** with different prompts and settings!

---

**Happy coding! 🚀**

For help: See full documentation in `docs/` or open an issue on GitHub.
