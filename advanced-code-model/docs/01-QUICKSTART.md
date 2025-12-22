# Quick Start Guide

> Get from zero to generating code in 5 minutes.

---

## Prerequisites

- Python 3.10+
- macOS with Apple Silicon (M1/M2/M3) or CUDA GPU
- 16GB+ RAM (48GB recommended for large model)

---

## Step 1: Setup Environment

```bash
# Clone the repository
git clone <repo-url>
cd advanced-code-model

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

---

## Step 2: Download Pre-trained Model

If you have a pre-trained model, skip to Step 3.

To train from scratch, see [04-TRAINING-PIPELINE.md](04-TRAINING-PIPELINE.md).

```bash
# Check if model exists
ls models/

# Expected files:
# - language_model_best.pth (Stage 1)
# - code_model_best.pth (Stage 2)
# - tool_model_best.pth (Stage 3)
```

---

## Step 3: Generate Code

### Option A: Interactive Mode

```bash
python scripts/interactive.py --checkpoint models/code_model_best.pth
```

Then type prompts like:
```
> #!/bin/bash
> # List all files in current directory
```

### Option B: Single Generation

```bash
python scripts/test_stage2.py
```

This runs pre-defined test prompts through the model.

### Option C: Python API

```python
import torch
from tokenizers import Tokenizer
from src.model import create_model
from src.model.config import get_config

# Load model
config = get_config('large')
config.use_rmsnorm = True
config.use_rope = True
model = create_model(config, device='mps')

# Load checkpoint
checkpoint = torch.load('models/code_model_best.pth', map_location='mps')
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Load tokenizer
tokenizer = Tokenizer.from_file('data/tokenizer/tokenizer.json')

# Generate
prompt = "#!/bin/bash\n# Backup directory\n"
encoding = tokenizer.encode(prompt)
input_ids = torch.tensor([encoding.ids], device='mps')

with torch.no_grad():
    for _ in range(100):
        logits = model(input_ids)
        next_token = logits[0, -1, :].argmax()
        input_ids = torch.cat([input_ids, next_token.unsqueeze(0).unsqueeze(0)], dim=1)

output = tokenizer.decode(input_ids[0].tolist())
print(output)
```

---

## Step 4: Try Different Prompts

### Bash Scripts
```
#!/bin/bash
# Check disk space and alert if low
```

### Python Functions
```
### Language: Python
### Code:
def calculate_fibonacci(n):
```

### SQL Queries
```
### Language: SQL
### Code:
-- Find top 10 customers by revenue
```

---

## Model Sizes

| Size | Parameters | VRAM | Use Case |
|------|------------|------|----------|
| tiny | 10M | 1GB | Testing, CI/CD |
| medium | 150M | 4GB | Development |
| large | 1.6B | 12GB | Production |
| xlarge | 7B | 24GB | Best quality |

Change model size in config:
```python
config = get_config('medium')  # or 'tiny', 'large', 'xlarge'
```

---

## Common Issues

### Out of Memory
```bash
# Reduce batch size or use smaller model
python scripts/test_stage2.py --model-size medium
```

### Tokenizer Not Found
```bash
# Train tokenizer first
python scripts/train_code_tokenizer.py
```

### Checkpoint Not Found
```bash
# Train model first (or download pre-trained)
python scripts/train.py --stage language
```

---

## Next Steps

- **Understand the model**: [02-CONCEPTS.md](02-CONCEPTS.md)
- **Train your own**: [04-TRAINING-PIPELINE.md](04-TRAINING-PIPELINE.md)
- **Deploy to production**: [05-DEPLOYMENT.md](05-DEPLOYMENT.md)

---

## Quick Command Reference

```bash
# Train tokenizer
python scripts/train_code_tokenizer.py

# Train language model (Stage 1)
python scripts/train.py --stage language

# Train code model (Stage 2)
python scripts/train.py --stage code --checkpoint models/language_model_best.pth

# Test the model
python scripts/test_stage2.py

# Interactive mode
python scripts/interactive.py --checkpoint models/code_model_best.pth
```
