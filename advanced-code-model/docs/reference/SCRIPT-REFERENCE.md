# Script Reference

> Complete reference for all scripts in the project.

---

## Quick Reference Table

| Script | Purpose | Stage |
|--------|---------|-------|
| `download_data.py` | Download TinyStories | 1 |
| `download_code_corpus.py` | Download code from HuggingFace | 2 |
| `train_code_tokenizer.py` | Train BPE tokenizer | 0 |
| `prepare_data.py` | Tokenize language data | 1 |
| `prepare_code_data.py` | Tokenize code data | 2 |
| `train.py` | Main training script | 1-3 |
| `test_stage1.py` | Test language model | 1 |
| `test_stage2.py` | Test code model | 2 |
| `serve_api.py` | Start API server | 5 |
| `interactive.py` | Interactive generation | - |

---

## Data Download Scripts

### download_data.py

Download TinyStories dataset for language pretraining.

```bash
python scripts/download_data.py
```

**Output:**
- `data/raw/TinyStoriesV2-GPT4-train.txt`
- `data/raw/TinyStoriesV2-GPT4-valid.txt`

**Options:**
- `--output-dir`: Output directory (default: `data/raw`)

---

### download_code_corpus.py

Download code datasets from HuggingFace.

```bash
python scripts/download_code_corpus.py
```

**Datasets downloaded:**
- MBPP (programming problems)
- HumanEval (code completion)
- Python Code Instructions
- CodeAlpaca
- Evol-Instruct-Code
- Code Exercises
- Glaive Code Assistant

**Output:**
- `data/raw/code_corpus.txt` (75K+ examples)

---

## Tokenizer Scripts

### train_code_tokenizer.py

Train a BPE tokenizer optimized for code.

```bash
python scripts/train_code_tokenizer.py [OPTIONS]
```

**Options:**
| Option | Default | Description |
|--------|---------|-------------|
| `--vocab-size` | 32000 | Vocabulary size |
| `--data-dir` | `data` | Data directory |
| `--output` | `data/tokenizer/tokenizer.json` | Output path |
| `--backup-old` | True | Backup old tokenizer |

**Output:**
- `data/tokenizer/tokenizer.json`

---

## Data Preparation Scripts

### prepare_data.py

Tokenize TinyStories for language training.

```bash
python scripts/prepare_data.py [OPTIONS]
```

**Options:**
| Option | Default | Description |
|--------|---------|-------------|
| `--max-length` | 512 | Sequence length |
| `--val-split` | 0.05 | Validation split |

**Output:**
- `data/processed/train_large.npy`
- `data/processed/val_large.npy`

---

### prepare_code_data.py

Tokenize code corpus for code training.

```bash
python scripts/prepare_code_data.py [OPTIONS]
```

**Options:**
| Option | Default | Description |
|--------|---------|-------------|
| `--input` | `data/raw/code_corpus.txt` | Input corpus |
| `--max-length` | 1024 | Sequence length |
| `--val-split` | 0.05 | Validation split |
| `--output-name` | `code_train_large` | Output file name |

**Output:**
- `data/processed/code_train_large.npy`
- `data/processed/code_val_large.npy`

---

### prepare_tool_calling_data.py

Generate synthetic tool calling examples.

```bash
python scripts/prepare_tool_calling_data.py [OPTIONS]
```

**Options:**
| Option | Default | Description |
|--------|---------|-------------|
| `--num-examples` | 10000 | Number of examples |
| `--output-dir` | `data/processed` | Output directory |

**Output:**
- `data/processed/tool_train.npy`
- `data/processed/tool_val.npy`

---

## Training Scripts

### train.py

Main training script for all stages.

```bash
python scripts/train.py [OPTIONS]
```

**Required Options:**
| Option | Description |
|--------|-------------|
| `--stage` | Training stage: `language`, `code`, `tools` |

**Model Options:**
| Option | Default | Description |
|--------|---------|-------------|
| `--model-size` | `large` | Model size: `tiny`, `medium`, `large`, `xlarge` |
| `--checkpoint` | None | Path to checkpoint to load |
| `--resume` | False | Resume training from checkpoint |

**Training Options:**
| Option | Default | Description |
|--------|---------|-------------|
| `--batch-size` | 2 | Batch size per device |
| `--num-epochs` | 3 | Number of training epochs |
| `--learning-rate` | 1e-4 | Learning rate |
| `--warmup-steps` | 2000 | Warmup steps |
| `--weight-decay` | 0.1 | Weight decay |
| `--grad-clip` | 1.0 | Gradient clipping |

**Data Options:**
| Option | Default | Description |
|--------|---------|-------------|
| `--data-file` | Auto | Training data file |
| `--val-file` | Auto | Validation data file |
| `--data-dir` | `data/processed` | Data directory |

**Optimization Options:**
| Option | Default | Description |
|--------|---------|-------------|
| `--use-rmsnorm` | False | Use RMSNorm |
| `--use-rope` | False | Use RoPE embeddings |
| `--use-compile` | False | Use torch.compile |
| `--use-amp` | False | Use automatic mixed precision |

**Example:**
```bash
python scripts/train.py \
    --stage code \
    --checkpoint models/language_model_best.pth \
    --model-size large \
    --batch-size 2 \
    --num-epochs 5 \
    --learning-rate 5e-6 \
    --use-rmsnorm --use-rope --use-compile --use-amp
```

---

## Testing Scripts

### test_stage1.py

Test language model with story prompts.

```bash
python scripts/test_stage1.py
```

---

### test_stage2.py

Test code model with code prompts.

```bash
python scripts/test_stage2.py
```

**Test prompts:**
- `#!/bin/bash\n# List all files`
- `#!/bin/bash\n# Backup directory`
- `#!/bin/bash\n# Check disk space`

---

### test_tool_calling.py

Test tool calling capability.

```bash
python scripts/test_tool_calling.py
```

---

## Deployment Scripts

### serve_api.py

Start REST API server.

```bash
python scripts/serve_api.py [OPTIONS]
```

**Options:**
| Option | Default | Description |
|--------|---------|-------------|
| `--checkpoint` | `models/code_model_best.pth` | Model checkpoint |
| `--port` | 8080 | API port |
| `--host` | `0.0.0.0` | API host |
| `--device` | `mps` | Device (mps/cuda/cpu) |

---

### interactive.py

Interactive text generation in terminal.

```bash
python scripts/interactive.py [OPTIONS]
```

**Options:**
| Option | Default | Description |
|--------|---------|-------------|
| `--checkpoint` | Required | Model checkpoint |
| `--temperature` | 0.7 | Sampling temperature |
| `--max-tokens` | 100 | Max tokens to generate |

---

## Utility Scripts

### quantize.py

Quantize model for faster inference.

```bash
python scripts/quantize.py \
    --checkpoint models/code_model_best.pth \
    --output models/code_model_int8.pth \
    --bits 8
```

---

### validate_data.py

Check data quality and format.

```bash
python scripts/validate_data.py --data-file data/processed/code_train_large.npy
```

---

## Domain Data Scripts

Located in `scripts/domains/`:

| Script | Purpose |
|--------|---------|
| `sql_examples.py` | Generate SQL examples |
| `cloud_aws.py` | Generate AWS examples |
| `cloud_gcp.py` | Generate GCP examples |
| `cloud_azure.py` | Generate Azure examples |
| `devops.py` | Generate DevOps examples |
| `ml_examples.py` | Generate ML examples |

---

## Common Workflows

### Train from Scratch

```bash
# 1. Download data
python scripts/download_data.py
python scripts/download_code_corpus.py

# 2. Train tokenizer
python scripts/train_code_tokenizer.py

# 3. Prepare data
python scripts/prepare_data.py
python scripts/prepare_code_data.py

# 4. Train Stage 1
python scripts/train.py --stage language

# 5. Train Stage 2
python scripts/train.py --stage code --checkpoint models/language_model_best.pth
```

### Resume Training

```bash
python scripts/train.py \
    --stage code \
    --checkpoint models/code_model_epoch_2.pth \
    --resume
```

### Quick Test

```bash
python scripts/test_stage2.py
```
