# Scripts Reference

> All scripts organized by training stage.

---

## Directory Structure

Scripts are organized into numbered subdirectories by stage:

```
scripts/
├── 0_setup/          # Data download and environment setup
├── 1_tokenizer/      # Tokenizer training
├── 2_pretrain/       # Language pretraining (Stage 1)
├── 3_finetune/       # Code fine-tuning (Stage 2)
├── 4_tools/          # Tool calling training
├── 5_rlhf/           # RLHF alignment
├── 6_deploy/         # Deployment and serving
├── test/             # Testing scripts
├── utils/            # Utility scripts and domain generators
└── domains/          # Domain-specific data generators
```

Each subdirectory contains a README with detailed instructions.

---

## Quick Reference

```
Stage 0: Tokenizer       → train_code_tokenizer.py
Stage 1: Language        → download_data.py → prepare_datasets.py → train.py --stage language
Stage 2: Code            → download_code_corpus.py → prepare_code_data.py → train.py --stage code
Stage 3: Tools           → prepare_tool_calling_data.py → train.py --stage tools
Stage 4: RLHF            → prepare_rlhf_data.py → train_reward_model.py → train_rlhf.py
Stage 5: Multi-Modal     → prepare_multimodal_data.py → train.py --stage multimodal
Stage 6: RAG             → build_codebase_index.py → rag_search.py
Stage 7: Agent           → run_agent.py
Stage 8: Domain          → generate_domain_data.py → prepare_domain_data.py
Stage 9: Optimization    → quantize_model.py
Stage 10: Deploy         → serve_api.py
```

---

## By Stage

### Stage 0: Tokenizer Training

| Script | Purpose | Command |
|--------|---------|---------|
| `train_code_tokenizer.py` | Train BPE tokenizer on code | `python scripts/train_code_tokenizer.py` |
| `train_tokenizer.py` | Legacy tokenizer trainer | (deprecated) |

---

### Stage 1: Language Pretraining

| Script | Purpose | Command |
|--------|---------|---------|
| `download_sample_data.py` | Download TinyStories | `python scripts/download_sample_data.py` |
| `prepare_datasets.py` | Tokenize for training | `python scripts/prepare_datasets.py` |
| `train.py` | Train language model | `python scripts/train.py --stage language` |

---

### Stage 2: Code Fine-tuning

| Script | Purpose | Command |
|--------|---------|---------|
| `download_code_corpus.py` | Download code from HuggingFace | `python scripts/download_code_corpus.py` |
| `download_bash_corpus.py` | Download bash scripts | `python scripts/download_bash_corpus.py` |
| `prepare_code_data.py` | Tokenize code corpus | `python scripts/prepare_code_data.py` |
| `train.py` | Train code model | `python scripts/train.py --stage code --checkpoint models/language_model_best.pth` |
| `test_stage2.py` | Test code generation | `python scripts/test_stage2.py` |

---

### Stage 3: Tool Calling

| Script | Purpose | Command |
|--------|---------|---------|
| `prepare_tool_calling_data.py` | Generate tool examples | `python scripts/prepare_tool_calling_data.py` |
| `train.py` | Train tool model | `python scripts/train.py --stage tools --checkpoint models/code_model_best.pth` |
| `inference_tool_calling.py` | Interactive tool testing | `python scripts/inference_tool_calling.py` |

---

### Stage 4: RLHF

| Script | Purpose | Command |
|--------|---------|---------|
| `prepare_rlhf_data.py` | Generate preference pairs | `python scripts/prepare_rlhf_data.py` |
| `train_reward_model.py` | Train reward model | `python scripts/train_reward_model.py` |
| `train_rlhf.py` | PPO training | `python scripts/train_rlhf.py` |

---

### Stage 5: Multi-Modal

| Script | Purpose | Command |
|--------|---------|---------|
| `prepare_multimodal_data.py` | Prepare image-text pairs | `python scripts/prepare_multimodal_data.py` |
| `train.py` | Train multimodal model | `python scripts/train.py --stage multimodal` |

---

### Stage 6: RAG

| Script | Purpose | Command |
|--------|---------|---------|
| `build_codebase_index.py` | Index a codebase | `python scripts/build_codebase_index.py --codebase /path` |
| `rag_search.py` | Search indexed code | `python scripts/rag_search.py` |

---

### Stage 7: Agent

| Script | Purpose | Command |
|--------|---------|---------|
| `run_agent.py` | Run autonomous agent | `python scripts/run_agent.py --objective "task"` |

---

### Stage 8: Domain Specialization

| Script | Purpose | Command |
|--------|---------|---------|
| `generate_domain_data.py` | Generate domain examples | `python scripts/generate_domain_data.py` |
| `prepare_domain_data.py` | Prepare domain training data | `python scripts/prepare_domain_data.py` |
| `domains/*.py` | Domain-specific generators | See `domains/` folder |

---

### Stage 9: Optimization

| Script | Purpose | Command |
|--------|---------|---------|
| `quantize_model.py` | Quantize model to INT8/INT4 | `python scripts/quantize_model.py --bits 8` |
| `test_model_quality.py` | Benchmark model quality | `python scripts/test_model_quality.py` |

---

### Stage 10: Deployment

| Script | Purpose | Command |
|--------|---------|---------|
| `serve_api.py` | Start REST API server | `python scripts/serve_api.py --port 8080` |
| `security_scanner.py` | Scan generated code | `python scripts/security_scanner.py` |

---

### Utility Scripts

| Script | Purpose | Command |
|--------|---------|---------|
| `continual_learning.py` | Online learning loop | `python scripts/continual_learning.py` |
| `setup_data.sh` | Setup all data | `bash scripts/setup_data.sh` |

---

## Main Training Script (train.py)

The main training script supports all stages:

```bash
python scripts/train.py \
    --stage {language|code|tools|multimodal|domain} \
    --model-size {tiny|medium|large|xlarge} \
    --checkpoint PATH \
    --batch-size 2 \
    --num-epochs 3 \
    --learning-rate 1e-4 \
    --warmup-steps 2000 \
    --use-rmsnorm \
    --use-rope \
    --use-compile \
    --use-amp
```

See [CONFIG-REFERENCE.md](../docs/reference/CONFIG-REFERENCE.md) for all options.

---

## Domain Data Generators

Located in `scripts/domains/`:

| Script | Domain |
|--------|--------|
| `sql_examples.py` | SQL queries |
| `cloud_aws.py` | AWS CLI and SDK |
| `cloud_gcp.py` | Google Cloud |
| `cloud_azure.py` | Azure |
| `devops.py` | DevOps tooling |
| `ml_examples.py` | Machine Learning |
| `data_engineering.py` | Data pipelines |
| `web_mobile.py` | Web/Mobile development |
| `diagrams.py` | Diagram generation |

---

## Common Workflows

### Train From Scratch

```bash
# 1. Download data
python scripts/download_sample_data.py
python scripts/download_code_corpus.py

# 2. Train tokenizer
python scripts/train_code_tokenizer.py

# 3. Prepare data
python scripts/prepare_datasets.py
python scripts/prepare_code_data.py

# 4. Train Stage 1
python scripts/train.py --stage language --model-size large \
    --use-rmsnorm --use-rope --use-compile --use-amp

# 5. Train Stage 2
python scripts/train.py --stage code \
    --checkpoint models/language_model_best.pth \
    --use-rmsnorm --use-rope --use-compile --use-amp
```

### Quick Test

```bash
python scripts/test_stage2.py
```

### Deploy API

```bash
python scripts/serve_api.py --checkpoint models/code_model_best.pth --port 8080
```

---

## Script Dependencies

```
download_*.py  →  prepare_*.py  →  train.py  →  test_*.py  →  serve_api.py
     ↓              ↓                ↓
 data/raw/     data/processed/    models/
```

Each stage depends on the previous stage's checkpoint.
