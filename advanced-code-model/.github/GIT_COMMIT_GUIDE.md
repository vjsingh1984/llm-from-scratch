# Git Commit Guide

**What to commit and what to ignore**

---

## ✅ **SAFE TO COMMIT** (Small Files, ~50MB total)

### **Source Code**
```bash
src/
├── model/
│   ├── __init__.py
│   ├── config.py
│   ├── transformer.py
│   ├── mamba.py
│   ├── moe.py
│   ├── hybrid.py
│   ├── vision_encoder.py
│   └── multimodal.py
├── rag/
│   └── vector_store.py
└── agent/
    └── planner.py
```

### **Scripts**
```bash
scripts/
├── train.py                        # Main training script
├── train_tokenizer.py              # Tokenizer training
├── download_data.py                # Data download/prep
├── setup_data.sh                   # NEW! Data setup automation
├── test_stage2.py                  # NEW! Model testing
├── inference_tool_calling.py       # Tool calling inference
├── prepare_tool_calling_data.py    # Stage 3 data prep
├── prepare_rlhf_data.py            # Stage 4 data prep
├── train_reward_model.py           # Stage 4a training
├── train_rlhf.py                   # Stage 4b PPO training
├── prepare_multimodal_data.py      # Stage 5 data prep
├── build_codebase_index.py         # Stage 6 indexing
├── rag_search.py                   # Stage 6 search
├── run_agent.py                    # Stage 7 agent
├── prepare_domain_data.py          # Stage 8 data prep
├── quantize_model.py               # Stage 9 optimization
└── continual_learning.py           # Stage 10 continuous learning
```

### **Documentation**
```bash
docs/
├── README.md
├── STEP_BY_STEP_GUIDE.md          # NEW! Step-by-step instructions
├── TRAINING_STRATEGIES.md         # NEW! Visual guide with SVG diagrams
├── QUICK_REFERENCE.md             # NEW! Quick reference card
├── DATA_SETUP.md                  # NEW! Data setup guide
├── TOOL_CALLING_GUIDE.md
├── RLHF_GUIDE.md
├── MULTIMODAL_GUIDE.md
├── RAG_GUIDE.md
├── AGENT_GUIDE.md
├── ADVANCED_STAGES_GUIDE.md
├── ARCHITECTURE_COMPARISON.md
├── TRAINING_GUIDE.md
└── FUTURE_RESEARCH.md
```

### **Config Files**
```bash
.gitignore                         # UPDATED! Comprehensive rules
README.md                          # UPDATED! Complete 10-stage pipeline
GIT_COMMIT_GUIDE.md               # NEW! This file
requirements.txt                   # If you have one
```

### **Small Data Files** (Optional)
```bash
data/tokenizer/tokenizer.json     # ~3MB - Consider including
models/*_training_history.json    # Small JSON logs - OK to include
```

---

## ❌ **DO NOT COMMIT** (Large Files, in .gitignore)

### **Model Checkpoints** (500MB - 2GB each!)
```bash
models/language_model_best.pth        # ~2GB ❌
models/code_model_best.pth            # ~2GB ❌
models/tool_calling_model_best.pth    # ~2GB ❌
models/*.pth                          # ALL model files ❌
```

### **Training Data** (Hundreds of MB)
```bash
data/processed/*.npy                  # ~500MB ❌
data/raw/                             # ~100MB ❌
data/bash/raw/repos/                  # ~400MB ❌
data/processed/images/                # Generated images ❌
data/rag/                             # RAG vector store ❌
```

### **Python Cache**
```bash
__pycache__/                          # ❌
*.pyc                                 # ❌
venv/                                 # ❌
.DS_Store                             # ❌
```

---

## 📊 **Size Comparison**

**WITH large files (DON'T DO THIS!):**
```
Repository size: ~5-10GB 😱
Clone time: 20-30 minutes
Push time: 30-60 minutes
```

**WITHOUT large files (CORRECT!):**
```
Repository size: ~50MB ✅
Clone time: 10-30 seconds
Push time: 30-60 seconds
```

---

## 🚀 **How to Commit**

### **First Time Setup**

```bash
cd /Users/vijaysingh/code/vijayllm/llm-from-scratch/advanced-code-model

# Initialize git (if not already)
git init

# Check .gitignore is working
git status

# You should see ONLY small files listed, NOT:
# - models/*.pth
# - data/processed/*.npy
# - data/raw/
```

### **Add Files**

```bash
# Add all safe files
git add .

# Verify what will be committed
git status

# Expected: ~100-200 files, all small
# NOT expected: Any .pth or .npy files
```

### **Commit**

```bash
git commit -m "Complete 10-stage LLM training pipeline

- Added all 10 training stages (Language → Continuous Learning)
- Visual training strategy guide with SVG diagrams
- Comprehensive documentation for each stage
- Data setup automation scripts
- RLHF, Multi-Modal, RAG, Agentic, and optimization stages
- Step-by-step guides and quick reference
- Complete .gitignore for large files"
```

### **Push to GitHub**

```bash
# Create repo on GitHub first, then:
git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPO.git
git branch -M main
git push -u origin main
```

---

## 🔍 **Verify Before Pushing**

### **Check Repository Size**

```bash
# Should be ~50MB or less
du -sh .git
```

### **Check No Large Files**

```bash
# This should return EMPTY (no large files staged)
git ls-files | grep -E '\.(pth|npy)$'

# If you see any .pth or .npy files, they're NOT in .gitignore!
# Fix .gitignore and run: git rm --cached <file>
```

### **Check .gitignore is Working**

```bash
# Check ignored files (should include models/*.pth, data/processed/*.npy)
git status --ignored

# You should see:
# Ignored files:
#   models/*.pth
#   data/processed/*.npy
#   data/raw/
#   etc.
```

---

## 📝 **README for Other Users**

When someone clones your repo, they should:

1. **Clone repository:**
   ```bash
   git clone https://github.com/YOUR_USERNAME/YOUR_REPO.git
   cd YOUR_REPO
   ```

2. **Setup data:**
   ```bash
   chmod +x scripts/setup_data.sh
   ./scripts/setup_data.sh
   ```

3. **Start training:**
   ```bash
   python3 scripts/train.py --stage language --model-size large
   ```

---

## 🎯 **Summary**

### **DO Commit:**
- ✅ Source code (.py files)
- ✅ Documentation (.md files)
- ✅ Scripts (setup, training, inference)
- ✅ Config files (.gitignore, requirements.txt)
- ✅ Small data files (<10MB)

### **DON'T Commit:**
- ❌ Model checkpoints (.pth files)
- ❌ Training data (.npy files)
- ❌ Downloaded datasets
- ❌ Generated images
- ❌ Python cache
- ❌ Virtual environments

### **Instead, Provide:**
- ✅ Scripts to download data
- ✅ Scripts to generate synthetic data
- ✅ Documentation on data setup
- ✅ Clear instructions in README

---

## ✅ **You're Ready!**

Your repository is now configured correctly:
1. ✅ Comprehensive .gitignore
2. ✅ Data setup scripts
3. ✅ Complete documentation
4. ✅ All source code

**Safe to commit and push!** 🚀

```bash
git add .
git status  # Verify no large files
git commit -m "Your commit message"
git push
```

---

**Repository size will be ~50MB instead of ~5GB!** 🎉
