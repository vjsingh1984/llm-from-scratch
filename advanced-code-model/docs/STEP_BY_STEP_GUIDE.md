# Complete Step-by-Step Training Guide

**📋 Clear, sequential instructions for all 10 stages**

---

## 📚 **Important: Read This First!**

**[→ Training Strategies Guide (Staged vs Joint)](TRAINING_STRATEGIES.md)**

Before you start, understand WHY we use staged training:
- 🎓 Visual diagrams explaining curriculum learning
- ⚠️ Why joint training fails (with examples)
- 🔧 How to avoid catastrophic forgetting
- 📊 Quality vs time trade-offs

**This will help you understand the training process and avoid common mistakes!**

---

## ✅ Stage 1: Language Pretraining [COMPLETED]

**Status**: Already done in your setup
**Data**: TinyStories dataset (already prepared)
**Output**: `models/language_model_best.pth`

---

## ✅ Stage 2: Code Fine-tuning [COMPLETED]

**Status**: ✓ You just finished this!
**Data**: Bash scripts (already prepared)
**Output**: `models/code_model_best.pth`

### Test Stage 2 (Optional but Recommended)

```bash
python3 scripts/test_stage2.py
```

**What it does**: Generates bash scripts from prompts to verify your model works

**Expected output**: 3 generated bash scripts

---

## 🔨 Stage 3: Tool Calling [NEXT STEP]

**Prerequisites**:
- ✅ Stage 2 complete (`models/code_model_best.pth` exists)

**Data needed**: Synthetic tool-calling examples (you'll generate these)

### Step 3.1: Prepare Data (Required)

```bash
python3 scripts/prepare_tool_calling_data.py
```

**What it does**:
- Generates 2,000 synthetic examples of tool usage
- Creates format: prompt → tool call → tool result → response

**Output files**:
- `data/processed/tool_calling_train.npy` (1,800 examples)
- `data/processed/tool_calling_val.npy` (200 examples)
- `data/processed/tool_definitions.json`

**Time**: ~1-2 minutes

**How to verify**:
```bash
ls -lh data/processed/tool_calling_*
```

You should see:
```
tool_calling_train.npy    (~14MB)
tool_calling_val.npy      (~1.5MB)
tool_calling_example.txt
```

---

### Step 3.2: Train Tool Calling

```bash
python3 scripts/train.py \
  --stage tool_calling \
  --architecture dense \
  --model-size large \
  --checkpoint models/code_model_best.pth \
  --batch-size 2 \
  --gradient-accumulation-steps 4 \
  --use-compile \
  --use-rmsnorm \
  --use-rope \
  --use-amp \
  --use-gradient-checkpointing \
  --num-epochs 3 \
  --steps-per-epoch 300 \
  --learning-rate 5e-6 \
  --warmup-steps 100
```

**What it does**: Fine-tunes your code model to use tools (execute, search, analyze)

**Output**: `models/tool_calling_model_best.pth`

**Expected**:
- **Time**: 1-2 hours
- **Memory**: 10-12GB
- **Target val loss**: <2.0

---

### Step 3.3: Test Tool Calling

```bash
python3 scripts/inference_tool_calling.py \
  --checkpoint models/tool_calling_model_best.pth \
  --model-size large \
  --device mps
```

**What it does**: Interactive session where model can execute code, search docs, etc.

**Example**:
```
You: Write a factorial function and test it

🔧 Tool Call: execute_python
   Code: def factorial(n): ...

📤 Result: 120
```

Type `quit` to exit.

---

## 🎯 Stage 4: RLHF (Alignment)

**Prerequisites**:
- ✅ Stage 3 complete (`models/tool_calling_model_best.pth` exists)

**What it does**: Teaches model to prefer clean, efficient, secure code

### Step 4.1: Prepare Preference Data

```bash
python3 scripts/prepare_rlhf_data.py
```

**Output**:
- `data/processed/rlhf_train_chosen.npy`
- `data/processed/rlhf_train_rejected.npy`
- `data/processed/rlhf_preferences.json`

**Time**: ~1-2 minutes

---

### Step 4.2: Train Reward Model

```bash
python3 scripts/train_reward_model.py \
  --checkpoint models/tool_calling_model_best.pth \
  --model-size large \
  --batch-size 4 \
  --num-epochs 3 \
  --learning-rate 1e-5 \
  --device mps
```

**Output**: `models/reward_model_best.pth`

**Expected**:
- **Time**: 30-45 minutes
- **Target val accuracy**: >80%

---

### Step 4.3: PPO Training

```bash
python3 scripts/train_rlhf.py \
  --checkpoint models/tool_calling_model_best.pth \
  --reward-model models/reward_model_best.pth \
  --model-size large \
  --batch-size 2 \
  --num-epochs 1 \
  --steps-per-epoch 100 \
  --learning-rate 1e-6 \
  --device mps
```

**Output**: `models/rlhf_model_best.pth`

**Expected**:
- **Time**: 1-2 hours
- **Metric**: Increasing mean reward

---

## 🖼️ Stage 5: Multi-Modal (Vision + Code)

**Prerequisites**:
- ✅ Stage 4 complete (`models/rlhf_model_best.pth` exists)

**What it does**: Model can understand images, diagrams, UI mockups

### Step 5.1: Prepare Multi-Modal Data

```bash
python3 scripts/prepare_multimodal_data.py
```

**Output**:
- `data/processed/multimodal_train_tokens.npy`
- `data/processed/multimodal_val_tokens.npy`
- `data/processed/images/` (1,000 synthetic images)

**Time**: ~2-3 minutes

---

### Step 5.2: Train Multi-Modal

```bash
python3 scripts/train.py \
  --stage multimodal \
  --architecture dense \
  --model-size large \
  --checkpoint models/rlhf_model_best.pth \
  --batch-size 2 \
  --num-epochs 3 \
  --learning-rate 1e-5 \
  --device mps
```

**Output**: `models/multimodal_model_best.pth`

**Expected**:
- **Time**: 2-3 hours
- **Target val loss**: <2.5

---

## 🔍 Stage 6: RAG (Codebase Search)

**Prerequisites**:
- ✅ Stage 5 complete (`models/multimodal_model_best.pth` exists)
- A codebase to index (can use this project!)

**What it does**: Enables semantic search over large codebases

### Step 6.1: Build Codebase Index

```bash
# Index this project as example
python3 scripts/build_codebase_index.py \
  --checkpoint models/multimodal_model_best.pth \
  --codebase /Users/vijaysingh/code/vijayllm/llm-from-scratch/advanced-code-model \
  --model-size large \
  --device mps \
  --chunk-size 512 \
  --extensions .py,.md
```

**Output**:
- `data/rag/embeddings.npy` (code embeddings)
- `data/rag/metadata.json` (chunk info)

**Time**: 1-2 hours (depends on codebase size)

---

### Step 6.2: Test RAG Search

```bash
# Interactive search
python3 scripts/rag_search.py

# Or single query
python3 scripts/rag_search.py --query "training loop"
```

---

## 🤖 Stage 7: Agentic Workflows

**Prerequisites**:
- ✅ Stage 6 complete (RAG index built)

**What it does**: Multi-step task planning and execution

### Test Agent

```bash
python3 scripts/run_agent.py \
  --objective "Implement a binary search tree with tests"
```

**What it does**: Creates plan, executes steps sequentially

**No training needed** - planning system uses existing model

---

## 🎯 Stage 8: Domain Specialization

**Prerequisites**:
- ✅ Any model from Stage 3+ (recommend Stage 5+)

**What it does**: Specialize in web, data science, devops, mobile, or backend

### Step 8.1: Prepare Domain Data

```bash
python3 scripts/prepare_domain_data.py
```

**Output**: Domain-specific examples for all 5 domains

**Time**: ~2-3 minutes

---

### Step 8.2: Train on Domain

```bash
# Example: Web development specialist
python3 scripts/train.py \
  --stage domain \
  --architecture dense \
  --model-size large \
  --checkpoint models/multimodal_model_best.pth \
  --batch-size 2 \
  --num-epochs 2 \
  --learning-rate 1e-6
```

**Output**: `models/domain_model_best.pth`

**Time**: 1-2 hours

---

## ⚡ Stage 9: Model Optimization

**Prerequisites**:
- ✅ Any trained model

**What it does**: Reduce model size 4-8x, increase speed 2-3x

### Quantize Model

```bash
# INT8 quantization (4x smaller, 2x faster)
python3 scripts/quantize_model.py \
  --checkpoint models/multimodal_model_best.pth \
  --model-size large \
  --quantization int8
```

**Output**: `models/model_int8.pth`

**Time**: ~5-10 minutes

---

## 🔄 Stage 10: Continuous Learning

**Prerequisites**:
- ✅ Any trained model

**What it does**: Learn from new examples without forgetting

### Initialize Continual Learner

```bash
python3 scripts/continual_learning.py \
  --checkpoint models/multimodal_model_best.pth \
  --model-size large \
  --learning-rate 1e-6 \
  --device mps
```

**Ongoing** - integrates with your application to learn from user feedback

---

## 📊 Quick Progress Checklist

Track your progress:

- [ ] Stage 1: Language ✓ (Pre-done)
- [ ] Stage 2: Code ✓ (You just completed!)
- [ ] Stage 3: Tool Calling (Next)
  - [ ] Run `prepare_tool_calling_data.py`
  - [ ] Train tool calling
  - [ ] Test interactive inference
- [ ] Stage 4: RLHF
  - [ ] Prepare preferences
  - [ ] Train reward model
  - [ ] Run PPO
- [ ] Stage 5: Multi-Modal
  - [ ] Prepare image data
  - [ ] Train vision encoder
- [ ] Stage 6: RAG
  - [ ] Index codebase
  - [ ] Test search
- [ ] Stage 7: Agentic ✓ (No training needed)
- [ ] Stage 8: Domain (Optional)
- [ ] Stage 9: Optimization (Optional)
- [ ] Stage 10: Continuous (Optional)

---

## ⚠️ Common Issues

### "Data not found" error

If you see:
```
❌ ERROR: TOOL_CALLING DATA NOT FOUND!
```

**Fix**: Run the data preparation script first:
```bash
python3 scripts/prepare_tool_calling_data.py
```

This applies to all stages that need data preparation!

---

### Out of memory

**Fix**: Reduce batch size:
```bash
--batch-size 1  # Instead of 2
```

---

### Checkpoint not found

**Fix**: Make sure previous stage completed. Check:
```bash
ls -lh models/
```

You should see `*_model_best.pth` files for each completed stage.

---

## 🎯 Recommended Path

### Minimal (Basic Agent)
1. ✓ Stage 1: Language
2. ✓ Stage 2: Code
3. → Stage 3: Tool Calling
**Total**: ~14 hours

### Standard (Full Featured)
1-3 (above) +
4. Stage 4: RLHF
5. Stage 5: Multi-Modal
6. Stage 6: RAG
**Total**: ~22 hours

### Complete (Production Ready)
All stages 1-10
**Total**: ~26 hours

---

## 📚 Detailed Guides

- [Stage 3: Tool Calling](TOOL_CALLING_GUIDE.md)
- [Stage 4: RLHF](RLHF_GUIDE.md)
- [Stage 5: Multi-Modal](MULTIMODAL_GUIDE.md)
- [Stage 6: RAG](RAG_GUIDE.md)
- [Stage 7: Agentic](AGENT_GUIDE.md)
- [Stages 8-10: Advanced](ADVANCED_STAGES_GUIDE.md)

---

**Ready to continue? Run Stage 3!** 🚀

```bash
# Step 1: Generate data
python3 scripts/prepare_tool_calling_data.py

# Step 2: Train
python3 scripts/train.py --stage tool_calling \
  --checkpoint models/code_model_best.pth \
  --use-rmsnorm --use-rope --batch-size 2 \
  --num-epochs 3 --learning-rate 5e-6
```
