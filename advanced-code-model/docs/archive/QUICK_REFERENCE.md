# Quick Reference Card

**🚀 Essential commands and troubleshooting**

---

## ⚡ Quick Fix for Your Situation

**Problem:** Stage 2 val loss = 6.62 (way too high!)
**Cause:** Learning rate too high → Catastrophic forgetting

**Solution:**
```bash
python3 scripts/train.py \
  --stage code \
  --checkpoint models/language_model_best.pth \
  --batch-size 2 \
  --num-epochs 10 \
  --learning-rate 5e-6 \
  --warmup-steps 300 \
  --use-rmsnorm --use-rope --use-compile --use-amp
```

**Target:** Val loss ~2.0-2.5 (not 6.6!)
**Time:** ~4-5 hours

---

## 📊 Loss Benchmarks

| Stage | Good Loss | Your Loss | Status |
|-------|-----------|-----------|--------|
| Stage 1 (Language) | 3.0-3.5 | 3.44 | ✅ Perfect! |
| Stage 2 (Code) | 2.0-2.5 | 6.62 | ❌ Need to retrain |
| Stage 3 (Tools) | <2.0 | - | ⏸️ Waiting |

---

## 🎯 Critical Settings

### For Pretraining (Stage 1):
- Learning rate: **3e-5** ✓
- Batch size: **2+** ✓
- Can be aggressive

### For Fine-tuning (Stage 2+):
- Learning rate: **5e-6 or lower** ⚠️ CRITICAL!
- Batch size: **2+** ✓
- Must be gentle

### Why the difference?
- **Pretraining:** Starting from random → Can move fast
- **Fine-tuning:** Starting from trained model → Must be careful not to forget

---

## 🔍 Quick Diagnostics

### Check Stage Losses
```bash
# Stage 1
python3 -c "import json; h=json.load(open('models/language_training_history.json')); print(f'Stage 1: {h[-1][\"val_loss\"]:.2f}')"

# Stage 2
python3 -c "import json; h=json.load(open('models/code_training_history.json')); print(f'Stage 2: {h[-1][\"val_loss\"]:.2f}')"
```

### Check Model Files
```bash
ls -lh models/*.pth
```

### Test Stage 2 Quality
```bash
python3 scripts/test_stage2.py
```

---

## ⚠️ Common Issues

### Issue: Val loss increasing
**Cause:** Learning rate too high
**Fix:** Lower LR by 50% (1e-5 → 5e-6)

### Issue: Training very slow
**Cause:** Batch size too small
**Fix:** Increase batch size (1 → 2)

### Issue: Out of memory
**Cause:** Batch too large
**Fix:** Reduce batch size, increase grad accumulation

### Issue: Model generating gibberish
**Cause:** Val loss too high (>3.5)
**Fix:** Retrain with correct settings

---

## 📁 File Structure

```
models/
├── language_model_best.pth      # Stage 1 (safe, don't touch)
├── code_model_best.pth          # Stage 2 (will be replaced)
├── tool_calling_model_best.pth  # Stage 3 (future)
└── ...

data/processed/
├── language_train.npy           # Stage 1 data (ready)
├── code_train.npy               # Stage 2 data (ready)
├── tool_calling_train.npy       # Stage 3 data (need to generate)
└── ...
```

---

## 🚦 Training Progress Indicator

Monitor your training with these checkpoints:

**Epoch 1-2:** Loss should stay near Stage 1 (~3.5-4.0)
**Epoch 3-5:** Loss should decrease to ~3.0
**Epoch 6-8:** Loss should reach ~2.5
**Epoch 9-10:** Loss should stabilize at ~2.0-2.3

**If loss >5.0 after epoch 2:** STOP! Something is wrong.

---

## 📚 Documentation Links

- **[Step-by-Step Guide](STEP_BY_STEP_GUIDE.md)** - Sequential instructions
- **[Training Strategies](TRAINING_STRATEGIES.md)** - Visual explanations with diagrams
- **[Tool Calling Guide](TOOL_CALLING_GUIDE.md)** - Stage 3 details
- **[RLHF Guide](RLHF_GUIDE.md)** - Stage 4 details

---

## 🎓 Remember

**Staged training is like learning a language:**
1. First, learn grammar and basics (Stage 1: Language)
2. Then, learn specialized vocabulary (Stage 2: Code)
3. Then, practice using it (Stage 3: Tools)

**Not:**
"Learn everything at once" → Confusion and poor results

---

## ✅ Your Next Steps

1. ✅ Read [Training Strategies](TRAINING_STRATEGIES.md) (5 min)
2. ⏱️ Retrain Stage 2 with corrected command (4-5 hours)
3. ✅ Test model quality
4. ➡️ Proceed to Stage 3

**Good luck!** 🚀
