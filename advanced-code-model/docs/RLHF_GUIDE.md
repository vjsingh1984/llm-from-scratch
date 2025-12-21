## RLHF Guide (Stage 4)

**Complete 4-Stage Training Pipeline**

```
Stage 1: Language → Stage 2: Code → Stage 3: Tool Calling → Stage 4: RLHF
```

---

## What is RLHF?

RLHF (Reinforcement Learning from Human Feedback) aligns the model with human preferences:
1. **Collect preferences**: Human rankings of model outputs (chosen vs rejected)
2. **Train reward model**: Learn to predict human preferences
3. **PPO training**: Optimize policy to maximize reward

Used by ChatGPT, Claude, and all modern aligned LLMs!

---

## Stage 4 Training Pipeline

### Step 1: Prepare Preference Dataset

```bash
python3 scripts/prepare_rlhf_data.py
```

**What it does:**
- Generates 2,000 preference pairs (chosen vs rejected responses)
- Creates `rlhf_train_chosen.npy` and `rlhf_train_rejected.npy`
- Saves preference pairs to `rlhf_preferences.json`

**Example format:**
```
<|user|>Write a function to check if a number is prime<|end|>
<|assistant|>def is_prime(n: int) -> bool:
    """Check if a number is prime."""
    if n < 2:
        return False
    for i in range(2, int(n**0.5) + 1):
        if n % i == 0:
            return False
    return True<|end|>
<|preference|>chosen<|end|>
---
<|user|>Write a function to check if a number is prime<|end|>
<|assistant|>def is_prime(n):
    for i in range(2, n):
        if n % i == 0:
            return False
    return True<|end|>
<|preference|>rejected<|end|>
```

**Preference criteria:**
- Correctness (handles edge cases, no bugs)
- Readability (clear names, comments, structure)
- Efficiency (optimal complexity, space usage)
- Style (PEP8, formatting, pythonic)

---

### Step 2: Train Reward Model

**Prerequisites:**
- ✅ Stage 3 completed (`models/tool_calling_model_best.pth`)

**Train command:**
```bash
python3 scripts/train_reward_model.py \
  --checkpoint models/tool_calling_model_best.pth \
  --model-size large \
  --batch-size 4 \
  --num-epochs 3 \
  --learning-rate 1e-5 \
  --device mps
```

**What it does:**
- Loads Stage 3 model as base
- Adds reward head (hidden states → scalar reward)
- Trains to rank chosen > rejected responses
- Freezes base model, only trains reward head

**Expected:**
- **Time**: ~30-45 minutes
- **Memory**: 10-12GB
- **Target val accuracy**: >80%

**Architecture:**
```
Input [batch, seq_len]
    ↓
Base Model [batch, seq_len, d_model]
    ↓
Last Token Hidden State [batch, d_model]
    ↓
Reward Head (Linear → ReLU → Linear)
    ↓
Scalar Reward [batch]
```

**Loss function:**
```
Ranking Loss = max(0, margin - (reward_chosen - reward_rejected))
```

---

### Step 3: PPO Training

**Prerequisites:**
- ✅ Stage 3 completed (`models/tool_calling_model_best.pth`)
- ✅ Reward model trained (`models/reward_model_best.pth`)

**Train command:**
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

**What it does:**
- Loads policy model (Stage 3) and reward model
- Generates responses from current policy
- Computes rewards using reward model
- Updates policy using PPO objective
- Repeats until convergence

**Expected:**
- **Time**: ~1-2 hours
- **Memory**: 12-14GB
- **Target mean reward**: Increasing over epochs

**PPO Algorithm:**
```
For each step:
  1. Generate responses from policy: r ~ π(a|s)
  2. Compute rewards: R = reward_model(s, r)
  3. Compute advantages: A = R - V(s)
  4. Update policy: maximize E[min(ratio * A, clip(ratio) * A)]
```

**PPO Clipped Objective:**
```
ratio = π_new(a|s) / π_old(a|s)
L = -min(ratio * A, clip(ratio, 1-ε, 1+ε) * A)
```

---

## Complete 4-Stage Pipeline

```bash
# Stage 1: Language Pretraining
python3 scripts/train.py \
  --stage language \
  --architecture dense \
  --model-size large \
  --batch-size 2 \
  --gradient-accumulation-steps 4 \
  --use-compile --use-rmsnorm --use-rope --use-amp --use-gradient-checkpointing \
  --num-epochs 3 --steps-per-epoch 1000 --learning-rate 3e-5 --warmup-steps 300

# Stage 2: Code Fine-tuning
python3 scripts/train.py \
  --stage code \
  --architecture dense \
  --model-size large \
  --checkpoint models/language_model_best.pth \
  --batch-size 1 \
  --gradient-accumulation-steps 4 \
  --use-compile --use-rmsnorm --use-rope --use-amp --use-gradient-checkpointing \
  --num-epochs 5 --steps-per-epoch 400 --learning-rate 1e-5 --warmup-steps 150

# Stage 3: Tool Calling
python3 scripts/prepare_tool_calling_data.py

python3 scripts/train.py \
  --stage tool_calling \
  --architecture dense \
  --model-size large \
  --checkpoint models/code_model_best.pth \
  --batch-size 2 \
  --gradient-accumulation-steps 4 \
  --use-compile --use-rmsnorm --use-rope --use-amp --use-gradient-checkpointing \
  --num-epochs 3 --steps-per-epoch 300 --learning-rate 5e-6 --warmup-steps 100

# Stage 4a: Prepare RLHF Data
python3 scripts/prepare_rlhf_data.py

# Stage 4b: Train Reward Model
python3 scripts/train_reward_model.py \
  --checkpoint models/tool_calling_model_best.pth \
  --model-size large \
  --batch-size 4 \
  --num-epochs 3 \
  --learning-rate 1e-5 \
  --device mps

# Stage 4c: PPO Training
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

---

## Training Timeline

| Stage | Time | Memory | Dataset | Output |
|-------|------|--------|---------|--------|
| **Stage 1** | 9-11h | 16-18GB | 100M tokens | language_model_best.pth |
| **Stage 2** | 2-3h | 10-12GB | 7M tokens | code_model_best.pth |
| **Stage 3** | 1-2h | 10-12GB | 2K examples | tool_calling_model_best.pth |
| **Stage 4a** | 30-45m | 10-12GB | 2K pairs | reward_model_best.pth |
| **Stage 4b** | 1-2h | 12-14GB | 100 steps | rlhf_model_best.pth |
| **Total** | **14-19h** | **18GB peak** | - | Fully aligned coding agent |

---

## Why RLHF?

**Before RLHF:**
- Model outputs technically correct but may lack quality
- No preference for readable, efficient, or safe code
- May produce verbose or convoluted solutions

**After RLHF:**
- Model learns human preferences
- Prefers clean, efficient, well-documented code
- Avoids insecure patterns (eval, exec)
- Better instruction following

**Example:**

**Prompt:** "Write a function to check if a number is prime"

**Before RLHF (Stage 3):**
```python
def is_prime(n):
    for i in range(2, n):
        if n % i == 0:
            return False
    return True
```
❌ Inefficient (O(n) instead of O(√n))
❌ No edge cases (n < 2)
❌ No documentation

**After RLHF (Stage 4):**
```python
def is_prime(n: int) -> bool:
    """Check if a number is prime.

    Args:
        n: Number to check

    Returns:
        True if prime, False otherwise
    """
    if n < 2:
        return False
    if n == 2:
        return True
    if n % 2 == 0:
        return False

    # Check odd divisors up to sqrt(n)
    for i in range(3, int(n**0.5) + 1, 2):
        if n % i == 0:
            return False
    return True
```
✅ Efficient O(√n)
✅ Handles edge cases
✅ Type hints and documentation
✅ Clear comments

---

## Preference Dataset Quality

**Good preference pairs:**
- Clear quality difference between chosen and rejected
- Objective criteria (correctness, efficiency)
- Diverse prompts and responses

**Our synthetic dataset includes:**
- Correctness: Handles edge cases vs missing them
- Efficiency: Optimal vs naive algorithms
- Readability: Documented vs undocumented
- Style: Clean vs messy formatting

**To improve quality:**
1. Collect real human preferences (manual annotation)
2. Use multiple annotators per pair
3. Include rationale for preferences
4. Diverse prompt distribution

---

## Reward Model Insights

**What the reward model learns:**
- High rewards for: Type hints, documentation, edge cases, efficiency
- Low rewards for: Security issues, bugs, poor naming, inefficiency

**Reward model architecture:**
```
Base LLM (frozen)
    ↓
Hidden states [seq_len, d_model]
    ↓
Last token [d_model]
    ↓
Linear(d_model, d_model/2) + ReLU
    ↓
Linear(d_model/2, 1)
    ↓
Scalar reward
```

**Why freeze base model?**
- Preserves language and code understanding
- Only learns preference function
- Prevents overfitting
- Faster training

---

## PPO Hyperparameters

**Key hyperparameters:**
- `epsilon (ε)`: Clipping range (default: 0.2)
  - Too small: Conservative updates, slow learning
  - Too large: Unstable updates

- `learning_rate`: Policy update step size (default: 1e-6)
  - Much lower than supervised learning
  - Prevents policy collapse

- `gamma (γ)`: Reward discount (default: 0.99)
  - Future reward importance

- `lambda (λ)`: GAE parameter (default: 0.95)
  - Bias-variance tradeoff for advantages

**Typical ranges:**
- ε: 0.1 - 0.3
- learning_rate: 1e-7 - 1e-5
- gamma: 0.95 - 0.99
- lambda: 0.9 - 0.98

---

## Troubleshooting

**Q: Reward model accuracy stuck at 50%**
- A: Increase training epochs or improve preference pair quality

**Q: PPO training unstable (rewards decreasing)**
- A: Lower learning rate or decrease epsilon clipping

**Q: Policy collapse (model generates nonsense)**
- A: Add KL penalty to keep policy close to original

**Q: Out of memory during PPO**
- A: Reduce batch size or max_new_tokens

**Q: Rewards not improving**
- A: Check reward model quality, increase PPO steps

---

## Advanced: KL Penalty

To prevent policy from deviating too much from original:

```python
# Compute KL divergence between new and old policy
kl_div = F.kl_div(
    F.log_softmax(new_logits, dim=-1),
    F.softmax(old_logits, dim=-1),
    reduction='batchmean'
)

# Add to loss
total_loss = ppo_loss + beta * kl_div  # beta = 0.01 - 0.1
```

**Benefits:**
- Prevents policy collapse
- Maintains language capabilities
- More stable training

---

## Next Steps

1. ✅ Train Stage 4 RLHF
2. 🔬 Test model alignment (preference for quality code)
3. 🎨 Collect real human preferences for production
4. 📈 Scale to larger models (XLarge)
5. 🚀 Move to Stage 5: Multi-Modal

**You now have a complete aligned coding agent with RLHF!** 🎉

Similar capabilities to ChatGPT and Claude, but:
- 100% local
- Fully customizable
- Zero cost
- Private data
- Transparent alignment

---

**Ready for multi-modal code understanding!** 🚀
