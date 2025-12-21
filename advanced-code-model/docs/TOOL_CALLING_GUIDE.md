## Tool Calling Guide (Stage 3)

**Complete 3-Stage Training Pipeline**

```
Stage 1: Language → Stage 2: Code → Stage 3: Tool Calling
```

---

## What is Tool Calling?

Tool calling enables the model to:
1. **Recognize when to use tools** (execute code, search docs, analyze, debug)
2. **Generate correct tool arguments** (JSON format)
3. **Use tool results** to provide better responses

Similar to ChatGPT Code Interpreter, Claude Code, and GPT-4 with function calling!

---

## Stage 3 Training Pipeline

### Step 1: Prepare Tool Calling Dataset

```bash
python3 scripts/prepare_tool_calling_data.py
```

**What it does:**
- Generates 2,000 synthetic tool-calling examples
- Creates `tool_calling_train.npy` and `tool_calling_val.npy`
- Saves tool definitions to `tool_definitions.json`

**Example format:**
```
<|user|>Write a factorial function and test it<|end|>
<|assistant|><|tool_call|>
Tool: execute_python
Arguments: {"code": "def factorial(n): ..."}
<|end|>
<|tool_result|>
Tool: execute_python
Output: 120
<|end|>
<|assistant|>Here's a factorial function... Result: 120 ✓<|end|>
```

---

### Step 2: Train Stage 3

**Prerequisites:**
- ✅ Stage 2 completed (`models/code_model_best.pth`)

**Train command:**
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

**Expected:**
- **Time**: ~1-2 hours
- **Memory**: 10-12GB
- **Dataset**: 1,800 train, 200 val examples
- **Target val loss**: <2.0

---

### Step 3: Inference with Tools

**Interactive mode:**
```bash
python3 scripts/inference_tool_calling.py \
  --checkpoint models/tool_calling_model_best.pth \
  --model-size large \
  --device mps
```

**Example session:**
```
You: Write a function to reverse a string and test it

🔧 Tool Call #1
  Tool: execute_python
  Arguments: {
    "code": "def reverse_string(s):\n    return s[::-1]\n\nassert reverse_string('hello') == 'olleh'\nprint('Test passed!')"
  }

📤 Tool Result:
  Test passed!

Assistant Response:
Here's a string reversal function:

def reverse_string(s):
    return s[::-1]

Tested successfully ✓
```

---

## Available Tools

### 1. **execute_python**
Execute Python code and return output
```json
{"code": "print(2 + 2)"}
```

### 2. **execute_bash**
Run bash commands
```json
{"command": "ls -la"}
```

### 3. **search_docs**
Search Python documentation
```json
{"query": "list comprehensions"}
```

### 4. **analyze_code**
Analyze code for issues
```json
{
  "code": "eval(input())",
  "analysis_type": "security"  // syntax, security, complexity
}
```

### 5. **run_tests**
Run unit tests
```json
{
  "code": "def add(a, b): return a + b\nassert add(2, 3) == 5",
  "test_framework": "pytest"
}
```

### 6. **debug_code**
Debug code and suggest fixes
```json
{
  "code": "my_list[10]",
  "error": "IndexError: list index out of range"
}
```

---

## Complete 3-Stage Pipeline

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
# First, prepare data
python3 scripts/prepare_tool_calling_data.py

# Then train
python3 scripts/train.py \
  --stage tool_calling \
  --architecture dense \
  --model-size large \
  --checkpoint models/code_model_best.pth \
  --batch-size 2 \
  --gradient-accumulation-steps 4 \
  --use-compile --use-rmsnorm --use-rope --use-amp --use-gradient-checkpointing \
  --num-epochs 3 --steps-per-epoch 300 --learning-rate 5e-6 --warmup-steps 100

# Inference
python3 scripts/inference_tool_calling.py \
  --checkpoint models/tool_calling_model_best.pth \
  --model-size large
```

---

## Safety Features

**Safe Mode (Default)**:
- ✅ Asks for confirmation before executing code
- ✅ 5-second timeout for all executions
- ✅ Runs in subprocess (isolated)
- ✅ No access to system files

**Example:**
```
PYTHON CODE TO EXECUTE:
import os; os.system('rm -rf /')

Execute this code? (yes/no): no
> Execution cancelled by user
```

---

## Use Cases

### 1. **Code Generation + Testing**
```
You: Create a binary search function

🔧 Generates code
🔧 Executes tests
✓ Returns working, tested code
```

### 2. **Interactive Debugging**
```
You: Fix this error: IndexError in my_list[5]

🔧 Analyzes code
🔧 Suggests fix
🔧 Tests fix
✓ Returns corrected code
```

### 3. **Documentation Search**
```
You: How do decorators work?

🔧 Searches docs
🔧 Finds examples
✓ Returns explanation with examples
```

### 4. **Security Analysis**
```
You: Is this code safe? eval(user_input)

🔧 Analyzes for security issues
⚠️ Detects eval() risk
✓ Suggests safe alternative
```

---

## Training Timeline

| Stage | Time | Memory | Dataset | Output |
|-------|------|--------|---------|--------|
| **Stage 1** | 9-11h | 16-18GB | 100M tokens | language_model_best.pth |
| **Stage 2** | 2-3h | 10-12GB | 7M tokens | code_model_best.pth |
| **Stage 3** | 1-2h | 10-12GB | 2K examples | tool_calling_model_best.pth |
| **Total** | **12-16h** | **18GB peak** | - | Fully capable coding agent |

---

## Advanced: Custom Tools

**Add your own tools:**

```python
# In inference_tool_calling.py, add to ToolExecutor class:

def custom_tool(self, arg1: str, arg2: int) -> str:
    """Your custom tool logic."""
    result = f"Processed {arg1} with {arg2}"
    return result

# In execute_tool method:
elif tool_name == "custom_tool":
    return self.custom_tool(
        arguments.get("arg1", ""),
        arguments.get("arg2", 0)
    )
```

**Update training data:**
```python
# In prepare_tool_calling_data.py, add to CODING_TOOLS:

"custom_tool": {
    "description": "Your tool description",
    "parameters": {
        "arg1": "string - Description",
        "arg2": "integer - Description"
    }
}
```

---

## Comparison to Other Systems

| Feature | Your Model | ChatGPT Code | Claude Code | GPT-4 Functions |
|---------|-----------|--------------|-------------|-----------------|
| **Code Execution** | ✅ | ✅ | ✅ | ✅ |
| **Bash Commands** | ✅ | ✅ | ✅ | ❌ |
| **Doc Search** | ✅ | ❌ | ✅ | ✅ |
| **Security Analysis** | ✅ | ❌ | ✅ | ❌ |
| **Offline** | ✅ | ❌ | ❌ | ❌ |
| **Customizable** | ✅ | ❌ | ❌ | ✅ |
| **Cost** | **$0** | $$$ | $$$ | $$$$ |

---

## Troubleshooting

**Q: Model generates malformed JSON**
- A: Increase training epochs or use lower temperature

**Q: Tool calls timeout**
- A: Increase timeout in `ToolExecutor` (default: 5s)

**Q: Safe mode too restrictive**
- A: Set `safe_mode=False` in `ToolExecutor` (use with caution!)

**Q: Want more tool calling examples**
- A: Increase `num_examples` in `prepare_tool_calling_data.py`

---

## Next Steps

1. ✅ Train Stage 3
2. 🔬 Test with real coding tasks
3. 🎨 Add custom tools for your use case
4. 📈 Scale to larger models (XLarge)
5. 🚀 Deploy as coding assistant

**You now have a complete coding agent with tool calling!** 🎉

Similar capabilities to ChatGPT Code Interpreter and Claude Code, but:
- 100% local
- Fully customizable
- Zero cost
- Private data

---

**Ready to build the future of AI coding assistants!** 🚀
