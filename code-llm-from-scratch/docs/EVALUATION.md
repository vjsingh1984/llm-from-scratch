# Evaluation and Benchmarking Guide

Learn how to measure model quality and improve performance through systematic evaluation.

## Table of Contents

1. [Foundational Metrics](#1-foundational-metrics)
2. [Automated Evaluation](#2-automated-evaluation)
3. [Human Evaluation](#3-human-evaluation)
4. [Comparative Analysis](#4-comparative-analysis)
5. [Advanced Metrics](#5-advanced-metrics)
6. [Debugging Poor Performance](#6-debugging-poor-performance)
7. [Continuous Improvement](#7-continuous-improvement)

---

## 1. Foundational Metrics

Start with these basic metrics to understand model behavior.

### 1.1 Training Loss

**What it measures**: How well the model predicts the next token during training.

**How to track**:

```python
# Already logged during training
# Epoch 1/10: loss=3.8
# Epoch 10/10: loss=2.3
```

**Interpretation**:
- **High loss (>4.0)**: Model hasn't learned much
- **Medium loss (2.0-3.0)**: Decent learning, typical for language stage
- **Low loss (1.0-2.0)**: Good learning, typical for code stage
- **Very low loss (<0.5)**: Possible overfitting

**Example visualization**:

```python
import matplotlib.pyplot as plt
import json

# Load training history
with open('models/code/training_history.json') as f:
    history = json.load(f)

plt.figure(figsize=(10, 5))
plt.plot(history['train_losses'], label='Training Loss')
if 'val_losses' in history:
    plt.plot(history['val_losses'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Progress')
plt.legend()
plt.grid(True)
plt.savefig('training_curve.png')
```

### 1.2 Perplexity

**What it measures**: How "surprised" the model is by the test data (lower is better).

**Formula**: `perplexity = exp(loss)`

**Calculate**:

```python
import math

train_loss = 2.3
perplexity = math.exp(train_loss)
print(f"Perplexity: {perplexity:.2f}")
# Perplexity: 9.97
```

**Interpretation**:
- **Perplexity < 10**: Excellent
- **Perplexity 10-50**: Good
- **Perplexity 50-100**: Acceptable
- **Perplexity > 100**: Needs improvement

### 1.3 Generation Length

**What it measures**: How much code the model generates.

**Track**:

```python
def measure_generation_length(model, tokenizer, prompts, device):
    """Measure average generation length."""
    lengths = []

    for prompt in prompts:
        input_ids = tokenizer.encode(prompt)
        input_tensor = torch.tensor([input_ids], device=device)

        generated = generate_tokens(input_tensor, model, max_length=200)
        new_tokens = generated.shape[1] - input_tensor.shape[1]
        lengths.append(new_tokens)

    return {
        'mean': sum(lengths) / len(lengths),
        'min': min(lengths),
        'max': max(lengths),
        'lengths': lengths
    }

# Test
test_prompts = [
    "#!/bin/bash\n# Create a backup",
    "#!/bin/bash\n# Monitor system",
    "#!/bin/bash\n# Deploy application"
]

stats = measure_generation_length(model, tokenizer, test_prompts, device)
print(f"Average length: {stats['mean']:.1f} tokens")
```

**Interpretation**:
- Too short (<20 tokens): Model may be stopping early
- Good range (50-150 tokens): Generates complete scripts
- Too long (>200 tokens): May be repeating or rambling

---

## 2. Automated Evaluation

Automate quality checks without manual inspection.

### 2.1 Syntactic Correctness

**What it measures**: Whether generated code is syntactically valid.

**Implementation**:

```python
import subprocess
import tempfile
import os

def check_bash_syntax(code: str) -> bool:
    """
    Check if bash code is syntactically valid.

    Returns:
        True if syntax is valid, False otherwise
    """
    try:
        # Use bash -n to check syntax without executing
        result = subprocess.run(
            ['bash', '-n'],
            input=code.encode(),
            capture_output=True,
            timeout=5
        )
        return result.returncode == 0
    except Exception as e:
        return False


def evaluate_syntax_correctness(model, tokenizer, prompts, device):
    """
    Evaluate what percentage of generated scripts are syntactically valid.
    """
    valid_count = 0
    total_count = len(prompts)
    results = []

    for prompt in prompts:
        # Generate code
        code = generate_code(model, tokenizer, prompt, device)

        # Check syntax
        is_valid = check_bash_syntax(code)
        valid_count += is_valid

        results.append({
            'prompt': prompt,
            'code': code,
            'valid': is_valid
        })

    accuracy = valid_count / total_count * 100

    return {
        'syntax_accuracy': accuracy,
        'valid_count': valid_count,
        'total_count': total_count,
        'results': results
    }


# Example usage
test_prompts = [
    "#!/bin/bash\n# Create a backup script",
    "#!/bin/bash\n# Monitor disk space",
    "#!/bin/bash\n# Deploy to production",
    # Add 20+ more prompts for robust evaluation
]

eval_results = evaluate_syntax_correctness(model, tokenizer, test_prompts, device)
print(f"Syntax Correctness: {eval_results['syntax_accuracy']:.1f}%")
```

**Target**: **>85%** for production use

### 2.2 Keyword/Pattern Matching

**What it measures**: Whether generated code contains expected patterns.

```python
def check_bash_patterns(code: str, prompt: str) -> dict:
    """Check if code contains expected bash patterns."""
    checks = {
        'has_shebang': code.startswith('#!/bin/bash'),
        'has_comments': '#' in code,
        'has_variables': '$' in code or '${' in code,
        'has_conditionals': any(kw in code for kw in ['if', 'then', 'fi']),
        'has_loops': any(kw in code for kw in ['for', 'while', 'do', 'done']),
        'has_functions': 'function' in code or '() {' in code,
        'has_error_handling': any(kw in code for kw in ['set -e', 'trap', 'exit']),
    }

    # Check prompt-specific expectations
    if 'backup' in prompt.lower():
        checks['backup_relevant'] = any(kw in code for kw in ['tar', 'rsync', 'cp', 'backup'])
    if 'monitor' in prompt.lower():
        checks['monitor_relevant'] = any(kw in code for kw in ['top', 'df', 'free', 'ps'])
    if 'deploy' in prompt.lower():
        checks['deploy_relevant'] = any(kw in code for kw in ['git', 'docker', 'systemctl', 'service'])

    return checks


def evaluate_pattern_matching(model, tokenizer, prompts, device):
    """Evaluate pattern matching across prompts."""
    all_results = []

    for prompt in prompts:
        code = generate_code(model, tokenizer, prompt, device)
        patterns = check_bash_patterns(code, prompt)
        all_results.append(patterns)

    # Aggregate statistics
    stats = {}
    for key in all_results[0].keys():
        stats[key] = sum(1 for r in all_results if r[key]) / len(all_results) * 100

    return stats


# Example
pattern_stats = evaluate_pattern_matching(model, tokenizer, test_prompts, device)
for pattern, percentage in pattern_stats.items():
    print(f"{pattern}: {percentage:.1f}%")
```

### 2.3 Code Quality Metrics

**Use shellcheck for bash code quality**:

```python
def check_code_quality(code: str) -> dict:
    """
    Check code quality using shellcheck.

    Install: apt-get install shellcheck
    """
    try:
        # Write code to temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.sh', delete=False) as f:
            f.write(code)
            temp_path = f.name

        # Run shellcheck
        result = subprocess.run(
            ['shellcheck', '-f', 'json', temp_path],
            capture_output=True,
            text=True
        )

        os.unlink(temp_path)

        # Parse results
        if result.returncode == 0:
            return {'quality_score': 100, 'issues': []}
        else:
            import json
            issues = json.loads(result.stdout)
            # Deduct points based on issue severity
            score = 100
            for issue in issues:
                if issue['level'] == 'error':
                    score -= 10
                elif issue['level'] == 'warning':
                    score -= 5
                elif issue['level'] == 'info':
                    score -= 2
            return {'quality_score': max(0, score), 'issues': issues}

    except Exception as e:
        return {'quality_score': 0, 'issues': [str(e)]}
```

---

## 3. Human Evaluation

For nuanced quality assessment, human evaluation is essential.

### 3.1 Rating Scale

Create a standardized rating rubric:

```python
"""
Evaluation Rubric (1-5 scale):

1. Syntax (Does it run?)
   1 = Syntax errors
   2 = Runs but with warnings
   3 = Clean syntax
   4 = Clean with good practices
   5 = Perfect syntax + best practices

2. Correctness (Does it do what was asked?)
   1 = Completely wrong
   2 = Partially addresses task
   3 = Mostly correct
   4 = Fully correct
   5 = Exceeds expectations

3. Code Quality (Is it well-written?)
   1 = Poor style, no structure
   2 = Basic structure
   3 = Good structure and naming
   4 = Excellent practices
   5 = Production-ready

4. Completeness (Is it a complete solution?)
   1 = Fragment only
   2 = Basic outline
   3 = Working solution
   4 = Complete with error handling
   5 = Complete with documentation

5. Bash Idioms (Does it use bash properly?)
   1 = Non-idiomatic
   2 = Some bash features
   3 = Good use of bash
   4 = Excellent bash patterns
   5 = Expert-level bash
"""

def human_evaluation_template():
    """Template for human evaluation."""
    return {
        'prompt': '',
        'generated_code': '',
        'ratings': {
            'syntax': 0,  # 1-5
            'correctness': 0,  # 1-5
            'quality': 0,  # 1-5
            'completeness': 0,  # 1-5
            'bash_idioms': 0,  # 1-5
        },
        'comments': '',
        'would_use_in_production': False,  # yes/no
    }
```

### 3.2 Comparative Evaluation

Compare your model against baselines:

```python
"""
A/B Testing Format:

Given prompt: "Create a backup script"

Version A (Your Model):
[Generated code A]

Version B (Baseline/GPT-4/etc):
[Generated code B]

Questions:
1. Which version would you prefer to use? [A/B/Tie]
2. Which is more correct? [A/B/Tie]
3. Which has better code quality? [A/B/Tie]
4. Which is more complete? [A/B/Tie]
"""

def create_ab_test(prompt, code_a, code_b):
    """Create an A/B test comparison."""
    return {
        'prompt': prompt,
        'code_a': code_a,
        'code_b': code_b,
        'preference': None,  # 'A', 'B', or 'Tie'
        'correctness_winner': None,
        'quality_winner': None,
        'completeness_winner': None,
        'comments': ''
    }
```

---

## 4. Comparative Analysis

Compare your model's performance over time or against benchmarks.

### 4.1 Model Comparison Script

```python
import json
from datetime import datetime

def compare_models(models_config, test_prompts, device):
    """
    Compare multiple model checkpoints.

    Args:
        models_config: List of {'name': str, 'path': str, 'tokenizer': str}
        test_prompts: List of test prompts
        device: torch device
    """
    results = []

    for config in models_config:
        print(f"\nEvaluating {config['name']}...")

        # Load model and tokenizer
        model, tokenizer = load_model_and_tokenizer(
            config['path'],
            config['tokenizer'],
            device
        )

        # Evaluate
        syntax_results = evaluate_syntax_correctness(
            model, tokenizer, test_prompts, device
        )
        pattern_results = evaluate_pattern_matching(
            model, tokenizer, test_prompts, device
        )

        results.append({
            'model': config['name'],
            'syntax_accuracy': syntax_results['syntax_accuracy'],
            'pattern_stats': pattern_results,
            'timestamp': datetime.now().isoformat()
        })

    # Create comparison report
    print("\n" + "="*60)
    print("MODEL COMPARISON")
    print("="*60)

    for result in results:
        print(f"\n{result['model']}:")
        print(f"  Syntax Accuracy: {result['syntax_accuracy']:.1f}%")
        print(f"  Has Shebang: {result['pattern_stats'].get('has_shebang', 0):.1f}%")
        print(f"  Has Error Handling: {result['pattern_stats'].get('has_error_handling', 0):.1f}%")

    # Save results
    with open('evaluation_results.json', 'w') as f:
        json.dump(results, f, indent=2)

    return results


# Example usage
models_to_compare = [
    {
        'name': 'Baseline (after language training)',
        'path': 'models/language/language_model_final.pt',
        'tokenizer': 'models/language/language_tokenizer.json'
    },
    {
        'name': 'After code fine-tuning (epoch 10)',
        'path': 'models/code/checkpoint_epoch_10.pt',
        'tokenizer': 'models/language/language_tokenizer.json'
    },
    {
        'name': 'Final model (epoch 20)',
        'path': 'models/code/code_model_final.pt',
        'tokenizer': 'models/language/language_tokenizer.json'
    },
]

comparison_results = compare_models(models_to_compare, test_prompts, device)
```

### 4.2 Track Metrics Over Time

```python
def log_evaluation_metrics(model_name, metrics, log_file='eval_log.jsonl'):
    """Log evaluation metrics with timestamp."""
    entry = {
        'timestamp': datetime.now().isoformat(),
        'model': model_name,
        'metrics': metrics
    }

    with open(log_file, 'a') as f:
        f.write(json.dumps(entry) + '\n')


def plot_metrics_over_time(log_file='eval_log.jsonl'):
    """Plot how metrics evolved over time."""
    import pandas as pd
    import matplotlib.pyplot as plt

    # Load logs
    entries = []
    with open(log_file) as f:
        for line in f:
            entries.append(json.loads(line))

    df = pd.DataFrame(entries)
    df['timestamp'] = pd.to_datetime(df['timestamp'])

    # Extract metrics
    df['syntax_accuracy'] = df['metrics'].apply(lambda x: x.get('syntax_accuracy', 0))

    # Plot
    plt.figure(figsize=(12, 6))
    plt.plot(df['timestamp'], df['syntax_accuracy'], marker='o')
    plt.xlabel('Date')
    plt.ylabel('Syntax Accuracy (%)')
    plt.title('Model Quality Over Time')
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('quality_over_time.png')
```

---

## 5. Advanced Metrics

### 5.1 BLEU Score (for reference comparison)

```python
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

def calculate_bleu(reference_code, generated_code):
    """
    Calculate BLEU score between reference and generated code.

    Note: BLEU is not perfect for code, but gives a rough similarity measure.
    """
    # Tokenize by whitespace
    reference_tokens = [reference_code.split()]
    generated_tokens = generated_code.split()

    # Calculate BLEU with smoothing
    smoothing = SmoothingFunction().method1
    score = sentence_bleu(
        reference_tokens,
        generated_tokens,
        smoothing_function=smoothing
    )

    return score * 100  # Convert to percentage


# Example
reference = """#!/bin/bash
tar -czf backup.tar.gz /data
echo "Backup complete"
"""

generated = """#!/bin/bash
tar -czf backup.tar.gz /data
if [ $? -eq 0 ]; then
    echo "Backup completed successfully"
fi
"""

bleu_score = calculate_bleu(reference, generated)
print(f"BLEU Score: {bleu_score:.2f}")
```

### 5.2 Code Similarity (AST-based)

```python
import ast
import difflib

def calculate_code_similarity(code1, code2):
    """Calculate similarity using diff-based approach."""
    similarity = difflib.SequenceMatcher(None, code1, code2).ratio()
    return similarity * 100


def calculate_structural_similarity(code1, code2):
    """Calculate similarity based on code structure (lines, tokens)."""
    lines1 = [l.strip() for l in code1.split('\n') if l.strip()]
    lines2 = [l.strip() for l in code2.split('\n') if l.strip()]

    # Line-based similarity
    matcher = difflib.SequenceMatcher(None, lines1, lines2)
    return matcher.ratio() * 100
```

### 5.3 Diversity Metrics

```python
def calculate_diversity(generated_codes):
    """
    Measure diversity in generated code.

    Higher diversity = model generates varied solutions
    Lower diversity = model repeats same patterns
    """
    # Calculate pairwise similarities
    similarities = []
    for i in range(len(generated_codes)):
        for j in range(i+1, len(generated_codes)):
            sim = calculate_code_similarity(generated_codes[i], generated_codes[j])
            similarities.append(sim)

    avg_similarity = sum(similarities) / len(similarities) if similarities else 0
    diversity_score = 100 - avg_similarity

    return {
        'diversity_score': diversity_score,
        'avg_similarity': avg_similarity,
        'num_comparisons': len(similarities)
    }


# Test: Generate same prompt multiple times
prompt = "#!/bin/bash\n# Create a backup script"
generations = []

for _ in range(5):
    code = generate_code(model, tokenizer, prompt, device, temperature=0.8)
    generations.append(code)

diversity_stats = calculate_diversity(generations)
print(f"Diversity Score: {diversity_stats['diversity_score']:.1f}")
```

---

## 6. Debugging Poor Performance

When model quality is low, systematically debug:

### 6.1 Diagnostic Checklist

```
□ Check training loss curve
  - Is loss decreasing?
  - Is there a plateau?
  - Signs of overfitting (train ↓, val ↑)?

□ Inspect generated samples
  - Are they syntactically valid?
  - Do they repeat phrases?
  - Are they too short/long?
  - Do they match the prompt?

□ Verify data quality
  - Is training data clean?
  - Enough diversity?
  - Correct format?

□ Check hyperparameters
  - Learning rate too high/low?
  - Batch size appropriate?
  - Enough training steps?

□ Validate tokenizer
  - Vocabulary size reasonable?
  - Can it encode/decode correctly?
  - Special tokens handled?

□ Model architecture
  - Model size appropriate?
  - Enough capacity?
  - Too much regularization?
```

### 6.2 Common Issues and Fixes

| Issue | Symptoms | Fix |
|-------|----------|-----|
| **Repetition** | Model repeats same phrases | • Increase temperature<br>• Use top-p sampling<br>• Add repetition penalty |
| **Gibberish** | Invalid syntax, random characters | • Train longer<br>• Check tokenizer<br>• Verify data quality |
| **Too Generic** | Always generates same code | • Increase model size<br>• More diverse training data<br>• Higher temperature |
| **Off-Topic** | Doesn't follow prompt | • More code fine-tuning epochs<br>• Better prompt engineering<br>• Check if language model is loaded |
| **Too Short** | Generates 1-2 lines only | • Adjust stopping criteria<br>• Train on longer examples<br>• Increase max_length |

### 6.3 Debug Script

```python
def debug_model(model, tokenizer, device):
    """Run diagnostic tests on model."""
    print("="*60)
    print("MODEL DIAGNOSTICS")
    print("="*60)

    # Test 1: Can it generate anything?
    print("\n1. Basic Generation Test")
    try:
        code = generate_code(model, tokenizer, "#!/bin/bash", device, max_length=50)
        print(f"✓ Generated {len(code)} characters")
        print(f"Preview: {code[:100]}...")
    except Exception as e:
        print(f"✗ Failed: {e}")

    # Test 2: Syntax check
    print("\n2. Syntax Validity Test")
    is_valid = check_bash_syntax(code)
    print(f"{'✓' if is_valid else '✗'} Syntax {'valid' if is_valid else 'invalid'}")

    # Test 3: Diversity test
    print("\n3. Diversity Test (same prompt, 3 generations)")
    generations = [generate_code(model, tokenizer, "#!/bin/bash\n# test", device, temperature=0.8)
                   for _ in range(3)]
    unique = len(set(generations))
    print(f"{'✓' if unique > 1 else '✗'} {unique}/3 unique generations")

    # Test 4: Prompt following
    print("\n4. Prompt Following Test")
    test_cases = [
        ("#!/bin/bash\n# backup", "backup"),
        ("#!/bin/bash\n# monitor", "monitor"),
        ("#!/bin/bash\n# deploy", "deploy")
    ]

    for prompt, keyword in test_cases:
        gen = generate_code(model, tokenizer, prompt, device)
        contains = keyword in gen.lower()
        print(f"{'✓' if contains else '✗'} '{keyword}' prompt → {'contains' if contains else 'missing'} keyword")

    print("\n" + "="*60)
```

---

## 7. Continuous Improvement

### 7.1 Evaluation Pipeline

Create an automated evaluation pipeline:

```bash
#!/bin/bash
# evaluate_model.sh - Run comprehensive evaluation

echo "Running Model Evaluation Pipeline..."

# 1. Syntax evaluation
python scripts/evaluate_syntax.py \
    --model models/code/code_model_final.pt \
    --test-prompts data/test_prompts.json \
    --output results/syntax_eval.json

# 2. Pattern matching
python scripts/evaluate_patterns.py \
    --model models/code/code_model_final.pt \
    --test-prompts data/test_prompts.json \
    --output results/pattern_eval.json

# 3. Generate report
python scripts/generate_report.py \
    --syntax-results results/syntax_eval.json \
    --pattern-results results/pattern_eval.json \
    --output results/evaluation_report.html

echo "Evaluation complete! See results/evaluation_report.html"
```

### 7.2 Benchmark Suite

Create a standard benchmark:

```python
# Create test_prompts.json with diverse prompts
benchmark_prompts = {
    "backup": [
        "#!/bin/bash\n# Create a backup of /data",
        "#!/bin/bash\n# Incremental backup script",
        "#!/bin/bash\n# Database backup with rotation"
    ],
    "monitoring": [
        "#!/bin/bash\n# Monitor CPU and memory",
        "#!/bin/bash\n# Disk space alert script",
        "#!/bin/bash\n# Log system metrics"
    ],
    "deployment": [
        "#!/bin/bash\n# Deploy web application",
        "#!/bin/bash\n# Rolling deployment script",
        "#!/bin/bash\n# Blue-green deployment"
    ],
    # Add more categories...
}

import json
with open('data/test_prompts.json', 'w') as f:
    json.dump(benchmark_prompts, f, indent=2)
```

### 7.3 Iterative Improvement Workflow

```
1. Train model
   ↓
2. Evaluate on benchmark
   ↓
3. Identify weaknesses
   ↓
4. Augment training data (add examples for weak areas)
   ↓
5. Fine-tune model
   ↓
6. Re-evaluate
   ↓
7. Compare with previous version
   ↓
8. Repeat until quality targets met
```

---

## Summary

### Evaluation Checklist

- [ ] **Training Metrics**: Loss decreasing, perplexity reasonable
- [ ] **Automated Tests**: Syntax >85%, patterns matching
- [ ] **Human Eval**: Rated 3+ on all dimensions
- [ ] **Comparative**: Better than baseline/previous versions
- [ ] **Diversity**: Generates varied solutions
- [ ] **Debugging**: No major issues identified
- [ ] **Benchmark**: Standard test set evaluated

### Quality Targets

| Metric | Minimum | Good | Excellent |
|--------|---------|------|-----------|
| Syntax Accuracy | 70% | 85% | 95% |
| Prompt Following | 60% | 80% | 90% |
| Code Quality Score | 60/100 | 80/100 | 90/100 |
| Human Rating | 2.5/5 | 3.5/5 | 4.5/5 |
| Perplexity | <50 | <20 | <10 |

---

**Next Steps**: Once you've evaluated your model, see [docs/ADVANCED_TOPICS.md](ADVANCED_TOPICS.md#4-performance-optimization) for optimization strategies.
