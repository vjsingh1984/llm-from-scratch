#!/usr/bin/env python3
"""
Prepare tool calling dataset for Stage 3 training.

Creates synthetic examples of tool usage for coding tasks:
- Code execution
- Documentation search
- Code analysis
- Debugging
- Testing
"""

import json
import random
from pathlib import Path
from typing import List, Dict, Any
import numpy as np
from tokenizers import Tokenizer

# Tool definitions for coding tasks
CODING_TOOLS = {
    "execute_python": {
        "description": "Execute Python code and return the output",
        "parameters": {
            "code": "string - The Python code to execute"
        }
    },
    "execute_bash": {
        "description": "Execute bash command and return the output",
        "parameters": {
            "command": "string - The bash command to execute"
        }
    },
    "search_docs": {
        "description": "Search Python documentation for a topic",
        "parameters": {
            "query": "string - The search query"
        }
    },
    "analyze_code": {
        "description": "Analyze code for errors, complexity, or patterns",
        "parameters": {
            "code": "string - The code to analyze",
            "analysis_type": "string - Type: 'syntax', 'complexity', 'security'"
        }
    },
    "run_tests": {
        "description": "Run unit tests for the given code",
        "parameters": {
            "code": "string - The code with test cases",
            "test_framework": "string - Framework: 'pytest', 'unittest'"
        }
    },
    "debug_code": {
        "description": "Debug code and suggest fixes",
        "parameters": {
            "code": "string - The buggy code",
            "error": "string - The error message"
        }
    }
}

# Example tool calling scenarios
TOOL_CALLING_EXAMPLES = [
    {
        "user": "Write a Python function to calculate factorial and test it with n=5",
        "assistant_reasoning": "I'll write the factorial function and test it using execute_python",
        "tool_calls": [
            {
                "tool": "execute_python",
                "arguments": {
                    "code": "def factorial(n):\n    if n <= 1:\n        return 1\n    return n * factorial(n-1)\n\nprint(factorial(5))"
                }
            }
        ],
        "tool_results": [
            {
                "tool": "execute_python",
                "output": "120"
            }
        ],
        "final_response": "Here's a factorial function:\n\n```python\ndef factorial(n):\n    if n <= 1:\n        return 1\n    return n * factorial(n-1)\n```\n\nTested with n=5, result: 120 ✓"
    },
    {
        "user": "How do I use list comprehensions in Python?",
        "assistant_reasoning": "I'll search the documentation for list comprehensions",
        "tool_calls": [
            {
                "tool": "search_docs",
                "arguments": {
                    "query": "list comprehensions python"
                }
            }
        ],
        "tool_results": [
            {
                "tool": "search_docs",
                "output": "List comprehensions provide a concise way to create lists. Syntax: [expression for item in iterable if condition]"
            }
        ],
        "final_response": "List comprehensions let you create lists concisely:\n\n```python\n# Syntax: [expression for item in iterable if condition]\nsquares = [x**2 for x in range(10)]\nevens = [x for x in range(20) if x % 2 == 0]\n```"
    },
    {
        "user": "Create a bash script to backup files and show me it works",
        "assistant_reasoning": "I'll write a backup script and test it with execute_bash",
        "tool_calls": [
            {
                "tool": "execute_bash",
                "arguments": {
                    "command": "cat << 'EOF' > backup.sh\n#!/bin/bash\ncp -r /path/to/source /path/to/backup_$(date +%Y%m%d)\nEOF\nchmod +x backup.sh\ncat backup.sh"
                }
            }
        ],
        "tool_results": [
            {
                "tool": "execute_bash",
                "output": "#!/bin/bash\ncp -r /path/to/source /path/to/backup_20250120"
            }
        ],
        "final_response": "Here's a backup script:\n\n```bash\n#!/bin/bash\ncp -r /path/to/source /path/to/backup_$(date +%Y%m%d)\n```\n\nThe script creates timestamped backups."
    },
    {
        "user": "Analyze this code for security issues: user_input = eval(input())",
        "assistant_reasoning": "I'll use analyze_code to check for security vulnerabilities",
        "tool_calls": [
            {
                "tool": "analyze_code",
                "arguments": {
                    "code": "user_input = eval(input())",
                    "analysis_type": "security"
                }
            }
        ],
        "tool_results": [
            {
                "tool": "analyze_code",
                "output": "CRITICAL: eval() allows arbitrary code execution. Use ast.literal_eval() instead for safe evaluation."
            }
        ],
        "final_response": "⚠️ Security issue detected!\n\nThe code uses `eval()` which allows arbitrary code execution. Replace with:\n\n```python\nimport ast\nuser_input = ast.literal_eval(input())\n```"
    },
    {
        "user": "Write a function to reverse a string and test it",
        "assistant_reasoning": "I'll write the function and test it using execute_python with test cases",
        "tool_calls": [
            {
                "tool": "execute_python",
                "arguments": {
                    "code": "def reverse_string(s):\n    return s[::-1]\n\n# Test cases\nassert reverse_string('hello') == 'olleh'\nassert reverse_string('') == ''\nassert reverse_string('a') == 'a'\nprint('All tests passed!')"
                }
            }
        ],
        "tool_results": [
            {
                "tool": "execute_python",
                "output": "All tests passed!"
            }
        ],
        "final_response": "Here's a string reversal function:\n\n```python\ndef reverse_string(s):\n    return s[::-1]\n```\n\nAll test cases passed ✓"
    }
]


def format_tool_calling_example(example: Dict[str, Any]) -> str:
    """
    Format a tool calling example into training format.

    Format:
    <|user|>User query<|end|>
    <|assistant|><|tool_call|>
    Tool: tool_name
    Arguments: {"arg": "value"}
    <|end|>
    <|tool_result|>
    Result: output
    <|end|>
    <|assistant|>Final response<|end|>
    """
    formatted = f"<|user|>{example['user']}<|end|>\n"

    # Add tool calls
    for tool_call in example['tool_calls']:
        formatted += "<|assistant|><|tool_call|>\n"
        formatted += f"Tool: {tool_call['tool']}\n"
        formatted += f"Arguments: {json.dumps(tool_call['arguments'])}\n"
        formatted += "<|end|>\n"

    # Add tool results
    for result in example['tool_results']:
        formatted += "<|tool_result|>\n"
        formatted += f"Tool: {result['tool']}\n"
        formatted += f"Output: {result['output']}\n"
        formatted += "<|end|>\n"

    # Add final response
    formatted += f"<|assistant|>{example['final_response']}<|end|>\n"

    return formatted


def generate_synthetic_examples(num_examples: int = 1000) -> List[str]:
    """Generate synthetic tool calling examples."""
    examples = []

    # Use base examples and create variations
    for i in range(num_examples):
        base = random.choice(TOOL_CALLING_EXAMPLES)
        # Create variation (for now, just use as-is)
        # In production, you'd generate variations with different parameters
        formatted = format_tool_calling_example(base)
        examples.append(formatted)

    return examples


def tokenize_and_save(examples: List[str], tokenizer_path: Path, output_dir: Path, split: str):
    """Tokenize examples and save as numpy arrays."""
    print(f"\nTokenizing {split} data...")
    print(f"  Examples: {len(examples)}")

    # Load tokenizer
    tokenizer = Tokenizer.from_file(str(tokenizer_path / "tokenizer.json"))

    # Tokenize each example
    all_tokens = []
    max_length = 1024  # Match Stage 1 and 2

    for example in examples:
        encoding = tokenizer.encode(example)
        tokens = encoding.ids

        # Truncate or pad to max_length
        if len(tokens) > max_length:
            tokens = tokens[:max_length]
        else:
            # Pad with zeros
            tokens = tokens + [0] * (max_length - len(tokens))

        all_tokens.append(tokens)

    # Convert to numpy array
    tokens_array = np.array(all_tokens, dtype=np.int32)

    # Save
    output_file = output_dir / f"tool_calling_{split}.npy"
    np.save(output_file, tokens_array)

    print(f"  Saved: {output_file}")
    print(f"  Shape: {tokens_array.shape}")
    print(f"  Tokens: {tokens_array.shape[0] * tokens_array.shape[1]:,}")


def main():
    # Paths
    project_root = Path(__file__).parent.parent
    tokenizer_path = project_root / "data" / "tokenizer"
    output_dir = project_root / "data" / "processed"

    print("=" * 60)
    print("Tool Calling Dataset Preparation (Stage 3)")
    print("=" * 60)

    # Generate synthetic examples
    print("\nGenerating synthetic tool calling examples...")
    all_examples = generate_synthetic_examples(num_examples=2000)

    # Split into train/val (90/10)
    split_idx = int(0.9 * len(all_examples))
    train_examples = all_examples[:split_idx]
    val_examples = all_examples[split_idx:]

    print(f"  Train examples: {len(train_examples)}")
    print(f"  Val examples: {len(val_examples)}")

    # Tokenize and save
    tokenize_and_save(train_examples, tokenizer_path, output_dir, "train")
    tokenize_and_save(val_examples, tokenizer_path, output_dir, "val")

    # Save tool definitions
    tools_file = output_dir / "tool_definitions.json"
    with open(tools_file, 'w') as f:
        json.dump(CODING_TOOLS, f, indent=2)
    print(f"\n✓ Saved tool definitions: {tools_file}")

    # Save example for reference
    example_file = output_dir / "tool_calling_example.txt"
    with open(example_file, 'w') as f:
        f.write(format_tool_calling_example(TOOL_CALLING_EXAMPLES[0]))
    print(f"✓ Saved example format: {example_file}")

    print("\n" + "=" * 60)
    print("✓ Tool calling dataset preparation complete!")
    print("=" * 60)
    print("\nNext step: Train Stage 3")
    print("  python scripts/train.py --stage tool_calling --checkpoint models/code_model_best.pth")


if __name__ == "__main__":
    main()
