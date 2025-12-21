#!/usr/bin/env python3
"""
Prepare RLHF (Reinforcement Learning from Human Feedback) dataset for Stage 4.

Creates preference pairs for training a reward model:
- Generate multiple outputs for same prompt
- Simulate human preferences (chosen vs rejected)
- Save preference pairs for reward model training
"""

import json
import random
from pathlib import Path
from typing import List, Dict, Any, Tuple
import numpy as np
from tokenizers import Tokenizer


# Coding task prompts for preference generation
CODING_PROMPTS = [
    "Write a function to check if a number is prime",
    "Implement binary search algorithm",
    "Create a function to reverse a linked list",
    "Write a merge sort implementation",
    "Implement a stack data structure",
    "Create a function to find the longest common subsequence",
    "Write a function to validate parentheses",
    "Implement breadth-first search for a graph",
    "Create a function to detect cycles in a linked list",
    "Write a function to find the kth largest element",
    "Implement a hash table with collision handling",
    "Create a function to rotate an array",
    "Write a function to find all permutations of a string",
    "Implement depth-first search for a graph",
    "Create a function to check if a string is a palindrome",
]

# Quality dimensions for preference scoring
QUALITY_CRITERIA = {
    "correctness": {
        "high": ["correct algorithm", "handles edge cases", "no bugs"],
        "low": ["incorrect logic", "missing edge cases", "has bugs"]
    },
    "readability": {
        "high": ["clear variable names", "good comments", "proper structure"],
        "low": ["unclear names", "no comments", "messy structure"]
    },
    "efficiency": {
        "high": ["optimal time complexity", "space efficient", "well optimized"],
        "low": ["inefficient algorithm", "wasteful space usage", "unoptimized"]
    },
    "style": {
        "high": ["follows PEP8", "consistent formatting", "pythonic"],
        "low": ["poor formatting", "inconsistent style", "unpythonic"]
    }
}


def generate_code_response(prompt: str, quality_level: str = "high") -> str:
    """
    Generate a code response based on quality level.

    Args:
        prompt: Coding task prompt
        quality_level: "high" or "low" quality response

    Returns:
        Generated code response
    """
    # Simplified response generation (in production, use actual model)
    templates = {
        "high": {
            "prime": """def is_prime(n: int) -> bool:
    \"\"\"Check if a number is prime.\"\"\"
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
    return True""",
            "binary_search": """def binary_search(arr: list, target: int) -> int:
    \"\"\"Binary search implementation.\"\"\"
    left, right = 0, len(arr) - 1

    while left <= right:
        mid = (left + right) // 2
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1

    return -1  # Not found""",
        },
        "low": {
            "prime": """def is_prime(n):
    for i in range(2, n):
        if n % i == 0:
            return False
    return True""",
            "binary_search": """def binary_search(arr, x):
    for i in range(len(arr)):
        if arr[i] == x:
            return i
    return -1""",
        }
    }

    # Simple keyword matching
    if "prime" in prompt.lower():
        return templates[quality_level]["prime"]
    elif "binary search" in prompt.lower():
        return templates[quality_level]["binary_search"]
    else:
        # Generic response
        if quality_level == "high":
            return f"""def solve_problem():
    \"\"\"High-quality solution with documentation.\"\"\"
    # TODO: Implement solution
    pass"""
        else:
            return """def solve():
    pass"""


def create_preference_pair(prompt: str) -> Dict[str, Any]:
    """
    Create a preference pair (chosen vs rejected response).

    Args:
        prompt: Coding task prompt

    Returns:
        Preference pair dictionary
    """
    # Generate high and low quality responses
    chosen_response = generate_code_response(prompt, "high")
    rejected_response = generate_code_response(prompt, "low")

    # Random quality criteria for reasoning
    criteria = random.choice(list(QUALITY_CRITERIA.keys()))

    return {
        "prompt": prompt,
        "chosen": chosen_response,
        "rejected": rejected_response,
        "criteria": criteria,
        "reason": f"Chosen response has better {criteria}"
    }


def format_preference_example(example: Dict[str, Any]) -> str:
    """
    Format preference pair for training.

    Format:
    <|user|>Prompt<|end|>
    <|assistant|>Chosen response<|end|>
    <|preference|>chosen<|end|>
    ---
    <|user|>Prompt<|end|>
    <|assistant|>Rejected response<|end|>
    <|preference|>rejected<|end|>
    """
    formatted = ""

    # Chosen example
    formatted += f"<|user|>{example['prompt']}<|end|>\n"
    formatted += f"<|assistant|>{example['chosen']}<|end|>\n"
    formatted += "<|preference|>chosen<|end|>\n"
    formatted += "---\n"

    # Rejected example
    formatted += f"<|user|>{example['prompt']}<|end|>\n"
    formatted += f"<|assistant|>{example['rejected']}<|end|>\n"
    formatted += "<|preference|>rejected<|end|>\n"

    return formatted


def generate_preference_dataset(num_pairs: int = 1000) -> List[Dict[str, Any]]:
    """Generate synthetic preference pairs."""
    pairs = []

    for i in range(num_pairs):
        prompt = random.choice(CODING_PROMPTS)
        pair = create_preference_pair(prompt)
        pairs.append(pair)

    return pairs


def save_preference_pairs(pairs: List[Dict[str, Any]], output_dir: Path):
    """Save preference pairs as JSON."""
    output_file = output_dir / "rlhf_preferences.json"

    with open(output_file, 'w') as f:
        json.dump(pairs, f, indent=2)

    print(f"\n✓ Saved preference pairs: {output_file}")
    print(f"  Total pairs: {len(pairs)}")


def tokenize_and_save_pairs(pairs: List[Dict[str, Any]], tokenizer_path: Path,
                            output_dir: Path, split: str):
    """Tokenize preference pairs and save for reward model training."""
    print(f"\nTokenizing {split} preference data...")
    print(f"  Pairs: {len(pairs)}")

    # Load tokenizer
    tokenizer = Tokenizer.from_file(str(tokenizer_path / "tokenizer.json"))

    chosen_tokens = []
    rejected_tokens = []
    max_length = 1024

    for pair in pairs:
        # Format chosen example
        chosen_text = f"<|user|>{pair['prompt']}<|end|>\n<|assistant|>{pair['chosen']}<|end|>"
        chosen_encoding = tokenizer.encode(chosen_text)
        chosen_ids = chosen_encoding.ids

        # Format rejected example
        rejected_text = f"<|user|>{pair['prompt']}<|end|>\n<|assistant|>{pair['rejected']}<|end|>"
        rejected_encoding = tokenizer.encode(rejected_text)
        rejected_ids = rejected_encoding.ids

        # Truncate or pad
        if len(chosen_ids) > max_length:
            chosen_ids = chosen_ids[:max_length]
        else:
            chosen_ids = chosen_ids + [0] * (max_length - len(chosen_ids))

        if len(rejected_ids) > max_length:
            rejected_ids = rejected_ids[:max_length]
        else:
            rejected_ids = rejected_ids + [0] * (max_length - len(rejected_ids))

        chosen_tokens.append(chosen_ids)
        rejected_tokens.append(rejected_ids)

    # Save as numpy arrays
    chosen_array = np.array(chosen_tokens, dtype=np.int32)
    rejected_array = np.array(rejected_tokens, dtype=np.int32)

    np.save(output_dir / f"rlhf_{split}_chosen.npy", chosen_array)
    np.save(output_dir / f"rlhf_{split}_rejected.npy", rejected_array)

    print(f"  Saved chosen: {output_dir / f'rlhf_{split}_chosen.npy'}")
    print(f"  Saved rejected: {output_dir / f'rlhf_{split}_rejected.npy'}")
    print(f"  Shape: {chosen_array.shape}")


def main():
    # Paths
    project_root = Path(__file__).parent.parent
    tokenizer_path = project_root / "data" / "tokenizer"
    output_dir = project_root / "data" / "processed"

    print("=" * 60)
    print("RLHF Preference Dataset Preparation (Stage 4)")
    print("=" * 60)

    # Generate preference pairs
    print("\nGenerating preference pairs...")
    all_pairs = generate_preference_dataset(num_pairs=2000)

    # Split into train/val (90/10)
    split_idx = int(0.9 * len(all_pairs))
    train_pairs = all_pairs[:split_idx]
    val_pairs = all_pairs[split_idx:]

    print(f"  Train pairs: {len(train_pairs)}")
    print(f"  Val pairs: {len(val_pairs)}")

    # Save raw JSON
    save_preference_pairs(all_pairs, output_dir)

    # Tokenize and save
    tokenize_and_save_pairs(train_pairs, tokenizer_path, output_dir, "train")
    tokenize_and_save_pairs(val_pairs, tokenizer_path, output_dir, "val")

    # Save example
    example_file = output_dir / "rlhf_preference_example.txt"
    with open(example_file, 'w') as f:
        f.write(format_preference_example(all_pairs[0]))
    print(f"\n✓ Saved example format: {example_file}")

    print("\n" + "=" * 60)
    print("✓ RLHF preference dataset preparation complete!")
    print("=" * 60)
    print("\nNext step: Train Reward Model")
    print("  python scripts/train_reward_model.py --checkpoint models/tool_calling_model_best.pth")


if __name__ == "__main__":
    main()
