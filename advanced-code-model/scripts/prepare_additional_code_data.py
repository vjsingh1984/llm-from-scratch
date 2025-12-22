#!/usr/bin/env python3
"""
Prepare Additional Code Training Data

Combines existing code data with additional bash scripts
and prepares for continued training.
"""

import os
import sys
from pathlib import Path
import numpy as np
from tokenizers import Tokenizer
import random

# Add project to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def load_and_tokenize_text(text_file: Path, tokenizer: Tokenizer, max_length: int = 1024) -> np.ndarray:
    """Load text file and tokenize into sequences."""

    print(f"Loading: {text_file}")
    content = text_file.read_text(encoding="utf-8", errors="ignore")

    # Split into individual scripts (by shebang or file markers)
    scripts = []
    current_script = []

    for line in content.split("\n"):
        if line.startswith("#!/") or line.startswith("# File:"):
            if current_script:
                scripts.append("\n".join(current_script))
            current_script = [line]
        else:
            current_script.append(line)

    if current_script:
        scripts.append("\n".join(current_script))

    print(f"  Found {len(scripts)} scripts")

    # Tokenize each script
    all_tokens = []
    for script in scripts:
        if len(script.strip()) < 50:  # Skip very short scripts
            continue

        encoding = tokenizer.encode(script)
        tokens = encoding.ids

        # Create fixed-length sequences
        for i in range(0, len(tokens), max_length):
            chunk = tokens[i:i + max_length]
            if len(chunk) < max_length:
                chunk = chunk + [0] * (max_length - len(chunk))
            all_tokens.append(chunk)

    return np.array(all_tokens, dtype=np.int32)


def main():
    project_root = Path(__file__).parent.parent
    tokenizer_path = project_root / "data" / "tokenizer" / "tokenizer.json"
    processed_dir = project_root / "data" / "processed"
    raw_dir = project_root / "data" / "raw"

    print("=" * 60)
    print("Preparing Additional Code Training Data")
    print("=" * 60)

    # Load tokenizer
    print("\nLoading tokenizer...")
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    print(f"  Vocabulary size: {tokenizer.get_vocab_size()}")

    # Load existing code data
    existing_train_path = processed_dir / "code_train.npy"
    existing_val_path = processed_dir / "code_val.npy"

    existing_train = None
    existing_val = None

    if existing_train_path.exists():
        existing_train = np.load(existing_train_path)
        print(f"\nExisting training data: {existing_train.shape}")

    if existing_val_path.exists():
        existing_val = np.load(existing_val_path)
        print(f"Existing validation data: {existing_val.shape}")

    # Process additional bash scripts
    additional_file = raw_dir / "additional_bash_scripts.txt"
    additional_tokens = None

    if additional_file.exists():
        print(f"\nProcessing additional scripts...")
        additional_tokens = load_and_tokenize_text(additional_file, tokenizer)
        print(f"  Additional tokens: {additional_tokens.shape}")
    else:
        print(f"\n⚠ Additional scripts file not found: {additional_file}")
        print("  Run: python3 scripts/download_more_bash.py first")

        # Still create some synthetic examples
        print("\nGenerating synthetic bash examples...")
        additional_tokens = generate_synthetic_bash(tokenizer, num_examples=500)

    # Combine datasets
    print("\nCombining datasets...")

    if existing_train is not None and additional_tokens is not None:
        # Split additional into train/val (90/10)
        n_additional = len(additional_tokens)
        split_idx = int(0.9 * n_additional)
        additional_train = additional_tokens[:split_idx]
        additional_val = additional_tokens[split_idx:]

        # Combine
        combined_train = np.concatenate([existing_train, additional_train], axis=0)
        combined_val = np.concatenate([existing_val, additional_val], axis=0)

        print(f"  Combined training: {combined_train.shape}")
        print(f"  Combined validation: {combined_val.shape}")
    elif additional_tokens is not None:
        split_idx = int(0.9 * len(additional_tokens))
        combined_train = additional_tokens[:split_idx]
        combined_val = additional_tokens[split_idx:]
    else:
        print("No data to process!")
        return

    # Shuffle
    np.random.shuffle(combined_train)

    # Save
    output_train = processed_dir / "code_train_extended.npy"
    output_val = processed_dir / "code_val_extended.npy"

    np.save(output_train, combined_train)
    np.save(output_val, combined_val)

    print(f"\n✓ Saved extended training data: {output_train}")
    print(f"✓ Saved extended validation data: {output_val}")

    # Calculate sizes
    train_tokens = combined_train.shape[0] * combined_train.shape[1]
    val_tokens = combined_val.shape[0] * combined_val.shape[1]
    train_size = combined_train.nbytes / (1024 * 1024)
    val_size = combined_val.nbytes / (1024 * 1024)

    print(f"\nDataset Statistics:")
    print(f"  Training: {combined_train.shape[0]} sequences, {train_tokens:,} tokens ({train_size:.1f} MB)")
    print(f"  Validation: {combined_val.shape[0]} sequences, {val_tokens:,} tokens ({val_size:.1f} MB)")

    # Instructions
    print("\n" + "=" * 60)
    print("Resume Training")
    print("=" * 60)
    print("""
Use the extended dataset to continue Stage 2 training:

python3 scripts/train.py \\
  --stage code \\
  --checkpoint models/code_model_best.pth \\
  --data-file code_train_extended.npy \\
  --batch-size 2 --num-epochs 10 \\
  --learning-rate 3e-6 \\
  --warmup-steps 200 \\
  --use-rmsnorm --use-rope --use-compile --use-amp

Or to train from scratch with combined data:

python3 scripts/train.py \\
  --stage code \\
  --checkpoint models/language_model_best.pth \\
  --data-file code_train_extended.npy \\
  --batch-size 2 --num-epochs 15 \\
  --learning-rate 5e-6 \\
  --warmup-steps 300 \\
  --use-rmsnorm --use-rope --use-compile --use-amp
""")


def generate_synthetic_bash(tokenizer: Tokenizer, num_examples: int = 500, max_length: int = 1024) -> np.ndarray:
    """Generate synthetic bash examples for training."""

    templates = [
        # File operations
        '''#!/bin/bash
# {comment}
set -euo pipefail

{var_name}="${{1:-{default}}}"

if [[ -f "${var_name}" ]]; then
    {action}
else
    echo "File not found: ${var_name}"
    exit 1
fi''',

        # Loop examples
        '''#!/bin/bash
# {comment}
for {var_name} in {items}; do
    echo "Processing: ${var_name}"
    {action}
done''',

        # Function examples
        '''#!/bin/bash
# {comment}

{func_name}() {{
    local {param}="$1"
    {body}
}}

# Main
{func_name} "$@"''',

        # Conditional examples
        '''#!/bin/bash
# {comment}

if [[ {condition} ]]; then
    {action1}
elif [[ {condition2} ]]; then
    {action2}
else
    {action3}
fi''',

        # Case statement
        '''#!/bin/bash
# {comment}

case "$1" in
    {option1})
        {action1}
        ;;
    {option2})
        {action2}
        ;;
    *)
        echo "Usage: $0 {{{option1}|{option2}}}"
        exit 1
        ;;
esac''',
    ]

    variables = {
        "comment": ["Process files", "Backup data", "Deploy application", "Check system", "Install packages",
                   "Configure service", "Update repository", "Clean temporary files", "Monitor logs"],
        "var_name": ["file", "dir", "path", "name", "input", "output", "target", "source", "config"],
        "default": ["/tmp/data", ".", "/var/log", "$HOME", "/etc/config", "/opt/app"],
        "action": ["cat \"$file\"", "rm -rf \"$dir\"", "cp \"$source\" \"$target\"", "chmod +x \"$file\"",
                  "tar -czf backup.tar.gz \"$dir\"", "wc -l \"$file\""],
        "items": ["*.txt", "$(seq 1 10)", "${files[@]}", "$(find . -name '*.sh')"],
        "func_name": ["process", "backup", "deploy", "check", "install", "configure", "cleanup"],
        "param": ["input", "file", "dir", "name", "value"],
        "body": ["echo \"$input\"", "cp \"$file\" /backup/", "rm -rf \"$dir\"", "return 0"],
        "condition": ["-z \"$var\"", "-n \"$var\"", "-f \"$file\"", "-d \"$dir\"", "$# -eq 0"],
        "condition2": ["-r \"$file\"", "-w \"$dir\"", "-x \"$script\""],
        "action1": ["echo 'Success'", "exit 0", "return 0", "continue"],
        "action2": ["echo 'Warning'", "exit 1", "return 1", "break"],
        "action3": ["echo 'Error'", "exit 2", "usage"],
        "option1": ["start", "install", "build", "test", "deploy"],
        "option2": ["stop", "uninstall", "clean", "lint", "rollback"],
    }

    examples = []
    for _ in range(num_examples):
        template = random.choice(templates)

        # Fill in template
        script = template
        for var, options in variables.items():
            placeholder = "{" + var + "}"
            if placeholder in script:
                script = script.replace(placeholder, random.choice(options), 1)

        encoding = tokenizer.encode(script)
        tokens = encoding.ids[:max_length]

        if len(tokens) < max_length:
            tokens = tokens + [0] * (max_length - len(tokens))

        examples.append(tokens)

    return np.array(examples, dtype=np.int32)


if __name__ == "__main__":
    main()
