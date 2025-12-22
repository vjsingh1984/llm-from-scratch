#!/usr/bin/env python3
"""
Example: Train a Tiny Model from Scratch

This script demonstrates the complete training pipeline using
the tiny model configuration. Perfect for testing and learning.

Time: ~30 minutes
Memory: ~1GB

Usage:
    python examples/01_train_tiny_model.py
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
import numpy as np
from tokenizers import Tokenizer
from src.model import create_model
from src.model.config import get_config


def main():
    print("=" * 60)
    print("Training a Tiny Model from Scratch")
    print("=" * 60)

    # Check device
    if torch.backends.mps.is_available():
        device = 'mps'
    elif torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'
    print(f"\nDevice: {device}")

    # Check if data exists
    tokenizer_path = project_root / "data/tokenizer/tokenizer.json"
    train_path = project_root / "data/processed/train_large.npy"

    if not tokenizer_path.exists():
        print(f"\n! Tokenizer not found: {tokenizer_path}")
        print("Run: python scripts/train_code_tokenizer.py")
        return

    if not train_path.exists():
        print(f"\n! Training data not found: {train_path}")
        print("Run: python scripts/prepare_data.py")
        return

    # Load tokenizer
    print("\nLoading tokenizer...")
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    print(f"  Vocabulary size: {tokenizer.get_vocab_size():,}")

    # Create tiny model
    print("\nCreating tiny model...")
    config = get_config('tiny')
    config.use_rmsnorm = True
    config.use_rope = True
    model = create_model(config, device=device)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {total_params:,}")
    print(f"  Memory estimate: {total_params * 4 / 1e9:.2f} GB")

    # Load a small sample of training data
    print("\nLoading training data (sample)...")
    train_data = np.load(train_path)
    sample_size = min(100, len(train_data))
    train_sample = train_data[:sample_size]
    print(f"  Using {sample_size} sequences for demo")

    # Simple training loop (demonstration)
    print("\nStarting training (3 steps for demo)...")
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    model.train()

    for step in range(3):
        # Get batch
        batch_idx = step % sample_size
        input_ids = torch.tensor(
            train_sample[batch_idx:batch_idx+1, :-1],
            device=device,
            dtype=torch.long
        )
        targets = torch.tensor(
            train_sample[batch_idx:batch_idx+1, 1:],
            device=device,
            dtype=torch.long
        )

        # Forward pass
        logits = model(input_ids)

        # Compute loss
        loss = torch.nn.functional.cross_entropy(
            logits.view(-1, config.vocab_size),
            targets.view(-1),
            ignore_index=0  # Ignore padding
        )

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(f"  Step {step + 1}/3: loss = {loss.item():.4f}")

    # Test generation
    print("\nTesting generation...")
    model.eval()
    prompt = "Once upon a time"
    encoding = tokenizer.encode(prompt)
    input_ids = torch.tensor([encoding.ids], device=device)

    with torch.no_grad():
        for _ in range(20):
            logits = model(input_ids)
            next_token = logits[0, -1, :].argmax().unsqueeze(0).unsqueeze(0)
            input_ids = torch.cat([input_ids, next_token], dim=1)

    output = tokenizer.decode(input_ids[0].tolist())
    print(f"  Prompt: '{prompt}'")
    print(f"  Output: '{output}'")

    print("\n" + "=" * 60)
    print("Demo Complete!")
    print("=" * 60)
    print("\nThis was a minimal demo. For full training, run:")
    print("  python scripts/train.py --stage language --model-size tiny")


if __name__ == "__main__":
    main()
