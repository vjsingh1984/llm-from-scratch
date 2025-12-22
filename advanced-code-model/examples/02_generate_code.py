#!/usr/bin/env python3
"""
Example: Generate Code with a Trained Model

This script demonstrates how to load a trained model
and generate code from prompts.

Usage:
    python examples/02_generate_code.py
    python examples/02_generate_code.py --checkpoint models/code_model_best.pth
    python examples/02_generate_code.py --prompt "#!/bin/bash\n# List files"
"""

import sys
from pathlib import Path
import argparse

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
import torch.nn.functional as F
from tokenizers import Tokenizer
from src.model import create_model
from src.model.config import get_config


def load_model(checkpoint_path: str, device: str = 'mps'):
    """Load a trained model from checkpoint."""
    print(f"Loading model from {checkpoint_path}...")

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Create model with same config
    config = get_config('large')
    config.use_rmsnorm = True
    config.use_rope = True
    model = create_model(config, device=device)

    # Handle _orig_mod prefix from compiled models
    state_dict = checkpoint['model_state_dict']
    if any(key.startswith('_orig_mod.') for key in state_dict.keys()):
        state_dict = {
            key.replace('_orig_mod.', ''): value
            for key, value in state_dict.items()
        }

    model.load_state_dict(state_dict)
    model.eval()
    return model


def generate(
    model,
    tokenizer,
    prompt: str,
    max_tokens: int = 100,
    temperature: float = 0.7,
    top_k: int = 50,
    device: str = 'mps'
) -> str:
    """Generate text from a prompt."""
    # Tokenize prompt
    encoding = tokenizer.encode(prompt)
    input_ids = torch.tensor([encoding.ids], device=device)

    # Generate
    with torch.no_grad():
        for _ in range(max_tokens):
            logits = model(input_ids)
            next_logits = logits[0, -1, :] / temperature

            # Top-k filtering
            if top_k > 0:
                values, indices = torch.topk(next_logits, top_k)
                next_logits = torch.full_like(next_logits, float('-inf'))
                next_logits.scatter_(0, indices, values)

            # Sample
            probs = F.softmax(next_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

            # Check for end token
            if next_token.item() == tokenizer.token_to_id("<eos>"):
                break

            # Append
            input_ids = torch.cat([input_ids, next_token.unsqueeze(0)], dim=1)

    return tokenizer.decode(input_ids[0].tolist())


def main():
    parser = argparse.ArgumentParser(description="Generate code with trained model")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="models/code_model_best.pth",
        help="Path to model checkpoint"
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default=None,
        help="Custom prompt (optional)"
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=100,
        help="Maximum tokens to generate"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature (lower = more focused)"
    )
    args = parser.parse_args()

    print("=" * 60)
    print("Code Generation Example")
    print("=" * 60)

    # Check device
    if torch.backends.mps.is_available():
        device = 'mps'
    elif torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'
    print(f"\nDevice: {device}")

    # Check if checkpoint exists
    checkpoint_path = project_root / args.checkpoint
    if not checkpoint_path.exists():
        print(f"\n! Checkpoint not found: {checkpoint_path}")
        print("Train a model first:")
        print("  python scripts/train.py --stage code --checkpoint models/language_model_best.pth")
        return

    # Load tokenizer
    tokenizer_path = project_root / "data/tokenizer/tokenizer.json"
    if not tokenizer_path.exists():
        print(f"\n! Tokenizer not found: {tokenizer_path}")
        return

    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    print(f"Tokenizer vocabulary: {tokenizer.get_vocab_size():,}")

    # Load model
    model = load_model(str(checkpoint_path), device=device)

    # Test prompts
    if args.prompt:
        prompts = [args.prompt]
    else:
        prompts = [
            "#!/bin/bash\n# List all files in current directory\n",
            "#!/bin/bash\n# Backup a directory\n",
            "### Language: Python\n### Code:\ndef fibonacci(n):\n",
        ]

    # Generate for each prompt
    print("\n" + "=" * 60)
    print("Generating Code")
    print("=" * 60)

    for i, prompt in enumerate(prompts, 1):
        print(f"\n--- Example {i} ---")
        print(f"Prompt:\n{prompt}")
        print("\nGenerated:")

        output = generate(
            model,
            tokenizer,
            prompt,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            device=device
        )

        # Print only the generated part
        generated = output[len(prompt):]
        print(generated)
        print("-" * 40)

    print("\n" + "=" * 60)
    print("Generation Complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
