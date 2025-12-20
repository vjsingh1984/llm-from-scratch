"""
Generate bash code using trained model.

Usage:
    python scripts/generate.py --checkpoint checkpoints/best_model.pt --prompt "#!/bin/bash"
"""

import sys
from pathlib import Path
import torch
import argparse

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from tokenizer import CodeTokenizer
from model import CodeTransformer


def load_model(checkpoint_path: Path, device: str = 'mps'):
    """Load trained model from checkpoint."""
    print(f"Loading model from {checkpoint_path}...")

    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Recreate config
    from model.config import CoderConfig
    config = CoderConfig(**checkpoint['config'])

    # Create model
    model = CodeTransformer(config)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()

    print(f"  Parameters: {model.count_parameters():,}")
    print(f"  Device: {device}")
    print(f"  Training step: {checkpoint['step']}")

    return model, config


def generate_code(
    model,
    tokenizer,
    prompt: str,
    max_tokens: int = 100,
    temperature: float = 0.7,
    top_k: int = 40,
    top_p: float = 0.9,
    num_samples: int = 1,
    device: str = 'mps'
):
    """
    Generate code from prompt.

    Args:
        model: Trained model
        tokenizer: Tokenizer
        prompt: Starting prompt
        max_tokens: Maximum tokens to generate
        temperature: Sampling temperature (higher = more random)
        top_k: Top-k sampling
        top_p: Nucleus sampling
        num_samples: Number of samples to generate
        device: Device

    Returns:
        List of generated code samples
    """
    samples = []

    for i in range(num_samples):
        # Encode prompt
        input_ids = tokenizer.encode(prompt, add_special_tokens=True)
        input_ids = torch.tensor([input_ids], device=device)

        # Generate
        with torch.no_grad():
            output_ids = model.generate(
                input_ids,
                max_new_tokens=max_tokens,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p
            )

        # Decode
        generated_text = tokenizer.decode(
            output_ids[0].cpu().tolist(),
            skip_special_tokens=True
        )

        samples.append(generated_text)

    return samples


def main():
    parser = argparse.ArgumentParser(description='Generate bash code')

    # Model
    parser.add_argument('--checkpoint', type=Path, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--tokenizer-dir', type=Path,
                        default=Path(__file__).parent.parent / 'tokenizer_trained',
                        help='Tokenizer directory')

    # Generation
    parser.add_argument('--prompt', type=str, default='#!/bin/bash\n',
                        help='Starting prompt')
    parser.add_argument('--max-tokens', type=int, default=100,
                        help='Maximum tokens to generate')
    parser.add_argument('--temperature', type=float, default=0.7,
                        help='Sampling temperature')
    parser.add_argument('--top-k', type=int, default=40,
                        help='Top-k sampling')
    parser.add_argument('--top-p', type=float, default=0.9,
                        help='Nucleus sampling (top-p)')
    parser.add_argument('--num-samples', type=int, default=3,
                        help='Number of samples to generate')

    # Device
    parser.add_argument('--device', type=str, default=None,
                        choices=['mps', 'cuda', 'cpu'],
                        help='Device (default: auto-detect)')

    args = parser.parse_args()

    # Auto-detect device
    if args.device is None:
        if torch.backends.mps.is_available():
            args.device = 'mps'
        elif torch.cuda.is_available():
            args.device = 'cuda'
        else:
            args.device = 'cpu'

    print("="*60)
    print("Bash Code Generation")
    print("="*60)
    print()

    # Load tokenizer
    print("Loading tokenizer...")
    tokenizer = CodeTokenizer.load(args.tokenizer_dir)

    # Load model
    model, config = load_model(args.checkpoint, device=args.device)

    # Generate
    print("\n" + "="*60)
    print(f"Generating {args.num_samples} samples")
    print("="*60)
    print(f"Prompt: {repr(args.prompt)}")
    print(f"Max tokens: {args.max_tokens}")
    print(f"Temperature: {args.temperature}")
    print(f"Top-k: {args.top_k}")
    print(f"Top-p: {args.top_p}")
    print()

    samples = generate_code(
        model=model,
        tokenizer=tokenizer,
        prompt=args.prompt,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        num_samples=args.num_samples,
        device=args.device
    )

    # Print results
    for i, sample in enumerate(samples, 1):
        print("-"*60)
        print(f"Sample {i}:")
        print("-"*60)
        print(sample)
        print()

    print("="*60)


def interactive_mode():
    """Interactive generation mode."""
    parser = argparse.ArgumentParser(description='Interactive bash code generation')
    parser.add_argument('--checkpoint', type=Path, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--tokenizer-dir', type=Path,
                        default=Path(__file__).parent.parent / 'tokenizer_trained',
                        help='Tokenizer directory')
    parser.add_argument('--device', type=str, default=None,
                        choices=['mps', 'cuda', 'cpu'],
                        help='Device (default: auto-detect)')

    args = parser.parse_args()

    # Auto-detect device
    if args.device is None:
        if torch.backends.mps.is_available():
            args.device = 'mps'
        elif torch.cuda.is_available():
            args.device = 'cuda'
        else:
            args.device = 'cpu'

    # Load
    tokenizer = CodeTokenizer.load(args.tokenizer_dir)
    model, _ = load_model(args.checkpoint, device=args.device)

    print("\n" + "="*60)
    print("Interactive Bash Code Generation")
    print("="*60)
    print("Type a prompt and press Enter to generate.")
    print("Type 'quit' to exit.")
    print("="*60)
    print()

    while True:
        try:
            prompt = input("\nPrompt: ")

            if prompt.lower() in ['quit', 'exit', 'q']:
                break

            samples = generate_code(
                model=model,
                tokenizer=tokenizer,
                prompt=prompt,
                max_tokens=50,
                temperature=0.7,
                top_k=40,
                top_p=0.9,
                num_samples=1,
                device=args.device
            )

            print("\nGenerated:")
            print(samples[0])

        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as e:
            print(f"Error: {e}")


if __name__ == '__main__':
    main()
