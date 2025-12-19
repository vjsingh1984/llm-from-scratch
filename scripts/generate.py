"""
Text generation script.

Load a trained model and generate text from prompts.

Example usage:
    python scripts/generate.py --checkpoint checkpoints/best.npz --prompt "Once upon a time"
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import mlx.core as mx
from model import create_model
from tokenizer import BPETokenizer
import argparse


def generate_text(
    model,
    tokenizer,
    prompt: str,
    max_tokens: int = 100,
    temperature: float = 0.8,
    top_k: int = 40,
    num_samples: int = 1
) -> list[str]:
    """
    Generate text from a prompt.

    Args:
        model: Trained model
        tokenizer: Tokenizer
        prompt: Text prompt
        max_tokens: Maximum tokens to generate
        temperature: Sampling temperature
        top_k: Top-k sampling
        num_samples: Number of samples to generate

    Returns:
        List of generated texts
    """
    samples = []

    for _ in range(num_samples):
        # Encode prompt
        prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
        input_ids = mx.array([prompt_ids])

        # Generate
        output_ids = model.generate(
            input_ids,
            max_new_tokens=max_tokens,
            temperature=temperature,
            top_k=top_k
        )

        # Decode
        generated_text = tokenizer.decode(
            output_ids[0].tolist(),
            skip_special_tokens=True
        )

        samples.append(generated_text)

    return samples


def main():
    parser = argparse.ArgumentParser(description='Generate text from trained model')

    # Model and checkpoint
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--tokenizer-path', type=str, default='tokenizer_model',
                       help='Path to tokenizer')
    parser.add_argument('--model-size', type=str, default='tiny',
                       help='Model size (must match training)')
    parser.add_argument('--vocab-size', type=int, default=8000,
                       help='Vocabulary size (must match training)')
    parser.add_argument('--seq-len', type=int, default=256,
                       help='Sequence length (must match training)')

    # Generation parameters
    parser.add_argument('--prompt', type=str, default=None,
                       help='Text prompt')
    parser.add_argument('--max-tokens', type=int, default=100,
                       help='Maximum tokens to generate')
    parser.add_argument('--temperature', type=float, default=0.8,
                       help='Sampling temperature')
    parser.add_argument('--top-k', type=int, default=40,
                       help='Top-k sampling')
    parser.add_argument('--num-samples', type=int, default=1,
                       help='Number of samples to generate')

    # Interactive mode
    parser.add_argument('--interactive', action='store_true',
                       help='Interactive mode')

    args = parser.parse_args()

    # Load tokenizer
    print("Loading tokenizer...")
    tokenizer = BPETokenizer.load(Path(args.tokenizer_path))
    print(f"Vocabulary size: {len(tokenizer.vocab)}")

    # Create model
    print("\nCreating model...")
    model = create_model(
        model_size=args.model_size,
        vocab_size=len(tokenizer.vocab),
        max_seq_len=args.seq_len
    )

    # Load checkpoint
    print(f"\nLoading checkpoint: {args.checkpoint}")
    model.load_weights(args.checkpoint)
    print("Model loaded successfully!")

    print("\n" + "="*60)
    print("Ready to generate text!")
    print("="*60)

    if args.interactive:
        # Interactive mode
        print("\nInteractive mode. Type prompts (or 'quit' to exit):")
        while True:
            prompt = input("\nPrompt: ").strip()
            if prompt.lower() in ['quit', 'exit', 'q']:
                break

            if not prompt:
                continue

            print("\nGenerating...")
            samples = generate_text(
                model,
                tokenizer,
                prompt,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
                top_k=args.top_k,
                num_samples=args.num_samples
            )

            print("\nGenerated text:")
            print("-" * 60)
            for i, sample in enumerate(samples):
                if args.num_samples > 1:
                    print(f"\nSample {i+1}:")
                print(sample)
            print("-" * 60)

    else:
        # Single generation
        if args.prompt is None:
            # Default prompts
            prompts = [
                "Once upon a time",
                "The cat sat on the",
                "In a galaxy far away",
            ]
        else:
            prompts = [args.prompt]

        for prompt in prompts:
            print(f"\nPrompt: {prompt}")
            print("-" * 60)

            samples = generate_text(
                model,
                tokenizer,
                prompt,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
                top_k=args.top_k,
                num_samples=args.num_samples
            )

            for i, sample in enumerate(samples):
                if args.num_samples > 1:
                    print(f"\nSample {i+1}:")
                print(sample)
            print("-" * 60)


if __name__ == '__main__':
    main()
