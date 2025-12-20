"""
Generate bash code from English prompts using the bilingual model.

The model understands English and generates bash code!
"""

import sys
from pathlib import Path
import torch
import argparse

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from tokenizer import BPETokenizer
from model import CodeTransformer
from model.config import CoderConfig


def load_model(checkpoint_path: Path, tokenizer_dir: Path, device: str):
    """Load model and tokenizer."""
    print("Loading model...")

    # Load tokenizer
    tokenizer = BPETokenizer.load(tokenizer_dir)

    # Load model
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = CoderConfig(**checkpoint['config'])

    model = CodeTransformer(config)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()

    print(f"✓ Model loaded ({model.count_parameters():,} parameters)")
    print(f"✓ Tokenizer loaded ({len(tokenizer.vocab)} vocab)")

    return model, tokenizer


def generate(model, tokenizer, prompt, max_tokens=100, temperature=0.7, top_k=40, device='mps'):
    """Generate text from prompt."""
    # Encode
    input_ids = tokenizer.encode(prompt, add_special_tokens=True)
    input_ids = torch.tensor([input_ids], device=device)

    # Generate
    with torch.no_grad():
        output_ids = model.generate(
            input_ids,
            max_new_tokens=max_tokens,
            temperature=temperature,
            top_k=top_k
        )

    # Decode
    generated = tokenizer.decode(output_ids[0].cpu().tolist(), skip_special_tokens=True)

    return generated


def main():
    parser = argparse.ArgumentParser(description='Generate code from English prompts')

    parser.add_argument('--checkpoint', type=Path,
                        default=Path(__file__).parent.parent / 'checkpoints_final' / 'best_model.pt',
                        help='Model checkpoint')
    parser.add_argument('--tokenizer-dir', type=Path,
                        default=Path(__file__).parent.parent / 'tokenizer_trained',
                        help='Tokenizer directory')

    parser.add_argument('--prompt', type=str,
                        default='Create a backup script',
                        help='Prompt (English or code start)')
    parser.add_argument('--max-tokens', type=int, default=100,
                        help='Maximum tokens to generate')
    parser.add_argument('--temperature', type=float, default=0.7,
                        help='Sampling temperature')
    parser.add_argument('--top-k', type=int, default=40,
                        help='Top-k sampling')

    parser.add_argument('--device', type=str, default=None,
                        choices=['mps', 'cuda', 'cpu'])

    parser.add_argument('--interactive', action='store_true',
                        help='Interactive mode')

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
    print("Bilingual Code Generator")
    print("="*60)
    print()

    # Load
    model, tokenizer = load_model(args.checkpoint, args.tokenizer_dir, args.device)

    if args.interactive:
        # Interactive mode
        print("\n" + "="*60)
        print("Interactive Mode")
        print("="*60)
        print("Type English prompts to generate bash code.")
        print("Type 'quit' to exit.")
        print("="*60)
        print()

        while True:
            try:
                prompt = input("\nPrompt: ")

                if prompt.lower() in ['quit', 'exit', 'q']:
                    break

                print("\nGenerating...")
                generated = generate(model, tokenizer, prompt, args.max_tokens, args.temperature, args.top_k, args.device)

                print("\n" + "-"*60)
                print(generated)
                print("-"*60)

            except KeyboardInterrupt:
                print("\n\nExiting...")
                break
            except Exception as e:
                print(f"Error: {e}")

    else:
        # Single generation
        print(f"\nPrompt: {repr(args.prompt)}")
        print(f"Max tokens: {args.max_tokens}")
        print(f"Temperature: {args.temperature}")
        print()

        print("Generating...")
        generated = generate(model, tokenizer, args.prompt, args.max_tokens, args.temperature, args.top_k, args.device)

        print("\n" + "="*60)
        print("Generated:")
        print("="*60)
        print(generated)
        print("="*60)


if __name__ == '__main__':
    main()
