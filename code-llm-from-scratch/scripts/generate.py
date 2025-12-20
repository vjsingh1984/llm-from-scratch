"""
Generate bash scripts using the trained model.

Usage:
    python scripts/generate.py --prompt "Create a backup script"
    python scripts/generate.py --prompt "Monitor system resources" --max-length 200
"""

import os
import sys
import json
import argparse
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from tokenizer import BPETokenizer
from model import CodeTransformer, CoderConfig
import torch
import torch.nn.functional as F


def load_model(model_path, config_path, tokenizer, device):
    """Load trained model."""
    print(f"Loading model from: {model_path}")

    checkpoint = torch.load(model_path, map_location=device)

    # Get config
    if 'config' in checkpoint:
        config = checkpoint['config']
    else:
        # Load from separate config file
        with open(config_path, 'r') as f:
            config_dict = json.load(f)
        config = CoderConfig(
            vocab_size=config_dict.get('vocab_size', len(tokenizer.vocab)),
            max_seq_len=config_dict.get('max_seq_len', 512)
        )

    # Create model
    model = CodeTransformer(config)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()

    print(f"✓ Model loaded ({sum(p.numel() for p in model.parameters()):,} parameters)")

    return model, config


def generate_text(
    model,
    tokenizer,
    prompt,
    max_length=500,
    temperature=0.8,
    top_k=50,
    top_p=0.95,
    device="cpu"
):
    """Generate text from prompt."""

    # Encode prompt
    input_ids = tokenizer.encode(prompt)
    input_ids = torch.tensor([input_ids], dtype=torch.long, device=device)

    generated = input_ids.clone()

    print(f"\nGenerating (max {max_length} tokens)...")
    print("=" * 60)

    with torch.no_grad():
        for i in range(max_length):
            # Get predictions
            logits, _ = model(generated)

            # Get logits for last token
            next_token_logits = logits[0, -1, :] / temperature

            # Apply top-k filtering
            if top_k > 0:
                indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
                next_token_logits[indices_to_remove] = float('-inf')

            # Apply top-p (nucleus) filtering
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

                # Remove tokens with cumulative probability above the threshold
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0

                indices_to_remove = sorted_indices[sorted_indices_to_remove]
                next_token_logits[indices_to_remove] = float('-inf')

            # Sample from distribution
            probs = F.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

            # Append to generated sequence
            generated = torch.cat([generated, next_token.unsqueeze(0)], dim=1)

            # Check for EOS token (if defined)
            if hasattr(tokenizer.vocab, 'eos_id') and next_token.item() == tokenizer.vocab.eos_id:
                break

            # Print progress
            if (i + 1) % 50 == 0:
                print(f"Generated {i + 1} tokens...")

    # Decode
    generated_ids = generated[0].tolist()
    generated_text = tokenizer.decode(generated_ids)

    return generated_text


def main():
    parser = argparse.ArgumentParser(description="Generate bash scripts")
    parser.add_argument("--prompt", type=str, required=True, help="Generation prompt")
    parser.add_argument("--model", type=str, default="models/code/code_model_final.pt", help="Model path")
    parser.add_argument("--tokenizer", type=str, default="models/language/language_tokenizer.json", help="Tokenizer path")
    parser.add_argument("--max-length", type=int, default=500, help="Maximum tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.8, help="Sampling temperature")
    parser.add_argument("--top-k", type=int, default=50, help="Top-k sampling")
    parser.add_argument("--top-p", type=float, default=0.95, help="Top-p (nucleus) sampling")
    parser.add_argument("--output", type=str, help="Output file (optional)")

    args = parser.parse_args()

    # Device
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load tokenizer
    print(f"\nLoading tokenizer from: {args.tokenizer}")
    tokenizer = BPETokenizer()
    tokenizer.load(args.tokenizer)
    print(f"✓ Tokenizer loaded (vocab size: {len(tokenizer.vocab):,})")

    # Load model
    config_path = os.path.join(os.path.dirname(args.model), "generation_config.json")
    model, config = load_model(args.model, config_path, tokenizer, device)

    # Generate
    print(f"\nPrompt: {args.prompt}")

    generated_text = generate_text(
        model=model,
        tokenizer=tokenizer,
        prompt=args.prompt,
        max_length=args.max_length,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        device=device
    )

    # Display result
    print("=" * 60)
    print("Generated Script:")
    print("=" * 60)
    print(generated_text)
    print("=" * 60)

    # Save if requested
    if args.output:
        with open(args.output, 'w') as f:
            f.write(generated_text)
        print(f"\n✓ Saved to: {args.output}")


if __name__ == "__main__":
    main()
