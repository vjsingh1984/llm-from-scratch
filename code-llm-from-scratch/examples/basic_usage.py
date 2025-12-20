"""
Basic Usage Example

This example demonstrates how to:
1. Load a trained model
2. Generate code from English prompts
3. Save generated scripts

Perfect for beginners and presentations!
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from tokenizer import BPETokenizer
from model import CodeTransformer, CoderConfig
import torch


def load_model_and_tokenizer():
    """Load the trained model and tokenizer."""
    print("Loading model and tokenizer...")

    # Paths
    model_path = "models/code/code_model_final.pt"
    tokenizer_path = "models/language/language_tokenizer.json"

    # Device
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    # Load tokenizer
    tokenizer = BPETokenizer()
    tokenizer.load(tokenizer_path)

    # Load model
    checkpoint = torch.load(model_path, map_location=device)

    if 'config' in checkpoint:
        config = checkpoint['config']
    else:
        config = CoderConfig(vocab_size=len(tokenizer.vocab))

    model = CodeTransformer(config)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()

    print(f"✓ Model loaded ({sum(p.numel() for p in model.parameters()):,} parameters)")
    print(f"✓ Tokenizer loaded (vocab size: {len(tokenizer.vocab):,})")

    return model, tokenizer, device


def generate_code(model, tokenizer, prompt, device, max_length=300):
    """Generate code from a prompt."""
    print(f"\nPrompt: {prompt}")
    print("=" * 60)

    # Encode prompt
    input_ids = tokenizer.encode(prompt)
    input_ids = torch.tensor([input_ids], dtype=torch.long, device=device)

    # Generate
    with torch.no_grad():
        for i in range(max_length):
            # Get predictions
            logits, _ = model(input_ids)

            # Sample next token
            probs = torch.softmax(logits[0, -1, :] / 0.8, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

            # Append
            input_ids = torch.cat([input_ids, next_token.unsqueeze(0)], dim=1)

            # Stop if we generate EOS (if defined)
            if hasattr(tokenizer.vocab, 'eos_id') and next_token.item() == tokenizer.vocab.eos_id:
                break

    # Decode
    generated_text = tokenizer.decode(input_ids[0].tolist())

    print(generated_text)
    print("=" * 60)

    return generated_text


def main():
    """Main example."""
    print("Code LLM - Basic Usage Example")
    print("=" * 60)

    # 1. Load model
    model, tokenizer, device = load_model_and_tokenizer()

    # 2. Example prompts
    prompts = [
        "#!/bin/bash\n# Backup script for MySQL database",
        "#!/bin/bash\n# System resource monitoring script",
        "#!/bin/bash\n# Deploy application to production",
    ]

    # 3. Generate code for each prompt
    for prompt in prompts:
        code = generate_code(model, tokenizer, prompt, device)

        # Optionally save
        filename = f"generated_{prompts.index(prompt) + 1}.sh"
        with open(filename, 'w') as f:
            f.write(code)
        print(f"✓ Saved to: {filename}\n")


if __name__ == "__main__":
    # Check if model exists
    if not Path("models/code/code_model_final.pt").exists():
        print("Error: Trained model not found!")
        print("Please run:")
        print("  1. python scripts/train_language.py")
        print("  2. python scripts/train_code.py")
        sys.exit(1)

    main()
