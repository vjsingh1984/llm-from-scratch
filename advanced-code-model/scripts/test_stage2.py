#!/usr/bin/env python3
"""
Quick test script for Stage 2 (Code) model.

Tests the trained code model with sample bash prompts.
"""

import torch
import torch.nn.functional as F
from pathlib import Path
from tokenizers import Tokenizer
import sys

sys.path.append(str(Path(__file__).parent.parent))
from src.model import create_model
from src.model.config import get_config


def fix_tokenizer_spacing(text):
    """Fix extra spaces added by tokenizer decode.

    The tokenizer was trained on natural language, so it adds spaces
    between tokens. This function cleans up code-specific patterns.
    """
    import re

    # Fix common code patterns
    fixes = [
        (r'# !/bin/', '#!/bin/'),           # shebang
        (r'# !', '#!'),                      # shebang start
        (r'\( \)', '()'),                    # empty parens
        (r'\[ \[', '[['),                    # bash [[
        (r'\] \]', ']]'),                    # bash ]]
        (r' \. ', '.'),                      # method access
        (r' \( ', '('),                      # opening paren
        (r' \) ', ') '),                     # closing paren
        (r' : ', ':'),                       # colon
        (r' ;', ';'),                        # semicolon
        (r' , ', ', '),                      # comma
        (r' / ', '/'),                       # path separator
        (r'" \$ ', '"$'),                    # variable in string
        (r' "', '"'),                        # quote spacing
        (r'" ', '"'),                        # quote spacing
        (r'\$ \{', '${'),                    # bash variable
        (r'\} \}', '}}'),                    # bash closing
        (r' = ', '='),                       # assignment (be careful)
        (r' \+ ', '+'),                      # operators
        (r' - ', '-'),                       # minus
    ]

    result = text
    for pattern, replacement in fixes:
        result = re.sub(pattern, replacement, result)

    return result


def generate(model, tokenizer, prompt, max_tokens=80, temperature=0.7, device='mps'):
    """Generate code from prompt."""
    model.eval()

    # Tokenize
    encoding = tokenizer.encode(prompt)
    input_ids = torch.tensor([encoding.ids], device=device)

    # Generate
    with torch.no_grad():
        for _ in range(max_tokens):
            logits = model(input_ids)
            next_logits = logits[0, -1, :] / temperature

            # Sample
            probs = F.softmax(next_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

            # Append
            input_ids = torch.cat([input_ids, next_token.unsqueeze(0)], dim=1)

    # Decode and fix spacing
    raw_output = tokenizer.decode(input_ids[0].tolist())
    return fix_tokenizer_spacing(raw_output)


def main():
    print("="*60)
    print("Testing Stage 2 (Code) Model")
    print("="*60)

    # Check if checkpoint exists
    checkpoint_path = Path('models/code_model_best.pth')
    if not checkpoint_path.exists():
        print(f"\n❌ ERROR: Checkpoint not found at {checkpoint_path}")
        print("\nMake sure Stage 2 training is complete.")
        print("The checkpoint should be at: models/code_model_best.pth")
        return

    # Load model
    print("\n📦 Loading model...")
    config = get_config('large')
    config.use_rmsnorm = True
    config.use_rope = True

    model = create_model(config, device='mps')

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='mps')
    state_dict = checkpoint['model_state_dict']

    # Strip _orig_mod prefix if present
    if any(key.startswith('_orig_mod.') for key in state_dict.keys()):
        state_dict = {
            key.replace('_orig_mod.', ''): value
            for key, value in state_dict.items()
        }

    model.load_state_dict(state_dict)
    model.eval()
    print("✓ Model loaded")

    # Load tokenizer
    print("\n📚 Loading tokenizer...")
    tokenizer = Tokenizer.from_file('data/tokenizer/tokenizer.json')
    print("✓ Tokenizer loaded")

    # Test prompts
    test_prompts = [
        "#!/bin/bash\n# List all files\n",
        "#!/bin/bash\n# Backup directory\n",
        "#!/bin/bash\n# Check disk space\n",
    ]

    print("\n" + "="*60)
    print("Generating Code")
    print("="*60)

    for i, prompt in enumerate(test_prompts, 1):
        print(f"\n{'─'*60}")
        print(f"Test {i}/{len(test_prompts)}")
        print('─'*60)
        print(f"📝 Prompt:\n{prompt}")
        print("🤖 Generating...")

        output = generate(model, tokenizer, prompt, max_tokens=80, temperature=0.7)

        print(f"\n✨ Generated:\n{output}")
        print('─'*60)

    print("\n" + "="*60)
    print("✓ Testing Complete!")
    print("="*60)
    print("\nYour Stage 2 model is working! 🎉")
    print("\nNext: Run Stage 3 (Tool Calling)")
    print("  python3 scripts/prepare_tool_calling_data.py")


if __name__ == "__main__":
    main()
