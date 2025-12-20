"""
Stage 2: Code Fine-Tuning

Fine-tune the pretrained language model on bash scripts.
This teaches the model to generate code while retaining language understanding.
"""

import os
import sys
import json
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from tokenizer import BPETokenizer
from model import CodeTransformer, CoderConfig
from training import Trainer, create_dataloaders
import torch


def load_bash_scripts(data_dir):
    """Load bash scripts from dataset."""
    print(f"Loading bash scripts from: {data_dir}")

    # Load from JSON
    json_path = Path(data_dir) / "bash_scripts.json"

    if not json_path.exists():
        raise FileNotFoundError(f"Bash scripts not found at: {json_path}")

    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    scripts = data['scripts']

    print(f"Loaded {len(scripts):,} bash scripts")
    print(f"Total characters: {sum(len(s) for s in scripts):,}")
    print(f"Average script length: {sum(len(s) for s in scripts) / len(scripts):.1f} characters")

    return scripts


def load_pretrained_model(model_path, tokenizer, device):
    """Load pretrained language model."""
    print(f"\nLoading pretrained model from: {model_path}")

    if not os.path.exists(model_path):
        print("Warning: No pretrained model found. Starting from scratch.")
        return None

    checkpoint = torch.load(model_path, map_location=device)

    # Get config from checkpoint
    if 'config' in checkpoint:
        config = checkpoint['config']
    else:
        # Try to infer from state dict
        print("Warning: No config found in checkpoint, using default")
        config = CoderConfig(vocab_size=len(tokenizer.vocab))

    # Create model
    model = CodeTransformer(config)

    # Load weights
    model.load_state_dict(checkpoint['model_state_dict'])

    print(f"✓ Loaded model with {sum(p.numel() for p in model.parameters()):,} parameters")

    return model, config


def prepare_code_data(scripts, tokenizer, val_split=0.1):
    """Prepare code training data."""
    print("\nPreparing code data...")

    # Split into train/val
    split_idx = int(len(scripts) * (1 - val_split))
    train_scripts = scripts[:split_idx]
    val_scripts = scripts[split_idx:]

    print(f"  Training scripts: {len(train_scripts):,}")
    print(f"  Validation scripts: {len(val_scripts):,}")

    # Tokenize
    print("  Tokenizing training scripts...")
    train_tokens = [tokenizer.encode(script) for script in train_scripts]

    print("  Tokenizing validation scripts...")
    val_tokens = [tokenizer.encode(script) for script in val_scripts]

    # Calculate statistics
    train_total_tokens = sum(len(t) for t in train_tokens)
    val_total_tokens = sum(len(t) for t in val_tokens)

    print(f"  Training tokens: {train_total_tokens:,}")
    print(f"  Validation tokens: {val_total_tokens:,}")
    print(f"  Average tokens per script: {train_total_tokens / len(train_tokens):.1f}")

    return train_tokens, val_tokens


def main():
    """Main fine-tuning pipeline."""
    print("=" * 60)
    print("Stage 2: Code Fine-Tuning")
    print("=" * 60)

    # Configuration
    PRETRAINED_MODEL = "models/language/language_model_final.pt"
    TOKENIZER_PATH = "models/language/language_tokenizer.json"
    CODE_DATA_DIR = "../data/code"
    BATCH_SIZE = 8  # Smaller batch for code (longer sequences)
    MAX_EPOCHS = 20
    LEARNING_RATE = 1e-4  # Lower LR for fine-tuning
    SAVE_DIR = "models/code"

    # Device
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"\nUsing device: {device}")

    # 1. Load tokenizer
    print(f"\nLoading tokenizer from: {TOKENIZER_PATH}")
    if not os.path.exists(TOKENIZER_PATH):
        raise FileNotFoundError(f"Tokenizer not found. Run train_language.py first!")

    tokenizer = BPETokenizer()
    tokenizer.load(TOKENIZER_PATH)
    print(f"  Vocabulary size: {len(tokenizer.vocab):,}")

    # 2. Load bash scripts
    scripts = load_bash_scripts(CODE_DATA_DIR)

    # 3. Load pretrained model or create new
    model_config = None
    if os.path.exists(PRETRAINED_MODEL):
        model, model_config = load_pretrained_model(PRETRAINED_MODEL, tokenizer, device)
    else:
        print("\nNo pretrained model found. Creating new model...")
        from model import CoderConfig, CodeTransformer

        model_config = CoderConfig(
            vocab_size=len(tokenizer.vocab),
            n_layers=6,
            d_model=384,
            n_heads=6,
            d_ff=1536,
            max_seq_len=512,
        )
        model = CodeTransformer(model_config)

        print(f"  Created model with {sum(p.numel() for p in model.parameters()):,} parameters")

    model = model.to(device)

    # 4. Prepare code data
    train_tokens, val_tokens = prepare_code_data(scripts, tokenizer)

    # 5. Create data loaders
    print("\nCreating data loaders...")
    train_loader, val_loader = create_dataloaders(
        train_tokens,
        val_tokens,
        batch_size=BATCH_SIZE,
        max_seq_len=model_config.max_seq_len,
        shuffle=True
    )

    print(f"  Training batches: {len(train_loader):,}")
    print(f"  Validation batches: {len(val_loader):,}")

    # 6. Create trainer
    print("\nCreating trainer...")
    os.makedirs(SAVE_DIR, exist_ok=True)

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        lr=LEARNING_RATE,
        device=device,
        save_dir=SAVE_DIR,
        log_interval=5
    )

    # 7. Fine-tune
    print("\nStarting fine-tuning...")
    print("=" * 60)

    trainer.train(
        num_epochs=MAX_EPOCHS,
        eval_interval=len(train_loader),  # Eval once per epoch
        save_interval=len(train_loader)   # Save once per epoch
    )

    # 8. Save final model
    final_path = f"{SAVE_DIR}/code_model_final.pt"
    trainer.save_checkpoint(final_path)

    # Save generation config
    gen_config = {
        "model_path": final_path,
        "tokenizer_path": TOKENIZER_PATH,
        "max_seq_len": model_config.max_seq_len,
        "vocab_size": len(tokenizer.vocab),
    }

    with open(f"{SAVE_DIR}/generation_config.json", 'w') as f:
        json.dump(gen_config, f, indent=2)

    print("\n" + "=" * 60)
    print("Code Fine-Tuning Complete!")
    print("=" * 60)
    print(f"✓ Final model saved to: {final_path}")
    print(f"✓ Generation config saved to: {SAVE_DIR}/generation_config.json")
    print("\nYour model is ready to generate bash scripts!")
    print("Use the generate.py script to test it.")


if __name__ == "__main__":
    main()
