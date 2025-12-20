"""
Stage 1: Language Pretraining

Train the model on natural language (TinyStories) to learn general language understanding.
This creates a foundation that will later be fine-tuned on code.
"""

import os
import sys
import json
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from tokenizer import BPETokenizer, Vocabulary
from model import CodeTransformer, CoderConfig
from training import Trainer, create_dataloaders
import torch


def load_language_data(data_path):
    """Load TinyStories dataset."""
    print(f"Loading language data from: {data_path}")

    with open(data_path, 'r', encoding='utf-8') as f:
        texts = f.readlines()

    # Clean and filter
    texts = [t.strip() for t in texts if t.strip()]

    print(f"Loaded {len(texts):,} text samples")
    print(f"Total characters: {sum(len(t) for t in texts):,}")

    return texts


def train_tokenizer(texts, vocab_size=8000, save_path="models/tokenizer"):
    """Train BPE tokenizer on language data."""
    print("\nTraining BPE tokenizer...")
    print(f"Target vocabulary size: {vocab_size:,}")

    # Create tokenizer
    tokenizer = BPETokenizer()
    tokenizer.target_vocab_size = vocab_size

    # Train on sample of data (for speed)
    sample_size = min(5000, len(texts))
    sample_texts = texts[:sample_size]

    print(f"Training on {sample_size:,} samples...")
    tokenizer.train(sample_texts, verbose=True)

    # Save tokenizer
    os.makedirs(save_path, exist_ok=True)
    tokenizer.save(f"{save_path}/language_tokenizer.json")

    print(f"\n✓ Tokenizer trained with vocab size: {len(tokenizer.vocab):,}")
    print(f"  Saved to: {save_path}/language_tokenizer.json")

    return tokenizer


def create_model(vocab_size, config_size="small"):
    """Create transformer model."""
    print(f"\nCreating {config_size} model...")

    # Model configurations
    configs = {
        "tiny": CoderConfig(
            vocab_size=vocab_size,
            n_layers=4,
            d_model=256,
            n_heads=4,
            d_ff=1024,
            max_seq_len=256,
        ),
        "small": CoderConfig(
            vocab_size=vocab_size,
            n_layers=6,
            d_model=384,
            n_heads=6,
            d_ff=1536,
            max_seq_len=512,
        ),
        "medium": CoderConfig(
            vocab_size=vocab_size,
            n_layers=12,
            d_model=768,
            n_heads=12,
            d_ff=3072,
            max_seq_len=1024,
        ),
    }

    config = configs[config_size]
    model = CodeTransformer(config)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"  Configuration: {config_size}")
    print(f"  Vocabulary: {config.vocab_size:,}")
    print(f"  Layers: {config.n_layers}")
    print(f"  Hidden size: {config.d_model}")
    print(f"  Attention heads: {config.n_heads}")
    print(f"  Max sequence length: {config.max_seq_len}")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    print(f"  Model size: {total_params * 4 / (1024**2):.1f} MB (FP32)")

    return model, config


def prepare_data(texts, tokenizer, val_split=0.1):
    """Prepare training and validation data."""
    print("\nPreparing data...")

    # Split into train/val
    split_idx = int(len(texts) * (1 - val_split))
    train_texts = texts[:split_idx]
    val_texts = texts[split_idx:]

    print(f"  Training samples: {len(train_texts):,}")
    print(f"  Validation samples: {len(val_texts):,}")

    # Tokenize
    print("  Tokenizing training data...")
    train_tokens = [tokenizer.encode(text) for text in train_texts]

    print("  Tokenizing validation data...")
    val_tokens = [tokenizer.encode(text) for text in val_texts]

    # Calculate statistics
    train_total_tokens = sum(len(t) for t in train_tokens)
    val_total_tokens = sum(len(t) for t in val_tokens)

    print(f"  Training tokens: {train_total_tokens:,}")
    print(f"  Validation tokens: {val_total_tokens:,}")
    print(f"  Average tokens per sample: {train_total_tokens / len(train_tokens):.1f}")

    return train_tokens, val_tokens


def main():
    """Main training pipeline."""
    print("=" * 60)
    print("Stage 1: Language Pretraining")
    print("=" * 60)

    # Configuration
    DATA_PATH = "../data/tinystories_5000.txt"
    MODEL_SIZE = "small"  # tiny, small, or medium
    VOCAB_SIZE = 8000
    BATCH_SIZE = 16
    MAX_EPOCHS = 10
    LEARNING_RATE = 3e-4
    SAVE_DIR = "models/language"

    # Device
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"\nUsing device: {device}")

    # 1. Load data
    texts = load_language_data(DATA_PATH)

    # 2. Train or load tokenizer
    tokenizer_path = f"{SAVE_DIR}/language_tokenizer.json"
    if os.path.exists(tokenizer_path):
        print(f"\nLoading existing tokenizer from: {tokenizer_path}")
        tokenizer = BPETokenizer()
        tokenizer.load(tokenizer_path)
        print(f"  Vocabulary size: {len(tokenizer.vocab):,}")
    else:
        tokenizer = train_tokenizer(texts, vocab_size=VOCAB_SIZE, save_path=SAVE_DIR)

    # 3. Create model
    model, config = create_model(len(tokenizer.vocab), config_size=MODEL_SIZE)
    model = model.to(device)

    # 4. Prepare data
    train_tokens, val_tokens = prepare_data(texts, tokenizer)

    # 5. Create data loaders
    print("\nCreating data loaders...")
    train_loader, val_loader = create_dataloaders(
        train_tokens,
        val_tokens,
        batch_size=BATCH_SIZE,
        max_seq_len=config.max_seq_len,
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
        log_interval=10
    )

    # 7. Train
    print("\nStarting training...")
    print("=" * 60)

    trainer.train(
        num_epochs=MAX_EPOCHS,
        eval_interval=len(train_loader),  # Eval once per epoch
        save_interval=len(train_loader)   # Save once per epoch
    )

    # 8. Save final model
    final_path = f"{SAVE_DIR}/language_model_final.pt"
    trainer.save_checkpoint(final_path)

    print("\n" + "=" * 60)
    print("Language Pretraining Complete!")
    print("=" * 60)
    print(f"✓ Final model saved to: {final_path}")
    print(f"✓ Tokenizer saved to: {tokenizer_path}")
    print("\nNext step: Run train_code.py to fine-tune on bash scripts")


if __name__ == "__main__":
    main()
