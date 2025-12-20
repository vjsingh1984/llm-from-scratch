#!/usr/bin/env python3
"""
Advanced Example: Fine-Tuning on Custom Data

This example demonstrates how to fine-tune the pretrained model on your own
bash scripts or code. It shows:

1. Data preparation and validation
2. Custom dataset creation
3. Training loop with monitoring
4. Checkpoint management
5. Evaluation and comparison

Usage:
    python examples/fine_tuning.py --data-path my_scripts/ --output-dir models/custom/

Requirements:
    - A pretrained language model in models/language/
    - Your custom bash scripts (text files or JSON)
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import List, Dict, Optional
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.model.transformer import CodeTransformer
from src.model.config import CoderConfig, get_model_config
from src.tokenizer.bpe import BPETokenizer
from src.training.optimizer import get_optimizer_and_scheduler


class CustomCodeDataset(Dataset):
    """Dataset for custom code/script files."""

    def __init__(
        self,
        scripts: List[str],
        tokenizer: BPETokenizer,
        max_length: int = 512
    ):
        self.scripts = scripts
        self.tokenizer = tokenizer
        self.max_length = max_length

        # Pre-tokenize all scripts
        print("Tokenizing scripts...")
        self.tokenized = []
        for script in tqdm(scripts):
            tokens = tokenizer.encode(script)
            if len(tokens) > 0:
                self.tokenized.append(tokens)

        print(f"Loaded {len(self.tokenized)} tokenized scripts")

    def __len__(self):
        return len(self.tokenized)

    def __getitem__(self, idx):
        tokens = self.tokenized[idx]

        # Truncate or pad to max_length
        if len(tokens) > self.max_length:
            tokens = tokens[:self.max_length]

        # Create input and target (shifted by 1)
        input_ids = torch.tensor(tokens[:-1], dtype=torch.long)
        target_ids = torch.tensor(tokens[1:], dtype=torch.long)

        return input_ids, target_ids


def collate_fn(batch):
    """Custom collate function to handle variable-length sequences."""
    input_ids = [item[0] for item in batch]
    target_ids = [item[1] for item in batch]

    # Find max length in batch
    max_len = max(len(ids) for ids in input_ids)

    # Pad sequences
    padded_inputs = []
    padded_targets = []

    for inp, tgt in zip(input_ids, target_ids):
        pad_len = max_len - len(inp)
        padded_inputs.append(F.pad(inp, (0, pad_len), value=0))
        padded_targets.append(F.pad(tgt, (0, pad_len), value=-100))  # -100 ignored in loss

    return torch.stack(padded_inputs), torch.stack(padded_targets)


def load_custom_data(data_path: str) -> List[str]:
    """
    Load custom bash scripts from directory or JSON file.

    Args:
        data_path: Path to directory of .sh files or JSON file

    Returns:
        List of script contents
    """
    scripts = []
    data_path = Path(data_path)

    if data_path.is_file():
        # Load from JSON
        if data_path.suffix == '.json':
            with open(data_path) as f:
                data = json.load(f)
                scripts = data.get('scripts', [])
        else:
            # Single file
            with open(data_path) as f:
                scripts = [f.read()]

    elif data_path.is_dir():
        # Load all .sh files from directory
        for script_file in sorted(data_path.glob('**/*.sh')):
            with open(script_file) as f:
                content = f.read()
                if content.strip():  # Skip empty files
                    scripts.append(content)

    else:
        raise ValueError(f"Data path not found: {data_path}")

    print(f"\nLoaded {len(scripts)} scripts from {data_path}")

    # Statistics
    total_chars = sum(len(s) for s in scripts)
    total_lines = sum(s.count('\n') for s in scripts)

    print(f"Total characters: {total_chars:,}")
    print(f"Total lines: {total_lines:,}")
    print(f"Average script length: {total_chars / len(scripts):.1f} chars")

    return scripts


def validate_data(scripts: List[str]) -> Dict[str, any]:
    """
    Validate and analyze the custom dataset.

    Returns:
        Dictionary with validation results and statistics
    """
    print("\n" + "="*60)
    print("DATA VALIDATION")
    print("="*60)

    results = {
        'num_scripts': len(scripts),
        'valid_scripts': 0,
        'empty_scripts': 0,
        'has_shebang': 0,
        'min_length': float('inf'),
        'max_length': 0,
        'warnings': []
    }

    for i, script in enumerate(scripts):
        if not script.strip():
            results['empty_scripts'] += 1
            results['warnings'].append(f"Script {i}: Empty content")
            continue

        results['valid_scripts'] += 1

        # Check shebang
        if script.startswith('#!'):
            results['has_shebang'] += 1

        # Track lengths
        script_len = len(script)
        results['min_length'] = min(results['min_length'], script_len)
        results['max_length'] = max(results['max_length'], script_len)

    # Print results
    print(f"Valid scripts: {results['valid_scripts']}/{results['num_scripts']}")
    print(f"Scripts with shebang: {results['has_shebang']}/{results['valid_scripts']}")
    print(f"Length range: {results['min_length']}-{results['max_length']} chars")

    if results['warnings']:
        print(f"\nWarnings ({len(results['warnings'])}):")
        for warning in results['warnings'][:5]:  # Show first 5
            print(f"  - {warning}")
        if len(results['warnings']) > 5:
            print(f"  ... and {len(results['warnings']) - 5} more")

    # Validation checks
    if results['valid_scripts'] < 10:
        print("\n⚠️  WARNING: Less than 10 valid scripts. Consider adding more data.")

    if results['has_shebang'] / results['valid_scripts'] < 0.5:
        print("\n⚠️  WARNING: Many scripts missing shebang. May affect generation quality.")

    print("="*60 + "\n")

    return results


def train_epoch(
    model: CodeTransformer,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int
) -> float:
    """Train for one epoch."""
    model.train()
    total_loss = 0

    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
    for batch_idx, (input_ids, target_ids) in enumerate(pbar):
        input_ids = input_ids.to(device)
        target_ids = target_ids.to(device)

        optimizer.zero_grad()

        logits, loss = model(input_ids, target_ids)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()

        # Update progress bar
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'avg_loss': f'{total_loss / (batch_idx + 1):.4f}'
        })

    return total_loss / len(dataloader)


def evaluate(
    model: CodeTransformer,
    dataloader: DataLoader,
    device: torch.device
) -> float:
    """Evaluate model on validation set."""
    model.eval()
    total_loss = 0

    with torch.no_grad():
        for input_ids, target_ids in dataloader:
            input_ids = input_ids.to(device)
            target_ids = target_ids.to(device)

            _, loss = model(input_ids, target_ids)
            total_loss += loss.item()

    return total_loss / len(dataloader)


def main():
    parser = argparse.ArgumentParser(description='Fine-tune model on custom data')
    parser.add_argument('--data-path', type=str, required=True,
                        help='Path to directory of .sh files or JSON file')
    parser.add_argument('--pretrained-model', type=str,
                        default='models/language/language_model_final.pt',
                        help='Path to pretrained model')
    parser.add_argument('--tokenizer', type=str,
                        default='models/language/language_tokenizer.json',
                        help='Path to tokenizer')
    parser.add_argument('--output-dir', type=str, default='models/custom',
                        help='Output directory for fine-tuned model')
    parser.add_argument('--batch-size', type=int, default=8,
                        help='Batch size for training')
    parser.add_argument('--num-epochs', type=int, default=30,
                        help='Number of training epochs')
    parser.add_argument('--learning-rate', type=float, default=1e-4,
                        help='Learning rate (lower for fine-tuning)')
    parser.add_argument('--validation-split', type=float, default=0.1,
                        help='Fraction of data for validation')
    parser.add_argument('--model-size', type=str, default='small',
                        choices=['tiny', 'small', 'medium'],
                        help='Model size (should match pretrained model)')

    args = parser.parse_args()

    print("\n" + "="*60)
    print("CUSTOM FINE-TUNING")
    print("="*60)

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Load custom data
    scripts = load_custom_data(args.data_path)

    # Validate data
    validation_results = validate_data(scripts)

    # Load tokenizer
    print("\nLoading tokenizer...")
    tokenizer = BPETokenizer()
    tokenizer.load(args.tokenizer)
    print(f"Tokenizer loaded: {len(tokenizer.vocab)} tokens")

    # Split data
    num_val = int(len(scripts) * args.validation_split)
    val_scripts = scripts[:num_val] if num_val > 0 else []
    train_scripts = scripts[num_val:]

    print(f"\nData split:")
    print(f"  Training: {len(train_scripts)} scripts")
    print(f"  Validation: {len(val_scripts)} scripts")

    # Create datasets
    train_dataset = CustomCodeDataset(train_scripts, tokenizer)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn
    )

    val_loader = None
    if val_scripts:
        val_dataset = CustomCodeDataset(val_scripts, tokenizer)
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            collate_fn=collate_fn
        )

    # Setup device
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')

    print(f"\nUsing device: {device}")

    # Load pretrained model
    print(f"\nLoading pretrained model from {args.pretrained_model}...")

    config = get_model_config(args.model_size, vocab_size=len(tokenizer.vocab))
    model = CodeTransformer(config)

    checkpoint = torch.load(args.pretrained_model, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)

    print(f"Model loaded: {sum(p.numel() for p in model.parameters()):,} parameters")

    # Setup optimizer
    optimizer, scheduler = get_optimizer_and_scheduler(
        model,
        learning_rate=args.learning_rate,
        num_training_steps=len(train_loader) * args.num_epochs
    )

    # Training loop
    print(f"\n{'='*60}")
    print("TRAINING")
    print(f"{'='*60}\n")

    best_val_loss = float('inf')
    train_losses = []
    val_losses = []

    for epoch in range(1, args.num_epochs + 1):
        # Train
        train_loss = train_epoch(model, train_loader, optimizer, device, epoch)
        train_losses.append(train_loss)

        # Validate
        val_loss = None
        if val_loader:
            val_loss = evaluate(model, val_loader, device)
            val_losses.append(val_loss)

        # Update scheduler
        if scheduler:
            scheduler.step()

        # Print epoch summary
        print(f"\nEpoch {epoch}/{args.num_epochs}:")
        print(f"  Train Loss: {train_loss:.4f}")
        if val_loss is not None:
            print(f"  Val Loss: {val_loss:.4f}")

            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                checkpoint_path = os.path.join(args.output_dir, 'best_model.pt')
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'train_loss': train_loss,
                    'val_loss': val_loss,
                }, checkpoint_path)
                print(f"  ✓ Best model saved")

        # Save checkpoint every 5 epochs
        if epoch % 5 == 0:
            checkpoint_path = os.path.join(args.output_dir, f'checkpoint_epoch_{epoch}.pt')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
            }, checkpoint_path)

    # Save final model
    final_path = os.path.join(args.output_dir, 'custom_model_final.pt')
    torch.save({
        'epoch': args.num_epochs,
        'model_state_dict': model.state_dict(),
        'config': config,
        'train_losses': train_losses,
        'val_losses': val_losses,
    }, final_path)

    print(f"\n{'='*60}")
    print("TRAINING COMPLETE")
    print(f"{'='*60}")
    print(f"\nFinal model saved to: {final_path}")

    if val_losses:
        print(f"Best validation loss: {best_val_loss:.4f}")
        print(f"Final validation loss: {val_losses[-1]:.4f}")

    # Save training history
    history_path = os.path.join(args.output_dir, 'training_history.json')
    with open(history_path, 'w') as f:
        json.dump({
            'train_losses': train_losses,
            'val_losses': val_losses,
            'config': {
                'data_path': args.data_path,
                'num_scripts': len(scripts),
                'num_epochs': args.num_epochs,
                'batch_size': args.batch_size,
                'learning_rate': args.learning_rate,
            }
        }, f, indent=2)

    print(f"Training history saved to: {history_path}")

    print("\nNext steps:")
    print(f"  1. Test generation: python scripts/generate.py --model {final_path}")
    print(f"  2. Compare with base model to see improvement")
    print(f"  3. Fine-tune further if needed with --num-epochs {args.num_epochs + 20}")


if __name__ == '__main__':
    main()
