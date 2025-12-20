"""
Data loader for code training.

PyTorch implementation with efficient batching.
"""

import torch
from torch.utils.data import Dataset, DataLoader
from typing import List, Optional
from pathlib import Path
import json


class CodeDataset(Dataset):
    """
    Dataset for code samples.

    Creates training examples by:
    1. Tokenizing each code sample
    2. Creating (input, target) pairs where target is input shifted by 1
    3. Padding/truncating to max_length
    """

    def __init__(
        self,
        scripts: List[str],
        tokenizer,
        max_length: int = 256,
        stride: Optional[int] = None
    ):
        """
        Initialize dataset.

        Args:
            scripts: List of code samples
            tokenizer: Tokenizer instance
            max_length: Maximum sequence length
            stride: Stride for creating overlapping windows (default: max_length)
        """
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.stride = stride if stride is not None else max_length

        # Tokenize all scripts
        self.samples = []
        for script in scripts:
            # Encode without special tokens (we'll add them during batching)
            token_ids = tokenizer.encode(script, add_special_tokens=False)

            # Create windows
            for i in range(0, len(token_ids), self.stride):
                window = token_ids[i:i + max_length]

                # Only keep windows with reasonable length
                if len(window) >= 10:
                    self.samples.append(window)

        print(f"Created {len(self.samples)} training samples from {len(scripts)} scripts")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        """
        Get a single item.

        Returns:
            Dictionary with:
            - input_ids: Input token IDs
            - target_ids: Target token IDs (shifted by 1)
            - attention_mask: Mask for padding
        """
        token_ids = self.samples[idx]

        # Truncate if too long (need room for special tokens)
        if len(token_ids) > self.max_length:
            token_ids = token_ids[:self.max_length]

        # Pad if needed
        if len(token_ids) < self.max_length:
            padding_length = self.max_length - len(token_ids)
            token_ids = token_ids + [self.tokenizer.pad_id] * padding_length

        # Create input and target
        # Both should be same length (max_length)
        # Input: [BOS, tok1, tok2, ..., tokN-2]
        # Target: [tok1, tok2, ..., tokN-2, EOS]
        input_ids = [self.tokenizer.bos_id] + token_ids[:-1]
        target_ids = token_ids[:-1] + [self.tokenizer.eos_id]

        # Create attention mask (1 for real tokens, 0 for padding)
        attention_mask = [1 if tid != self.tokenizer.pad_id else 0 for tid in input_ids]

        return {
            'input_ids': torch.tensor(input_ids, dtype=torch.long),
            'target_ids': torch.tensor(target_ids, dtype=torch.long),
            'attention_mask': torch.tensor(attention_mask, dtype=torch.float32)
        }


def create_dataloaders(
    data_path: Path,
    tokenizer,
    batch_size: int = 8,
    max_length: int = 256,
    train_split: float = 0.9,
    num_workers: int = 0
) -> tuple:
    """
    Create train and validation dataloaders.

    Args:
        data_path: Path to data directory
        tokenizer: Tokenizer instance
        batch_size: Batch size
        max_length: Maximum sequence length
        train_split: Fraction of data for training
        num_workers: Number of worker processes

    Returns:
        Tuple of (train_loader, val_loader)
    """
    # Load scripts
    json_path = data_path / 'bash_scripts.json'
    with open(json_path, 'r') as f:
        data = json.load(f)
        scripts = data['scripts']

    print(f"Loaded {len(scripts)} scripts from {json_path}")

    # Split into train/val
    split_idx = int(len(scripts) * train_split)
    train_scripts = scripts[:split_idx]
    val_scripts = scripts[split_idx:]

    print(f"  Training: {len(train_scripts)} scripts")
    print(f"  Validation: {len(val_scripts)} scripts")

    # Create datasets
    train_dataset = CodeDataset(train_scripts, tokenizer, max_length)
    val_dataset = CodeDataset(val_scripts, tokenizer, max_length)

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True  # Faster transfer to GPU
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    return train_loader, val_loader


def test_dataloader():
    """Test the dataloader."""
    print("Testing Code DataLoader")
    print("="*60)

    # Create tokenizer
    from tokenizer import CodeTokenizer

    tokenizer = CodeTokenizer()
    tokenizer.build_default_vocab()

    # Load data
    data_path = Path(__file__).parent.parent / 'data'
    train_loader, val_loader = create_dataloaders(
        data_path,
        tokenizer,
        batch_size=4,
        max_length=64
    )

    print(f"\nDataLoader info:")
    print(f"  Train batches: {len(train_loader)}")
    print(f"  Val batches: {len(val_loader)}")

    # Get a batch
    batch = next(iter(train_loader))

    print(f"\nBatch shape:")
    print(f"  Input IDs: {batch['input_ids'].shape}")
    print(f"  Target IDs: {batch['target_ids'].shape}")
    print(f"  Attention mask: {batch['attention_mask'].shape}")

    # Decode first sample
    print(f"\nFirst sample:")
    input_text = tokenizer.decode(batch['input_ids'][0].tolist())
    target_text = tokenizer.decode(batch['target_ids'][0].tolist())

    print("Input:")
    print(input_text[:200])
    print("\nTarget:")
    print(target_text[:200])


if __name__ == '__main__':
    test_dataloader()
