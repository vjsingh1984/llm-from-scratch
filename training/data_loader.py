"""
Data loading utilities for language model training.

Handles loading text data, creating batches, and preprocessing.
"""

import mlx.core as mx
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional, Iterator
import random


class TextDataset:
    """
    Dataset for language modeling.

    Loads tokenized text and creates sequences for next-token prediction.

    Args:
        token_ids: List or array of token IDs
        seq_len: Sequence length for training
        stride: Stride for creating sequences (default: seq_len for no overlap)
    """

    def __init__(
        self,
        token_ids: List[int],
        seq_len: int,
        stride: Optional[int] = None
    ):
        self.token_ids = mx.array(token_ids) if not isinstance(token_ids, mx.array) else token_ids
        self.seq_len = seq_len
        self.stride = stride if stride is not None else seq_len

        # Calculate number of sequences
        # We need seq_len + 1 tokens (input + target)
        total_tokens = len(self.token_ids)
        self.num_sequences = max(0, (total_tokens - self.seq_len - 1) // self.stride + 1)

    def __len__(self) -> int:
        """Return number of sequences."""
        return self.num_sequences

    def __getitem__(self, idx: int) -> Tuple[mx.array, mx.array]:
        """
        Get a sequence and its target.

        Args:
            idx: Sequence index

        Returns:
            Tuple of (input_seq, target_seq)
            - input_seq: [seq_len] token IDs
            - target_seq: [seq_len] token IDs (shifted by 1)
        """
        start = idx * self.stride
        end = start + self.seq_len + 1

        # Get sequence
        seq = self.token_ids[start:end]

        # Input is all but last token, target is all but first token
        input_seq = seq[:-1]
        target_seq = seq[1:]

        return input_seq, target_seq


class DataLoader:
    """
    Batched data loader for text dataset.

    Args:
        dataset: TextDataset instance
        batch_size: Batch size
        shuffle: Whether to shuffle data
        drop_last: Whether to drop last incomplete batch
    """

    def __init__(
        self,
        dataset: TextDataset,
        batch_size: int,
        shuffle: bool = True,
        drop_last: bool = False
    ):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last

        # Create index list
        self.indices = list(range(len(dataset)))

    def __len__(self) -> int:
        """Return number of batches."""
        if self.drop_last:
            return len(self.dataset) // self.batch_size
        else:
            return (len(self.dataset) + self.batch_size - 1) // self.batch_size

    def __iter__(self) -> Iterator[Tuple[mx.array, mx.array]]:
        """Iterate over batches."""
        # Shuffle indices if requested
        if self.shuffle:
            random.shuffle(self.indices)

        # Yield batches
        for i in range(0, len(self.indices), self.batch_size):
            batch_indices = self.indices[i:i + self.batch_size]

            # Skip incomplete batch if drop_last
            if self.drop_last and len(batch_indices) < self.batch_size:
                continue

            # Get sequences for this batch
            inputs = []
            targets = []

            for idx in batch_indices:
                input_seq, target_seq = self.dataset[idx]
                inputs.append(input_seq)
                targets.append(target_seq)

            # Stack into batch
            # Shape: [batch_size, seq_len]
            inputs = mx.stack(inputs)
            targets = mx.stack(targets)

            yield inputs, targets


def load_tokens_from_file(file_path: Path, tokenizer) -> List[int]:
    """
    Load text from file and tokenize.

    Args:
        file_path: Path to text file
        tokenizer: Tokenizer instance with encode method

    Returns:
        List of token IDs
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read()

    # Tokenize
    token_ids = tokenizer.encode(text, add_special_tokens=False)

    return token_ids


def create_data_loader(
    file_path: Path,
    tokenizer,
    seq_len: int,
    batch_size: int,
    shuffle: bool = True,
    stride: Optional[int] = None
) -> DataLoader:
    """
    Create data loader from text file.

    Args:
        file_path: Path to text file
        tokenizer: Tokenizer instance
        seq_len: Sequence length
        batch_size: Batch size
        shuffle: Whether to shuffle
        stride: Stride for creating sequences

    Returns:
        DataLoader instance
    """
    # Load and tokenize
    token_ids = load_tokens_from_file(file_path, tokenizer)

    # Create dataset
    dataset = TextDataset(token_ids, seq_len, stride)

    # Create data loader
    loader = DataLoader(dataset, batch_size, shuffle)

    return loader


def create_train_val_loaders(
    train_file: Path,
    val_file: Path,
    tokenizer,
    seq_len: int,
    batch_size: int,
    val_batch_size: Optional[int] = None
) -> Tuple[DataLoader, DataLoader]:
    """
    Create training and validation data loaders.

    Args:
        train_file: Path to training data
        val_file: Path to validation data
        tokenizer: Tokenizer instance
        seq_len: Sequence length
        batch_size: Training batch size
        val_batch_size: Validation batch size (default: same as batch_size)

    Returns:
        Tuple of (train_loader, val_loader)
    """
    if val_batch_size is None:
        val_batch_size = batch_size

    # Create loaders
    train_loader = create_data_loader(
        train_file,
        tokenizer,
        seq_len,
        batch_size,
        shuffle=True
    )

    val_loader = create_data_loader(
        val_file,
        tokenizer,
        seq_len,
        val_batch_size,
        shuffle=False
    )

    return train_loader, val_loader


def get_batch_stats(batch: Tuple[mx.array, mx.array]) -> dict:
    """
    Get statistics about a batch.

    Args:
        batch: Tuple of (inputs, targets)

    Returns:
        Dictionary with batch statistics
    """
    inputs, targets = batch

    return {
        'batch_size': inputs.shape[0],
        'seq_len': inputs.shape[1],
        'num_tokens': inputs.size,
        'input_shape': inputs.shape,
        'target_shape': targets.shape,
        'min_token_id': int(inputs.min()),
        'max_token_id': int(inputs.max()),
    }


# Utility function for creating sample data
def create_sample_data(
    num_tokens: int,
    vocab_size: int,
    output_file: Path
):
    """
    Create random sample data for testing.

    Args:
        num_tokens: Number of tokens to generate
        vocab_size: Vocabulary size
        output_file: Path to save data
    """
    # Generate random token IDs
    tokens = np.random.randint(0, vocab_size, size=num_tokens)

    # Save to file (simple format: space-separated IDs)
    with open(output_file, 'w') as f:
        f.write(' '.join(map(str, tokens)))

    print(f"Created sample data: {output_file}")
    print(f"  Tokens: {num_tokens}")
    print(f"  Vocabulary size: {vocab_size}")
