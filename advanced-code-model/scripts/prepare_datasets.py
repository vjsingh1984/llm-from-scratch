"""
Prepare and tokenize datasets for training.

This script:
1. Loads the trained BPE tokenizer
2. Tokenizes language and code data
3. Chunks into 4096-token sequences
4. Creates train/validation splits (95/5)
5. Saves as NumPy arrays for efficient training

Usage:
    python scripts/prepare_datasets.py
"""

import json
import numpy as np
from pathlib import Path
from typing import List, Tuple

from tokenizers import Tokenizer
from tqdm import tqdm


def load_tokenizer(tokenizer_path: Path) -> Tokenizer:
    """Load trained tokenizer."""
    return Tokenizer.from_file(str(tokenizer_path))


def tokenize_and_chunk(
    texts: List[str],
    tokenizer: Tokenizer,
    max_seq_len: int = 4096,
    concatenate: bool = True
) -> List[np.ndarray]:
    """
    Tokenize texts and chunk into fixed-length sequences.

    Args:
        texts: List of text strings
        tokenizer: Trained tokenizer
        max_seq_len: Maximum sequence length
        concatenate: If True, concatenate all texts before chunking

    Returns:
        List of tokenized sequences as numpy arrays
    """
    sequences = []

    if concatenate:
        # Concatenate all texts with separator
        print("Concatenating texts...")
        combined_text = " ".join(texts)

        # Tokenize combined text
        print("Tokenizing combined text...")
        encoded = tokenizer.encode(combined_text)
        all_tokens = encoded.ids

        print(f"Total tokens: {len(all_tokens):,}")

        # Chunk into fixed-length sequences
        print("Chunking into sequences...")
        for i in tqdm(range(0, len(all_tokens), max_seq_len), desc="Creating sequences"):
            chunk = all_tokens[i:i + max_seq_len]

            # Pad last chunk if needed
            if len(chunk) < max_seq_len:
                if len(chunk) >= max_seq_len // 4:  # Keep if at least 25% full
                    chunk = chunk + [0] * (max_seq_len - len(chunk))
                else:
                    break  # Skip very last tiny chunk

            sequences.append(np.array(chunk, dtype=np.int32))
    else:
        # Process each text separately
        for text in tqdm(texts, desc="Tokenizing"):
            encoded = tokenizer.encode(text)
            tokens = encoded.ids

            # Skip very short texts
            if len(tokens) < 10:
                continue

            # Pad or chunk as needed
            if len(tokens) <= max_seq_len:
                # Pad
                padded = tokens + [0] * (max_seq_len - len(tokens))
                sequences.append(np.array(padded, dtype=np.int32))
            else:
                # Chunk
                for i in range(0, len(tokens), max_seq_len):
                    chunk = tokens[i:i + max_seq_len]
                    if len(chunk) >= max_seq_len // 4:
                        if len(chunk) < max_seq_len:
                            chunk = chunk + [0] * (max_seq_len - len(chunk))
                        sequences.append(np.array(chunk, dtype=np.int32))

    return sequences


def prepare_language_dataset(
    language_dir: Path,
    tokenizer: Tokenizer,
    max_seq_len: int = 4096
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Prepare language dataset.

    Args:
        language_dir: Path to language data
        tokenizer: Trained tokenizer
        max_seq_len: Maximum sequence length

    Returns:
        Train and validation arrays
    """
    print("=" * 60)
    print("Preparing Language Dataset")
    print("=" * 60)
    print()

    # Load texts
    texts = []
    raw_dir = language_dir / "raw"
    batch_files = sorted(raw_dir.glob("batch_*.txt"))

    print(f"Loading {len(batch_files)} batch files...")

    for batch_file in tqdm(batch_files[:10], desc="Loading batches"):  # Use first 10 batches for demo
        with open(batch_file, "r", encoding="utf-8") as f:
            content = f.read()

        # Split into stories
        stories = content.strip().split("\n\n")
        texts.extend([s.strip() for s in stories if s.strip()])

    print(f"✓ Loaded {len(texts):,} stories")
    print()

    # Tokenize and chunk (concatenate all stories)
    sequences = tokenize_and_chunk(texts, tokenizer, max_seq_len, concatenate=True)

    print(f"✓ Created {len(sequences):,} sequences")
    print()

    # Convert to array
    all_data = np.array(sequences, dtype=np.int32)

    # Split train/val (95/5)
    n_train = int(len(all_data) * 0.95)

    train_data = all_data[:n_train]
    val_data = all_data[n_train:]

    print(f"Train sequences: {len(train_data):,}")
    print(f"Val sequences: {len(val_data):,}")
    print()

    return train_data, val_data


def prepare_code_dataset(
    code_dir: Path,
    tokenizer: Tokenizer,
    max_seq_len: int = 4096
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Prepare code dataset.

    Args:
        code_dir: Path to code data
        tokenizer: Trained tokenizer
        max_seq_len: Maximum sequence length

    Returns:
        Train and validation arrays
    """
    print("=" * 60)
    print("Preparing Code Dataset")
    print("=" * 60)
    print()

    # Load scripts
    texts = []
    scripts_dir = code_dir / "raw" / "scripts"
    script_files = sorted(scripts_dir.glob("*.sh"))

    print(f"Loading {len(script_files)} bash scripts...")

    for script_file in tqdm(script_files, desc="Loading scripts"):
        with open(script_file, "r", encoding="utf-8", errors="ignore") as f:
            content = f.read()

        if content.strip():
            texts.append(content.strip())

    print(f"✓ Loaded {len(texts):,} scripts")
    print()

    # Tokenize and chunk (keep scripts separate)
    sequences = tokenize_and_chunk(texts, tokenizer, max_seq_len, concatenate=False)

    print(f"✓ Created {len(sequences):,} sequences")
    print()

    # Convert to array
    all_data = np.array(sequences, dtype=np.int32)

    # Split train/val (95/5)
    n_train = int(len(all_data) * 0.95)

    train_data = all_data[:n_train]
    val_data = all_data[n_train:]

    print(f"Train sequences: {len(train_data):,}")
    print(f"Val sequences: {len(val_data):,}")
    print()

    return train_data, val_data


def main():
    """Main entry point."""
    # Paths
    base_dir = Path(__file__).parent.parent
    tokenizer_path = base_dir / "data" / "tokenizer" / "tokenizer.json"
    language_dir = base_dir / "data" / "language"
    code_dir = base_dir / "data" / "bash"
    output_dir = base_dir / "data" / "processed"

    output_dir.mkdir(parents=True, exist_ok=True)

    # Load tokenizer
    print("Loading tokenizer...")
    tokenizer = load_tokenizer(tokenizer_path)
    print(f"✓ Loaded tokenizer (vocab size: {tokenizer.get_vocab_size():,})")
    print()

    # Prepare language dataset
    lang_train, lang_val = prepare_language_dataset(
        language_dir,
        tokenizer,
        max_seq_len=1024
    )

    # Save language data
    lang_train_path = output_dir / "language_train.npy"
    lang_val_path = output_dir / "language_val.npy"

    np.save(lang_train_path, lang_train)
    np.save(lang_val_path, lang_val)

    print(f"✓ Saved language train: {lang_train_path}")
    print(f"✓ Saved language val: {lang_val_path}")
    print()

    # Prepare code dataset
    code_train, code_val = prepare_code_dataset(
        code_dir,
        tokenizer,
        max_seq_len=1024
    )

    # Save code data
    code_train_path = output_dir / "code_train.npy"
    code_val_path = output_dir / "code_val.npy"

    np.save(code_train_path, code_train)
    np.save(code_val_path, code_val)

    print(f"✓ Saved code train: {code_train_path}")
    print(f"✓ Saved code val: {code_val_path}")
    print()

    # Save metadata
    metadata = {
        "max_seq_len": 4096,
        "vocab_size": tokenizer.get_vocab_size(),
        "language": {
            "train_sequences": int(len(lang_train)),
            "val_sequences": int(len(lang_val)),
            "train_tokens": int(len(lang_train) * 4096),
            "val_tokens": int(len(lang_val) * 4096),
        },
        "code": {
            "train_sequences": int(len(code_train)),
            "val_sequences": int(len(code_val)),
            "train_tokens": int(len(code_train) * 4096),
            "val_tokens": int(len(code_val) * 4096),
        }
    }

    metadata_path = output_dir / "metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"✓ Saved metadata: {metadata_path}")
    print()

    # Print summary
    print("=" * 60)
    print("Dataset Preparation Complete!")
    print("=" * 60)
    print()
    print("Language Dataset:")
    print(f"  Train: {len(lang_train):,} sequences ({len(lang_train) * 4096:,} tokens)")
    print(f"  Val:   {len(lang_val):,} sequences ({len(lang_val) * 4096:,} tokens)")
    print()
    print("Code Dataset:")
    print(f"  Train: {len(code_train):,} sequences ({len(code_train) * 4096:,} tokens)")
    print(f"  Val:   {len(code_val):,} sequences ({len(code_val) * 4096:,} tokens)")
    print()
    print("Ready for training!")


if __name__ == "__main__":
    main()
