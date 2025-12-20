"""
Code-specific tokenizer for bash scripts.

Uses character-level tokenization with special handling for:
- Whitespace (preserve indentation)
- Special bash characters
- Common patterns
"""

import json
from pathlib import Path
from typing import List, Dict, Optional
import re


class CodeTokenizer:
    """
    Character-level tokenizer optimized for code.

    Why character-level for code?
    - No unknown tokens (can represent any code)
    - Preserves exact syntax
    - Simple and reliable
    - Works with any programming language
    """

    def __init__(self):
        # Special tokens
        self.pad_token = "<PAD>"
        self.bos_token = "<BOS>"  # Begin of sequence
        self.eos_token = "<EOS>"  # End of sequence
        self.unk_token = "<UNK>"  # Unknown (shouldn't be used)

        # Core vocabulary
        self.vocab: Dict[str, int] = {}
        self.id_to_token: Dict[int, str] = {}

        # Special token IDs
        self.pad_id = 0
        self.bos_id = 1
        self.eos_id = 2
        self.unk_id = 3

        # Initialize with special tokens
        self._add_special_tokens()

    def _add_special_tokens(self):
        """Add special tokens to vocabulary."""
        special_tokens = [
            self.pad_token,
            self.bos_token,
            self.eos_token,
            self.unk_token,
        ]

        for idx, token in enumerate(special_tokens):
            self.vocab[token] = idx
            self.id_to_token[idx] = token

    def build_vocab(self, texts: List[str], max_vocab_size: int = 512):
        """
        Build vocabulary from code samples.

        For code, we typically use:
        - All ASCII printable characters
        - Common special characters
        - Whitespace characters

        Args:
            texts: List of code samples
            max_vocab_size: Maximum vocabulary size
        """
        # Collect all unique characters
        char_freq = {}
        for text in texts:
            for char in text:
                char_freq[char] = char_freq.get(char, 0) + 1

        # Sort by frequency
        sorted_chars = sorted(char_freq.items(), key=lambda x: x[1], reverse=True)

        # Add to vocabulary (after special tokens)
        next_id = len(self.vocab)
        for char, freq in sorted_chars:
            if char not in self.vocab:
                self.vocab[char] = next_id
                self.id_to_token[next_id] = char
                next_id += 1

            if next_id >= max_vocab_size:
                break

        print(f"Vocabulary built: {len(self.vocab)} tokens")
        print(f"  Special tokens: 4")
        print(f"  Characters: {len(self.vocab) - 4}")

    def build_default_vocab(self):
        """
        Build a default vocabulary with common code characters.

        This covers:
        - ASCII letters (a-z, A-Z)
        - Digits (0-9)
        - Common punctuation and operators
        - Whitespace (space, tab, newline)
        """
        # Start after special tokens
        next_id = len(self.vocab)

        # Common characters in code
        common_chars = []

        # Whitespace
        common_chars.extend([' ', '\n', '\t', '\r'])

        # Letters
        common_chars.extend(chr(i) for i in range(ord('a'), ord('z') + 1))
        common_chars.extend(chr(i) for i in range(ord('A'), ord('Z') + 1))

        # Digits
        common_chars.extend(str(i) for i in range(10))

        # Common punctuation and operators
        common_chars.extend([
            '!', '"', '#', '$', '%', '&', "'", '(', ')', '*', '+', ',',
            '-', '.', '/', ':', ';', '<', '=', '>', '?', '@', '[', '\\',
            ']', '^', '_', '`', '{', '|', '}', '~'
        ])

        # Add to vocabulary
        for char in common_chars:
            if char not in self.vocab:
                self.vocab[char] = next_id
                self.id_to_token[next_id] = char
                next_id += 1

        print(f"Default vocabulary built: {len(self.vocab)} tokens")

    def encode(
        self,
        text: str,
        add_special_tokens: bool = True,
        max_length: Optional[int] = None
    ) -> List[int]:
        """
        Encode text to token IDs.

        Args:
            text: Input text (code)
            add_special_tokens: Add BOS/EOS tokens
            max_length: Maximum length (truncate if longer)

        Returns:
            List of token IDs
        """
        # Encode character by character
        token_ids = []

        if add_special_tokens:
            token_ids.append(self.bos_id)

        for char in text:
            token_id = self.vocab.get(char, self.unk_id)
            token_ids.append(token_id)

        if add_special_tokens:
            token_ids.append(self.eos_id)

        # Truncate if needed
        if max_length is not None and len(token_ids) > max_length:
            if add_special_tokens:
                # Keep BOS, truncate middle, add EOS
                token_ids = token_ids[:max_length-1] + [self.eos_id]
            else:
                token_ids = token_ids[:max_length]

        return token_ids

    def decode(
        self,
        token_ids: List[int],
        skip_special_tokens: bool = True
    ) -> str:
        """
        Decode token IDs back to text.

        Args:
            token_ids: List of token IDs
            skip_special_tokens: Skip special tokens in output

        Returns:
            Decoded text
        """
        chars = []

        for token_id in token_ids:
            token = self.id_to_token.get(token_id, self.unk_token)

            # Skip special tokens if requested
            if skip_special_tokens and token in [
                self.pad_token, self.bos_token, self.eos_token, self.unk_token
            ]:
                continue

            chars.append(token)

        return ''.join(chars)

    def save(self, directory: Path):
        """Save tokenizer to directory."""
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)

        # Save vocabulary
        vocab_path = directory / 'vocab.json'
        with open(vocab_path, 'w', encoding='utf-8') as f:
            json.dump({
                'vocab': self.vocab,
                'special_tokens': {
                    'pad': self.pad_token,
                    'bos': self.bos_token,
                    'eos': self.eos_token,
                    'unk': self.unk_token,
                }
            }, f, indent=2, ensure_ascii=False)

        print(f"Tokenizer saved to {directory}")

    @classmethod
    def load(cls, directory: Path) -> 'CodeTokenizer':
        """Load tokenizer from directory."""
        directory = Path(directory)
        vocab_path = directory / 'vocab.json'

        with open(vocab_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        tokenizer = cls()
        tokenizer.vocab = {k: int(v) for k, v in data['vocab'].items()}
        tokenizer.id_to_token = {int(v): k for k, v in tokenizer.vocab.items()}

        # Load special tokens
        special = data.get('special_tokens', {})
        tokenizer.pad_token = special.get('pad', '<PAD>')
        tokenizer.bos_token = special.get('bos', '<BOS>')
        tokenizer.eos_token = special.get('eos', '<EOS>')
        tokenizer.unk_token = special.get('unk', '<UNK>')

        tokenizer.pad_id = tokenizer.vocab[tokenizer.pad_token]
        tokenizer.bos_id = tokenizer.vocab[tokenizer.bos_token]
        tokenizer.eos_id = tokenizer.vocab[tokenizer.eos_token]
        tokenizer.unk_id = tokenizer.vocab[tokenizer.unk_token]

        print(f"Tokenizer loaded from {directory}")
        return tokenizer

    def __len__(self) -> int:
        """Return vocabulary size."""
        return len(self.vocab)

    def analyze_code(self, code: str) -> Dict:
        """
        Analyze code sample.

        Returns statistics about the code.
        """
        token_ids = self.encode(code, add_special_tokens=False)

        return {
            'num_chars': len(code),
            'num_tokens': len(token_ids),
            'num_lines': code.count('\n') + 1,
            'compression_ratio': len(token_ids) / len(code) if code else 0,
            'vocab_coverage': len(set(token_ids)) / len(self.vocab) if self.vocab else 0,
        }


def create_bash_tokenizer(bash_scripts: Optional[List[str]] = None) -> CodeTokenizer:
    """
    Create a tokenizer optimized for bash scripts.

    Args:
        bash_scripts: Optional list of bash scripts to build vocabulary from

    Returns:
        Configured CodeTokenizer
    """
    tokenizer = CodeTokenizer()

    if bash_scripts:
        # Build vocabulary from actual scripts
        tokenizer.build_vocab(bash_scripts, max_vocab_size=256)
    else:
        # Use default vocabulary
        tokenizer.build_default_vocab()

    return tokenizer


# Testing helper
def test_tokenizer():
    """Test the tokenizer with sample bash code."""
    print("Testing Code Tokenizer\n" + "="*60)

    # Create tokenizer
    tokenizer = CodeTokenizer()
    tokenizer.build_default_vocab()

    # Sample bash code
    bash_code = """#!/bin/bash
# List all files
ls -la | grep ".txt"
echo "Done!"
"""

    print(f"\nOriginal code:")
    print(bash_code)

    # Encode
    token_ids = tokenizer.encode(bash_code)
    print(f"\nEncoded: {len(token_ids)} tokens")
    print(f"First 20 tokens: {token_ids[:20]}")

    # Decode
    decoded = tokenizer.decode(token_ids)
    print(f"\nDecoded:")
    print(decoded)

    # Check match
    print(f"\nMatch: {'✓' if bash_code == decoded else '✗'}")

    # Analyze
    stats = tokenizer.analyze_code(bash_code)
    print(f"\nCode statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")


if __name__ == '__main__':
    test_tokenizer()
