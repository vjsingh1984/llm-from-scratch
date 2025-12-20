"""
Vocabulary management for tokenizer.

Handles bidirectional mapping between tokens and IDs.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional


class Vocabulary:
    """
    Manages the vocabulary for tokenization.

    Provides bidirectional mapping between tokens (strings) and their IDs (integers).
    Handles special tokens like <|endoftext|>, <|pad|>, etc.
    """

    def __init__(self):
        """Initialize empty vocabulary."""
        self.token_to_id: Dict[str, int] = {}
        self.id_to_token: Dict[int, str] = {}
        self._next_id = 0

        # Add special tokens
        self.pad_token = "<|pad|>"
        self.eos_token = "<|endoftext|>"
        self.unk_token = "<|unk|>"
        self.bos_token = "<|bos|>"

        # Reserve IDs for special tokens
        self.pad_id = self.add_token(self.pad_token)
        self.eos_id = self.add_token(self.eos_token)
        self.unk_id = self.add_token(self.unk_token)
        self.bos_id = self.add_token(self.bos_token)

    def add_token(self, token: str) -> int:
        """
        Add a token to the vocabulary.

        Args:
            token: String token to add

        Returns:
            ID assigned to the token
        """
        if token in self.token_to_id:
            return self.token_to_id[token]

        token_id = self._next_id
        self.token_to_id[token] = token_id
        self.id_to_token[token_id] = token
        self._next_id += 1
        return token_id

    def add_tokens(self, tokens: List[str]) -> List[int]:
        """
        Add multiple tokens to vocabulary.

        Args:
            tokens: List of tokens to add

        Returns:
            List of assigned IDs
        """
        return [self.add_token(token) for token in tokens]

    def encode_token(self, token: str) -> int:
        """
        Convert token to ID.

        Args:
            token: Token string

        Returns:
            Token ID, or unk_id if token not in vocabulary
        """
        return self.token_to_id.get(token, self.unk_id)

    def decode_token(self, token_id: int) -> str:
        """
        Convert ID to token.

        Args:
            token_id: Token ID

        Returns:
            Token string, or unk_token if ID not found

        Raises:
            KeyError: If token_id not in vocabulary
        """
        if token_id not in self.id_to_token:
            return self.unk_token
        return self.id_to_token[token_id]

    def __len__(self) -> int:
        """Return vocabulary size."""
        return len(self.token_to_id)

    def __contains__(self, token: str) -> bool:
        """Check if token is in vocabulary."""
        return token in self.token_to_id

    def save(self, path: Path):
        """
        Save vocabulary to JSON file.

        Args:
            path: Path to save vocabulary
        """
        vocab_data = {
            'token_to_id': self.token_to_id,
            'special_tokens': {
                'pad': self.pad_token,
                'eos': self.eos_token,
                'unk': self.unk_token,
                'bos': self.bos_token,
            }
        }
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(vocab_data, f, indent=2, ensure_ascii=False)

    @classmethod
    def load(cls, path: Path) -> 'Vocabulary':
        """
        Load vocabulary from JSON file.

        Args:
            path: Path to vocabulary file

        Returns:
            Loaded Vocabulary instance
        """
        with open(path, 'r', encoding='utf-8') as f:
            vocab_data = json.load(f)

        vocab = cls.__new__(cls)  # Create instance without calling __init__
        vocab.token_to_id = vocab_data['token_to_id']
        vocab.id_to_token = {int(k): v for k, v in vocab_data.get('id_to_token', {}).items()}

        # Rebuild id_to_token if not saved
        if not vocab.id_to_token:
            vocab.id_to_token = {v: k for k, v in vocab.token_to_id.items()}

        # Load special tokens
        special = vocab_data.get('special_tokens', {})
        vocab.pad_token = special.get('pad', '<|pad|>')
        vocab.eos_token = special.get('eos', '<|endoftext|>')
        vocab.unk_token = special.get('unk', '<|unk|>')
        vocab.bos_token = special.get('bos', '<|bos|>')

        vocab.pad_id = vocab.token_to_id[vocab.pad_token]
        vocab.eos_id = vocab.token_to_id[vocab.eos_token]
        vocab.unk_id = vocab.token_to_id[vocab.unk_token]
        vocab.bos_id = vocab.token_to_id[vocab.bos_token]

        vocab._next_id = max(vocab.id_to_token.keys()) + 1 if vocab.id_to_token else 0

        return vocab

    def get_token_id(self, token: str, default: Optional[int] = None) -> Optional[int]:
        """
        Get token ID with optional default.

        Args:
            token: Token string
            default: Default value if token not found

        Returns:
            Token ID or default
        """
        return self.token_to_id.get(token, default)
