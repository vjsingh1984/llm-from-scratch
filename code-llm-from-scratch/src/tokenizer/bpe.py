"""
Byte Pair Encoding (BPE) tokenizer implementation.

Based on the algorithm used in GPT-2, using byte-level BPE.
"""

import json
import regex as re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Tuple, Optional

from .vocab import Vocabulary


class BPETokenizer:
    """
    Byte Pair Encoding tokenizer.

    Implements byte-level BPE as used in GPT-2. Can handle any Unicode text
    by working at the byte level.
    """

    def __init__(self, vocab_size: int = 32000):
        """
        Initialize BPE tokenizer.

        Args:
            vocab_size: Target vocabulary size (including special tokens)
        """
        self.vocab_size = vocab_size
        self.vocab = Vocabulary()
        self.merges: List[Tuple[Tuple[str, str], str]] = []  # [(('a', 'b'), 'ab'), ...]
        self.merge_ranks: Dict[Tuple[str, str], int] = {}  # {('a', 'b'): 0, ...}

        # Regex pattern for pre-tokenization (split on whitespace and punctuation)
        # This pattern is from GPT-2
        self.pattern = re.compile(
            r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""",
            re.IGNORECASE
        )

        # Byte to unicode mapping for byte-level BPE
        self.byte_encoder = self._bytes_to_unicode()
        self.byte_decoder = {v: k for k, v in self.byte_encoder.items()}

    @staticmethod
    def _bytes_to_unicode() -> Dict[int, str]:
        """
        Create mapping from bytes to unicode strings.

        Returns printable characters for bytes 0-255, avoiding whitespace
        and control characters that might cause issues.

        Returns:
            Dictionary mapping byte values to unicode characters
        """
        # Printable ASCII characters
        bs = list(range(ord("!"), ord("~") + 1)) + \
             list(range(ord("¡"), ord("¬") + 1)) + \
             list(range(ord("®"), ord("ÿ") + 1))

        cs = bs[:]
        n = 0

        # For non-printable bytes, map to unused unicode characters
        for b in range(2**8):
            if b not in bs:
                bs.append(b)
                cs.append(2**8 + n)
                n += 1

        cs = [chr(c) for c in cs]
        return dict(zip(bs, cs))

    def _pre_tokenize(self, text: str) -> List[str]:
        """
        Split text into words/tokens before BPE.

        Uses regex to split on whitespace and punctuation boundaries.

        Args:
            text: Input text

        Returns:
            List of pre-tokenized segments
        """
        return re.findall(self.pattern, text)

    def _get_pair_counts(
        self, word_splits: Dict[str, List[str]], word_freqs: Dict[str, int]
    ) -> Counter:
        """
        Count frequency of adjacent token pairs.

        Args:
            word_splits: Dictionary mapping words to their current splits
            word_freqs: Frequency of each word in training data

        Returns:
            Counter of pair frequencies
        """
        pairs = Counter()
        for word, freq in word_freqs.items():
            splits = word_splits[word]
            if len(splits) < 2:
                continue
            for i in range(len(splits) - 1):
                pair = (splits[i], splits[i + 1])
                pairs[pair] += freq
        return pairs

    def _merge_pair(
        self, pair: Tuple[str, str], word_splits: Dict[str, List[str]]
    ) -> Dict[str, List[str]]:
        """
        Merge a pair in all word splits.

        Args:
            pair: Pair of tokens to merge
            word_splits: Current word splits

        Returns:
            Updated word splits
        """
        new_word_splits = {}
        merged_token = ''.join(pair)

        for word, splits in word_splits.items():
            new_splits = []
            i = 0
            while i < len(splits):
                if i < len(splits) - 1 and (splits[i], splits[i + 1]) == pair:
                    new_splits.append(merged_token)
                    i += 2
                else:
                    new_splits.append(splits[i])
                    i += 1
            new_word_splits[word] = new_splits

        return new_word_splits

    def train(self, texts: List[str], verbose: bool = True) -> None:
        """
        Train BPE tokenizer on corpus.

        Args:
            texts: List of training texts
            verbose: Print progress information
        """
        if verbose:
            print(f"Training BPE tokenizer with vocab size {self.vocab_size}")

        # Step 1: Pre-tokenize all text and count word frequencies
        word_freqs = Counter()
        for text in texts:
            words = self._pre_tokenize(text)
            for word in words:
                # Convert to byte-level representation
                byte_word = ''.join(self.byte_encoder[b] for b in word.encode('utf-8'))
                word_freqs[byte_word] += 1

        if verbose:
            print(f"Found {len(word_freqs)} unique words/tokens")

        # Step 2: Initialize splits with individual characters
        word_splits = {word: list(word) for word in word_freqs.keys()}

        # Step 3: Build base vocabulary from all characters
        base_vocab = set()
        for word in word_freqs.keys():
            base_vocab.update(word)

        # Add base vocabulary to our vocabulary
        for token in sorted(base_vocab):
            self.vocab.add_token(token)

        if verbose:
            print(f"Base vocabulary size: {len(self.vocab)} (includes special tokens)")

        # Step 4: Iteratively merge most frequent pairs
        num_merges = self.vocab_size - len(self.vocab)

        for i in range(num_merges):
            # Count pairs
            pair_freqs = self._get_pair_counts(word_splits, word_freqs)

            if not pair_freqs:
                if verbose:
                    print(f"No more pairs to merge at iteration {i}")
                break

            # Find most frequent pair
            best_pair = max(pair_freqs, key=pair_freqs.get)
            merged_token = ''.join(best_pair)

            # Save merge rule
            self.merges.append((best_pair, merged_token))
            self.merge_ranks[best_pair] = len(self.merges) - 1

            # Add to vocabulary
            self.vocab.add_token(merged_token)

            # Update word splits
            word_splits = self._merge_pair(best_pair, word_splits)

            if verbose and (i + 1) % 500 == 0:
                print(f"Merge {i+1}/{num_merges}: {best_pair} -> {merged_token} "
                      f"(freq: {pair_freqs[best_pair]})")

        if verbose:
            print(f"\nTraining complete! Final vocabulary size: {len(self.vocab)}")

    def _encode_word(self, word: str) -> List[str]:
        """
        Encode a single word using learned BPE merges.

        Args:
            word: Word to encode (already in byte-level representation)

        Returns:
            List of token strings
        """
        if len(word) == 0:
            return []

        # Start with character-level splits
        tokens = list(word)

        # Apply merges in order
        while len(tokens) > 1:
            # Find the pair with the lowest merge rank (earliest merge)
            pairs = [(tokens[i], tokens[i + 1]) for i in range(len(tokens) - 1)]
            pair_ranks = {
                pair: self.merge_ranks.get(pair, float('inf'))
                for pair in pairs
            }

            # If no valid merges, we're done
            if min(pair_ranks.values()) == float('inf'):
                break

            # Get the best pair to merge
            best_pair = min(pair_ranks, key=pair_ranks.get)

            # Merge the pair
            merged_token = ''.join(best_pair)
            new_tokens = []
            i = 0
            while i < len(tokens):
                if i < len(tokens) - 1 and (tokens[i], tokens[i + 1]) == best_pair:
                    new_tokens.append(merged_token)
                    i += 2
                else:
                    new_tokens.append(tokens[i])
                    i += 1
            tokens = new_tokens

        return tokens

    def encode(self, text: str, add_special_tokens: bool = False) -> List[int]:
        """
        Encode text to token IDs.

        Args:
            text: Input text
            add_special_tokens: Whether to add BOS/EOS tokens

        Returns:
            List of token IDs
        """
        # Pre-tokenize
        words = self._pre_tokenize(text)

        # Encode each word
        token_ids = []

        if add_special_tokens:
            token_ids.append(self.vocab.bos_id)

        for word in words:
            # Convert to byte-level
            byte_word = ''.join(self.byte_encoder[b] for b in word.encode('utf-8'))

            # Apply BPE
            tokens = self._encode_word(byte_word)

            # Convert to IDs
            for token in tokens:
                token_ids.append(self.vocab.encode_token(token))

        if add_special_tokens:
            token_ids.append(self.vocab.eos_id)

        return token_ids

    def decode(self, token_ids: List[int], skip_special_tokens: bool = True) -> str:
        """
        Decode token IDs back to text.

        Args:
            token_ids: List of token IDs
            skip_special_tokens: Whether to skip special tokens in output

        Returns:
            Decoded text
        """
        # Convert IDs to tokens
        tokens = []
        for token_id in token_ids:
            token = self.vocab.decode_token(token_id)

            # Skip special tokens if requested
            if skip_special_tokens and token in [
                self.vocab.pad_token,
                self.vocab.bos_token,
                self.vocab.eos_token,
            ]:
                continue

            tokens.append(token)

        # Join tokens and convert from byte-level back to text
        byte_string = ''.join(tokens)

        # Decode from unicode representation back to bytes, then to text
        text = bytearray([self.byte_decoder[c] for c in byte_string]).decode(
            'utf-8', errors='replace'
        )

        return text

    def save(self, directory: Path) -> None:
        """
        Save tokenizer to directory.

        Args:
            directory: Directory to save tokenizer files
        """
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)

        # Save vocabulary
        self.vocab.save(directory / 'vocab.json')

        # Save merges
        merges_data = {
            'merges': [{'pair': list(pair), 'token': token} for pair, token in self.merges],
            'vocab_size': self.vocab_size,
        }
        with open(directory / 'merges.json', 'w', encoding='utf-8') as f:
            json.dump(merges_data, f, indent=2, ensure_ascii=False)

        print(f"Tokenizer saved to {directory}")

    @classmethod
    def load(cls, directory: Path) -> 'BPETokenizer':
        """
        Load tokenizer from directory.

        Args:
            directory: Directory containing tokenizer files

        Returns:
            Loaded tokenizer
        """
        directory = Path(directory)

        # Load vocabulary
        vocab = Vocabulary.load(directory / 'vocab.json')

        # Load merges
        with open(directory / 'merges.json', 'r', encoding='utf-8') as f:
            merges_data = json.load(f)

        # Create tokenizer instance
        tokenizer = cls(vocab_size=merges_data['vocab_size'])
        tokenizer.vocab = vocab
        tokenizer.merges = [
            (tuple(m['pair']), m['token']) for m in merges_data['merges']
        ]
        tokenizer.merge_ranks = {pair: i for i, (pair, _) in enumerate(tokenizer.merges)}

        print(f"Tokenizer loaded from {directory}")
        return tokenizer
