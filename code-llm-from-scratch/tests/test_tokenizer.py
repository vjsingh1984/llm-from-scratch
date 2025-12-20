"""
Tests for the BPE tokenizer.

Tests cover:
- Basic encode/decode functionality
- Vocabulary training
- Save/load operations
- Edge cases and error handling
"""

import os
import pytest
from src.tokenizer.bpe import BPETokenizer


class TestBPETokenizer:
    """Test suite for BPE tokenizer."""

    def test_init(self):
        """Test tokenizer initialization."""
        tokenizer = BPETokenizer()
        assert hasattr(tokenizer, 'vocab')
        assert hasattr(tokenizer, 'merges')

    def test_train_basic(self, sample_texts):
        """Test basic training on sample texts."""
        tokenizer = BPETokenizer()
        tokenizer.target_vocab_size = 200
        tokenizer.train(sample_texts, verbose=False)

        assert len(tokenizer.vocab) > 0, "Vocabulary should not be empty"
        assert len(tokenizer.merges) > 0, "Should have some merge operations"

    def test_encode_decode_identity(self, small_vocab_tokenizer):
        """Test that decode(encode(text)) == text."""
        tokenizer = small_vocab_tokenizer
        test_text = "#!/bin/bash\necho 'test'"

        encoded = tokenizer.encode(test_text)
        decoded = tokenizer.decode(encoded)

        assert decoded == test_text, "Decoded text should match original"

    def test_encode_returns_list(self, small_vocab_tokenizer):
        """Test that encode returns a list of integers."""
        tokenizer = small_vocab_tokenizer
        encoded = tokenizer.encode("test")

        assert isinstance(encoded, list), "Encode should return a list"
        assert all(isinstance(x, int) for x in encoded), "All tokens should be integers"

    def test_decode_returns_string(self, small_vocab_tokenizer):
        """Test that decode returns a string."""
        tokenizer = small_vocab_tokenizer
        encoded = tokenizer.encode("test")
        decoded = tokenizer.decode(encoded)

        assert isinstance(decoded, str), "Decode should return a string"

    def test_empty_string_encoding(self, small_vocab_tokenizer):
        """Test encoding of empty string."""
        tokenizer = small_vocab_tokenizer
        encoded = tokenizer.encode("")

        assert isinstance(encoded, list), "Should return a list"
        assert len(encoded) == 0, "Empty string should encode to empty list"

    def test_empty_list_decoding(self, small_vocab_tokenizer):
        """Test decoding of empty list."""
        tokenizer = small_vocab_tokenizer
        decoded = tokenizer.decode([])

        assert decoded == "", "Empty list should decode to empty string"

    def test_special_characters(self, small_vocab_tokenizer):
        """Test encoding/decoding of special characters."""
        tokenizer = small_vocab_tokenizer
        special_chars = "!@#$%^&*()\n\t\r"

        encoded = tokenizer.encode(special_chars)
        decoded = tokenizer.decode(encoded)

        assert decoded == special_chars, "Special characters should roundtrip correctly"

    def test_save_and_load(self, small_vocab_tokenizer, temp_dir):
        """Test saving and loading tokenizer."""
        tokenizer = small_vocab_tokenizer
        save_path = os.path.join(temp_dir, "tokenizer.json")

        # Save
        tokenizer.save(save_path)
        assert os.path.exists(save_path), "Tokenizer file should exist"

        # Load
        loaded_tokenizer = BPETokenizer()
        loaded_tokenizer.load(save_path)

        # Compare
        test_text = "#!/bin/bash\necho test"
        original_encoded = tokenizer.encode(test_text)
        loaded_encoded = loaded_tokenizer.encode(test_text)

        assert original_encoded == loaded_encoded, "Loaded tokenizer should produce same encoding"
        assert len(loaded_tokenizer.vocab) == len(tokenizer.vocab), "Vocab size should match"

    def test_consistent_encoding(self, small_vocab_tokenizer):
        """Test that encoding the same text produces same result."""
        tokenizer = small_vocab_tokenizer
        text = "test string"

        encoded1 = tokenizer.encode(text)
        encoded2 = tokenizer.encode(text)

        assert encoded1 == encoded2, "Encoding should be deterministic"

    def test_different_texts_different_encodings(self, small_vocab_tokenizer):
        """Test that different texts produce different encodings."""
        tokenizer = small_vocab_tokenizer

        encoded1 = tokenizer.encode("hello")
        encoded2 = tokenizer.encode("world")

        assert encoded1 != encoded2, "Different texts should have different encodings"

    def test_longer_text(self, small_vocab_tokenizer):
        """Test encoding/decoding of longer text."""
        tokenizer = small_vocab_tokenizer
        long_text = """#!/bin/bash
# This is a longer bash script
for i in {1..100}; do
    echo "Processing item $i"
    sleep 1
done
echo "Done!"
"""

        encoded = tokenizer.encode(long_text)
        decoded = tokenizer.decode(encoded)

        assert decoded == long_text, "Long text should roundtrip correctly"
        assert len(encoded) > 0, "Should produce non-empty encoding"

    def test_unicode_handling(self, small_vocab_tokenizer):
        """Test handling of unicode characters."""
        tokenizer = small_vocab_tokenizer
        unicode_text = "echo 'Hello 世界 🌍'"

        encoded = tokenizer.encode(unicode_text)
        decoded = tokenizer.decode(encoded)

        assert decoded == unicode_text, "Unicode should roundtrip correctly"

    def test_vocab_size_approximately_correct(self, sample_texts):
        """Test that final vocab size is close to target."""
        tokenizer = BPETokenizer()
        target_size = 200
        tokenizer.target_vocab_size = target_size
        tokenizer.train(sample_texts, verbose=False)

        # Allow some tolerance (within 20%)
        actual_size = len(tokenizer.vocab)
        assert actual_size >= target_size * 0.8, f"Vocab too small: {actual_size} < {target_size * 0.8}"
        assert actual_size <= target_size * 1.2, f"Vocab too large: {actual_size} > {target_size * 1.2}"

    def test_train_with_single_text(self):
        """Test training with a single text."""
        tokenizer = BPETokenizer()
        tokenizer.target_vocab_size = 100
        tokenizer.train(["single text"], verbose=False)

        assert len(tokenizer.vocab) > 0, "Should create vocabulary from single text"

    def test_decode_invalid_token_ids(self, small_vocab_tokenizer):
        """Test decoding with some invalid token IDs."""
        tokenizer = small_vocab_tokenizer

        # Get valid tokens
        valid_tokens = tokenizer.encode("test")

        # Add an invalid token ID (very large number)
        invalid_tokens = valid_tokens + [999999]

        # Should handle gracefully (might skip invalid tokens or raise error)
        try:
            decoded = tokenizer.decode(invalid_tokens)
            # If it doesn't raise an error, that's fine too
            assert isinstance(decoded, str)
        except (KeyError, IndexError, ValueError):
            # Expected behavior for invalid tokens
            pass

    def test_encode_after_load(self, small_vocab_tokenizer, temp_dir):
        """Test that encoding works correctly after save/load cycle."""
        original_tokenizer = small_vocab_tokenizer
        save_path = os.path.join(temp_dir, "test_tokenizer.json")

        # Save
        original_tokenizer.save(save_path)

        # Load into new tokenizer
        new_tokenizer = BPETokenizer()
        new_tokenizer.load(save_path)

        # Test multiple texts
        test_texts = [
            "#!/bin/bash",
            "echo test",
            "for i in {1..10}; do echo $i; done"
        ]

        for text in test_texts:
            original_encoded = original_tokenizer.encode(text)
            new_encoded = new_tokenizer.encode(text)
            assert original_encoded == new_encoded, f"Encoding mismatch for: {text}"


class TestBPETokenizerEdgeCases:
    """Test edge cases and error handling."""

    def test_train_empty_texts(self):
        """Test training with empty text list."""
        tokenizer = BPETokenizer()
        tokenizer.target_vocab_size = 100

        # Should handle gracefully or raise meaningful error
        try:
            tokenizer.train([], verbose=False)
            # If it doesn't raise error, check vocab is initialized
            assert hasattr(tokenizer, 'vocab')
        except ValueError:
            # Expected behavior
            pass

    def test_train_with_all_empty_strings(self):
        """Test training with list of empty strings."""
        tokenizer = BPETokenizer()
        tokenizer.target_vocab_size = 100

        try:
            tokenizer.train(["", "", ""], verbose=False)
            assert hasattr(tokenizer, 'vocab')
        except ValueError:
            # Expected behavior
            pass

    def test_encode_before_training(self):
        """Test encoding before tokenizer is trained."""
        tokenizer = BPETokenizer()

        # Should either work with basic char-level encoding or raise error
        try:
            encoded = tokenizer.encode("test")
            assert isinstance(encoded, list)
        except (AttributeError, ValueError, KeyError):
            # Expected if vocab not initialized
            pass

    def test_very_long_text(self, small_vocab_tokenizer):
        """Test encoding of very long text."""
        tokenizer = small_vocab_tokenizer
        very_long_text = "echo test\n" * 1000

        encoded = tokenizer.encode(very_long_text)
        decoded = tokenizer.decode(encoded)

        assert decoded == very_long_text, "Very long text should roundtrip"

    def test_single_character(self, small_vocab_tokenizer):
        """Test encoding single characters."""
        tokenizer = small_vocab_tokenizer

        for char in "abcdefgh123!@#":
            encoded = tokenizer.encode(char)
            decoded = tokenizer.decode(encoded)
            assert decoded == char, f"Single character '{char}' should roundtrip"

    def test_whitespace_only(self, small_vocab_tokenizer):
        """Test encoding whitespace-only strings."""
        tokenizer = small_vocab_tokenizer

        whitespace_strings = [" ", "  ", "\n", "\t", "\n\n\n"]

        for ws in whitespace_strings:
            encoded = tokenizer.encode(ws)
            decoded = tokenizer.decode(encoded)
            assert decoded == ws, f"Whitespace '{repr(ws)}' should roundtrip"
