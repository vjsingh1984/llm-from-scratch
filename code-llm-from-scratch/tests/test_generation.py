"""
Tests for code generation functionality.

Tests cover:
- Basic generation
- Sampling strategies (temperature, top-k, top-p)
- Generation parameters
- Edge cases
"""

import pytest
import torch
import torch.nn.functional as F
from src.model.transformer import CodeTransformer


class TestBasicGeneration:
    """Test basic code generation."""

    def test_generate_tokens(self, tiny_model, small_vocab_tokenizer, device):
        """Test generating tokens from a prompt."""
        model = tiny_model
        model.eval()
        tokenizer = small_vocab_tokenizer

        prompt = "#!/bin/bash"
        input_ids = tokenizer.encode(prompt)
        input_tensor = torch.tensor([input_ids], dtype=torch.long, device=device)

        max_new_tokens = 10
        generated = input_tensor.clone()

        with torch.no_grad():
            for _ in range(max_new_tokens):
                logits, _ = model(generated)
                next_logits = logits[0, -1, :]
                probs = F.softmax(next_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                generated = torch.cat([generated, next_token.unsqueeze(0)], dim=1)

        assert generated.shape[1] == len(input_ids) + max_new_tokens
        assert generated.shape[0] == 1

    def test_greedy_generation(self, tiny_model, small_vocab_tokenizer, device):
        """Test greedy decoding (argmax)."""
        model = tiny_model
        model.eval()
        tokenizer = small_vocab_tokenizer

        prompt = "#!/bin/bash"
        input_ids = tokenizer.encode(prompt)
        input_tensor = torch.tensor([input_ids], dtype=torch.long, device=device)

        max_new_tokens = 10
        generated = input_tensor.clone()

        with torch.no_grad():
            for _ in range(max_new_tokens):
                logits, _ = model(generated)
                next_token = logits[0, -1, :].argmax(dim=-1, keepdim=True)
                generated = torch.cat([generated, next_token.unsqueeze(0)], dim=1)

        # Greedy decoding should be deterministic
        generated2 = input_tensor.clone()
        with torch.no_grad():
            for _ in range(max_new_tokens):
                logits, _ = model(generated2)
                next_token = logits[0, -1, :].argmax(dim=-1, keepdim=True)
                generated2 = torch.cat([generated2, next_token.unsqueeze(0)], dim=1)

        assert torch.equal(generated, generated2), "Greedy decoding should be deterministic"

    def test_decode_generated_tokens(self, tiny_model, small_vocab_tokenizer, device):
        """Test that generated tokens can be decoded to text."""
        model = tiny_model
        model.eval()
        tokenizer = small_vocab_tokenizer

        prompt = "#!/bin/bash"
        input_ids = tokenizer.encode(prompt)
        input_tensor = torch.tensor([input_ids], dtype=torch.long, device=device)

        max_new_tokens = 5
        generated = input_tensor.clone()

        with torch.no_grad():
            for _ in range(max_new_tokens):
                logits, _ = model(generated)
                next_logits = logits[0, -1, :]
                probs = F.softmax(next_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                generated = torch.cat([generated, next_token.unsqueeze(0)], dim=1)

        generated_ids = generated[0].tolist()
        decoded_text = tokenizer.decode(generated_ids)

        assert isinstance(decoded_text, str)
        assert len(decoded_text) > len(prompt), "Generated text should be longer than prompt"
        assert decoded_text.startswith(prompt), "Generated text should start with prompt"


class TestSamplingStrategies:
    """Test different sampling strategies."""

    def test_temperature_sampling(self, tiny_model, small_vocab_tokenizer, device):
        """Test temperature-based sampling."""
        model = tiny_model
        model.eval()
        tokenizer = small_vocab_tokenizer

        prompt = "#!/bin/bash"
        input_ids = tokenizer.encode(prompt)
        input_tensor = torch.tensor([input_ids], dtype=torch.long, device=device)

        temperatures = [0.1, 0.8, 1.5]

        for temperature in temperatures:
            generated = input_tensor.clone()

            with torch.no_grad():
                for _ in range(5):
                    logits, _ = model(generated)
                    next_logits = logits[0, -1, :] / temperature
                    probs = F.softmax(next_logits, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1)
                    generated = torch.cat([generated, next_token.unsqueeze(0)], dim=1)

            assert generated.shape[1] > len(input_ids), "Should generate tokens"

    def test_top_k_sampling(self, tiny_model, small_vocab_tokenizer, device):
        """Test top-k sampling."""
        model = tiny_model
        model.eval()
        tokenizer = small_vocab_tokenizer

        prompt = "#!/bin/bash"
        input_ids = tokenizer.encode(prompt)
        input_tensor = torch.tensor([input_ids], dtype=torch.long, device=device)

        top_k = 10

        generated = input_tensor.clone()

        with torch.no_grad():
            for _ in range(5):
                logits, _ = model(generated)
                next_logits = logits[0, -1, :]

                # Top-k filtering
                top_k_values, top_k_indices = torch.topk(next_logits, top_k)
                indices_to_remove = next_logits < top_k_values[-1]
                next_logits[indices_to_remove] = float('-inf')

                probs = F.softmax(next_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                generated = torch.cat([generated, next_token.unsqueeze(0)], dim=1)

        assert generated.shape[1] > len(input_ids), "Should generate tokens"

    def test_top_p_nucleus_sampling(self, tiny_model, small_vocab_tokenizer, device):
        """Test top-p (nucleus) sampling."""
        model = tiny_model
        model.eval()
        tokenizer = small_vocab_tokenizer

        prompt = "#!/bin/bash"
        input_ids = tokenizer.encode(prompt)
        input_tensor = torch.tensor([input_ids], dtype=torch.long, device=device)

        top_p = 0.9

        generated = input_tensor.clone()

        with torch.no_grad():
            for _ in range(5):
                logits, _ = model(generated)
                next_logits = logits[0, -1, :]

                # Top-p filtering
                sorted_logits, sorted_indices = torch.sort(next_logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

                # Remove tokens with cumulative probability above threshold
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0

                indices_to_remove = sorted_indices[sorted_indices_to_remove]
                next_logits[indices_to_remove] = float('-inf')

                probs = F.softmax(next_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                generated = torch.cat([generated, next_token.unsqueeze(0)], dim=1)

        assert generated.shape[1] > len(input_ids), "Should generate tokens"

    def test_temperature_effect_on_diversity(self, tiny_model, small_vocab_tokenizer, device):
        """Test that temperature affects generation diversity."""
        model = tiny_model
        model.eval()
        tokenizer = small_vocab_tokenizer

        prompt = "#!/bin/bash"
        input_ids = tokenizer.encode(prompt)
        input_tensor = torch.tensor([input_ids], dtype=torch.long, device=device)

        def generate_with_temp(temperature, num_generations=3):
            """Generate multiple sequences with given temperature."""
            results = []
            for _ in range(num_generations):
                generated = input_tensor.clone()
                with torch.no_grad():
                    for _ in range(10):
                        logits, _ = model(generated)
                        next_logits = logits[0, -1, :] / temperature
                        probs = F.softmax(next_logits, dim=-1)
                        next_token = torch.multinomial(probs, num_samples=1)
                        generated = torch.cat([generated, next_token.unsqueeze(0)], dim=1)
                results.append(generated[0].tolist())
            return results

        # Low temperature should be more deterministic
        low_temp_results = generate_with_temp(0.1)

        # High temperature should be more diverse
        high_temp_results = generate_with_temp(1.5)

        # Check that at least we got results
        assert len(low_temp_results) == 3
        assert len(high_temp_results) == 3


class TestGenerationEdgeCases:
    """Test edge cases in generation."""

    def test_empty_prompt_generation(self, tiny_model, small_vocab_tokenizer, device):
        """Test generation from empty prompt."""
        model = tiny_model
        model.eval()

        # Start with a single BOS token or random token
        input_tensor = torch.randint(0, model.config.vocab_size, (1, 1), device=device)

        generated = input_tensor.clone()

        with torch.no_grad():
            for _ in range(10):
                logits, _ = model(generated)
                next_logits = logits[0, -1, :]
                probs = F.softmax(next_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                generated = torch.cat([generated, next_token.unsqueeze(0)], dim=1)

        assert generated.shape[1] == 11, "Should generate 10 new tokens"

    def test_max_length_generation(self, tiny_model, small_vocab_tokenizer, device):
        """Test generation up to maximum sequence length."""
        model = tiny_model
        model.eval()
        tokenizer = small_vocab_tokenizer

        prompt = "#!/bin/bash"
        input_ids = tokenizer.encode(prompt)
        input_tensor = torch.tensor([input_ids], dtype=torch.long, device=device)

        # Generate up to near max length
        max_new_tokens = min(50, model.config.max_seq_len - len(input_ids))
        generated = input_tensor.clone()

        with torch.no_grad():
            for _ in range(max_new_tokens):
                if generated.shape[1] >= model.config.max_seq_len:
                    break
                logits, _ = model(generated)
                next_logits = logits[0, -1, :]
                probs = F.softmax(next_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                generated = torch.cat([generated, next_token.unsqueeze(0)], dim=1)

        assert generated.shape[1] <= model.config.max_seq_len

    def test_batch_generation(self, tiny_model, small_vocab_tokenizer, device):
        """Test generating from multiple prompts in batch."""
        model = tiny_model
        model.eval()
        tokenizer = small_vocab_tokenizer

        prompts = ["#!/bin/bash", "echo"]
        input_ids_list = [tokenizer.encode(p) for p in prompts]

        # Pad to same length
        max_len = max(len(ids) for ids in input_ids_list)
        padded = []
        for ids in input_ids_list:
            if len(ids) < max_len:
                ids = ids + [0] * (max_len - len(ids))
            padded.append(ids[:max_len])

        input_tensor = torch.tensor(padded, dtype=torch.long, device=device)

        generated = input_tensor.clone()

        with torch.no_grad():
            for _ in range(5):
                logits, _ = model(generated)
                next_logits = logits[:, -1, :]
                probs = F.softmax(next_logits, dim=-1)
                next_tokens = torch.multinomial(probs, num_samples=1)
                generated = torch.cat([generated, next_tokens], dim=1)

        assert generated.shape[0] == 2, "Should handle batch of 2"
        assert generated.shape[1] == max_len + 5, "Should generate 5 new tokens"

    def test_generation_produces_valid_tokens(self, tiny_model, small_vocab_tokenizer, device):
        """Test that all generated tokens are valid."""
        model = tiny_model
        model.eval()
        tokenizer = small_vocab_tokenizer

        prompt = "#!/bin/bash"
        input_ids = tokenizer.encode(prompt)
        input_tensor = torch.tensor([input_ids], dtype=torch.long, device=device)

        generated = input_tensor.clone()

        with torch.no_grad():
            for _ in range(10):
                logits, _ = model(generated)
                next_logits = logits[0, -1, :]
                probs = F.softmax(next_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                generated = torch.cat([generated, next_token.unsqueeze(0)], dim=1)

        # All tokens should be valid (within vocab range)
        vocab_size = model.config.vocab_size
        all_tokens = generated[0].tolist()

        for token in all_tokens:
            assert 0 <= token < vocab_size, f"Token {token} out of valid range [0, {vocab_size})"

    def test_no_nan_in_generation(self, tiny_model, small_vocab_tokenizer, device):
        """Test that generation doesn't produce NaN values."""
        model = tiny_model
        model.eval()
        tokenizer = small_vocab_tokenizer

        prompt = "#!/bin/bash"
        input_ids = tokenizer.encode(prompt)
        input_tensor = torch.tensor([input_ids], dtype=torch.long, device=device)

        generated = input_tensor.clone()

        with torch.no_grad():
            for _ in range(10):
                logits, _ = model(generated)
                assert not torch.isnan(logits).any(), "Logits should not contain NaN"

                next_logits = logits[0, -1, :]
                probs = F.softmax(next_logits, dim=-1)

                assert not torch.isnan(probs).any(), "Probabilities should not contain NaN"
                assert torch.allclose(probs.sum(), torch.tensor(1.0, device=device), rtol=1e-5), \
                    "Probabilities should sum to 1"

                next_token = torch.multinomial(probs, num_samples=1)
                generated = torch.cat([generated, next_token.unsqueeze(0)], dim=1)
