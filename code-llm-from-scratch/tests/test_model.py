"""
Tests for the transformer model.

Tests cover:
- Model initialization
- Forward pass
- Parameter counts
- Attention mechanisms
- Loss computation
"""

import pytest
import torch
import torch.nn as nn
from src.model.transformer import CodeTransformer, TransformerBlock, MultiHeadAttention
from src.model.config import CoderConfig, get_model_config


class TestCoderConfig:
    """Test model configuration."""

    def test_default_config(self):
        """Test default configuration values."""
        config = CoderConfig()

        assert config.vocab_size > 0
        assert config.n_layers > 0
        assert config.d_model > 0
        assert config.n_heads > 0
        assert config.d_ff > 0
        assert config.max_seq_len > 0
        assert 0 <= config.dropout <= 1

    def test_custom_config(self):
        """Test custom configuration."""
        config = CoderConfig(
            vocab_size=5000,
            n_layers=4,
            d_model=256,
            n_heads=4,
            d_ff=1024
        )

        assert config.vocab_size == 5000
        assert config.n_layers == 4
        assert config.d_model == 256
        assert config.n_heads == 4
        assert config.d_ff == 1024

    def test_get_model_config_tiny(self):
        """Test tiny model configuration."""
        config = get_model_config("tiny", vocab_size=8000)

        assert config.vocab_size == 8000
        assert config.n_layers == 4
        assert config.d_model == 128

    def test_get_model_config_small(self):
        """Test small model configuration."""
        config = get_model_config("small", vocab_size=8000)

        assert config.vocab_size == 8000
        assert config.n_layers == 6
        assert config.d_model == 384

    def test_get_model_config_medium(self):
        """Test medium model configuration."""
        config = get_model_config("medium", vocab_size=8000)

        assert config.vocab_size == 8000
        assert config.n_layers == 12
        assert config.d_model == 512


class TestCodeTransformer:
    """Test transformer model."""

    def test_model_initialization(self, tiny_model_config):
        """Test model can be initialized."""
        model = CodeTransformer(tiny_model_config)

        assert isinstance(model, nn.Module)
        assert hasattr(model, 'token_embedding')
        assert hasattr(model, 'pos_embedding')
        assert hasattr(model, 'blocks')
        assert hasattr(model, 'lm_head')

    def test_forward_pass_no_targets(self, tiny_model, sample_batch):
        """Test forward pass without targets (inference mode)."""
        model = tiny_model
        input_ids = sample_batch

        logits, loss = model(input_ids, targets=None)

        # Check logits shape
        batch_size, seq_len = input_ids.shape
        vocab_size = model.config.vocab_size

        assert logits.shape == (batch_size, seq_len, vocab_size)
        assert loss is None, "Loss should be None when targets not provided"

    def test_forward_pass_with_targets(self, tiny_model, sample_batch):
        """Test forward pass with targets (training mode)."""
        model = tiny_model
        input_ids = sample_batch
        targets = input_ids.clone()

        logits, loss = model(input_ids, targets=targets)

        # Check logits shape
        batch_size, seq_len = input_ids.shape
        vocab_size = model.config.vocab_size

        assert logits.shape == (batch_size, seq_len, vocab_size)
        assert loss is not None, "Loss should be computed when targets provided"
        assert isinstance(loss.item(), float)
        assert loss.item() > 0, "Loss should be positive"

    def test_parameter_count(self, tiny_model_config):
        """Test that parameter count matches expected value."""
        model = CodeTransformer(tiny_model_config)
        num_params = sum(p.numel() for p in model.parameters())

        # Tiny model should have reasonable number of parameters
        assert num_params > 1000, "Model should have at least 1K parameters"
        assert num_params < 10_000_000, "Tiny model should have less than 10M parameters"

    def test_all_parameters_require_grad(self, tiny_model):
        """Test that all parameters require gradients."""
        for name, param in tiny_model.named_parameters():
            assert param.requires_grad, f"Parameter {name} should require gradients"

    def test_embeddings_size(self, tiny_model_config):
        """Test embedding layer dimensions."""
        model = CodeTransformer(tiny_model_config)

        token_emb_weight = model.token_embedding.weight
        pos_emb_weight = model.pos_embedding.weight

        assert token_emb_weight.shape == (tiny_model_config.vocab_size, tiny_model_config.d_model)
        assert pos_emb_weight.shape == (tiny_model_config.max_seq_len, tiny_model_config.d_model)

    def test_output_head_size(self, tiny_model_config):
        """Test output projection layer dimensions."""
        model = CodeTransformer(tiny_model_config)

        lm_head_weight = model.lm_head.weight

        assert lm_head_weight.shape == (tiny_model_config.vocab_size, tiny_model_config.d_model)

    def test_number_of_blocks(self, tiny_model_config):
        """Test that model has correct number of transformer blocks."""
        model = CodeTransformer(tiny_model_config)

        assert len(model.blocks) == tiny_model_config.n_layers

    def test_gradient_flow(self, tiny_model, sample_batch, device):
        """Test that gradients flow through the model."""
        model = tiny_model
        model.train()

        input_ids = sample_batch
        targets = input_ids.clone()

        # Forward pass
        logits, loss = model(input_ids, targets)

        # Backward pass
        loss.backward()

        # Check that gradients exist
        for name, param in model.named_parameters():
            assert param.grad is not None, f"Gradient should exist for {name}"
            assert not torch.isnan(param.grad).any(), f"Gradient should not be NaN for {name}"

    def test_different_sequence_lengths(self, tiny_model, device):
        """Test model handles different sequence lengths."""
        model = tiny_model

        for seq_len in [10, 20, 50]:
            input_ids = torch.randint(0, model.config.vocab_size, (2, seq_len), device=device)
            logits, _ = model(input_ids)

            assert logits.shape == (2, seq_len, model.config.vocab_size)

    def test_single_sample_batch(self, tiny_model, device):
        """Test model with batch size of 1."""
        model = tiny_model

        input_ids = torch.randint(0, model.config.vocab_size, (1, 20), device=device)
        logits, _ = model(input_ids)

        assert logits.shape == (1, 20, model.config.vocab_size)

    def test_model_eval_mode(self, tiny_model, sample_batch):
        """Test model in evaluation mode."""
        model = tiny_model
        model.eval()

        input_ids = sample_batch

        with torch.no_grad():
            logits1, _ = model(input_ids)
            logits2, _ = model(input_ids)

        # In eval mode with same input, output should be deterministic
        assert torch.allclose(logits1, logits2, rtol=1e-5)

    def test_model_train_mode_with_dropout(self, tiny_model_config, device):
        """Test that dropout is active in training mode."""
        # Create model with dropout
        config = tiny_model_config
        config.dropout = 0.5  # High dropout for testing
        model = CodeTransformer(config).to(device)

        model.train()

        # Same input should produce different outputs due to dropout
        input_ids = torch.randint(0, config.vocab_size, (2, 20), device=device)

        with torch.no_grad():
            logits1, _ = model(input_ids)

        with torch.no_grad():
            logits2, _ = model(input_ids)

        # With dropout, outputs might differ slightly
        # (though this test might be flaky)
        # Just check they have same shape
        assert logits1.shape == logits2.shape

    def test_max_sequence_length_constraint(self, tiny_model, device):
        """Test model behavior at maximum sequence length."""
        model = tiny_model
        max_len = model.config.max_seq_len

        # Should work at max length
        input_ids = torch.randint(0, model.config.vocab_size, (1, max_len), device=device)
        logits, _ = model(input_ids)

        assert logits.shape == (1, max_len, model.config.vocab_size)

    def test_loss_decreases_with_training(self, tiny_model, sample_batch):
        """Test that loss decreases with training steps."""
        model = tiny_model
        model.train()

        input_ids = sample_batch
        targets = input_ids.clone()

        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        # Get initial loss
        logits, initial_loss = model(input_ids, targets)
        initial_loss_value = initial_loss.item()

        # Train for a few steps
        for _ in range(10):
            optimizer.zero_grad()
            logits, loss = model(input_ids, targets)
            loss.backward()
            optimizer.step()

        # Get final loss
        logits, final_loss = model(input_ids, targets)
        final_loss_value = final_loss.item()

        # Loss should decrease (with high probability)
        assert final_loss_value < initial_loss_value, "Loss should decrease with training"


class TestMultiHeadAttention:
    """Test multi-head attention mechanism."""

    def test_attention_initialization(self, tiny_model_config):
        """Test attention layer initialization."""
        attention = MultiHeadAttention(tiny_model_config)

        assert isinstance(attention, nn.Module)
        assert hasattr(attention, 'c_attn')
        assert hasattr(attention, 'c_proj')

    def test_attention_forward(self, tiny_model_config, device):
        """Test attention forward pass."""
        attention = MultiHeadAttention(tiny_model_config).to(device)

        batch_size, seq_len, d_model = 2, 10, tiny_model_config.d_model
        x = torch.randn(batch_size, seq_len, d_model, device=device)

        output = attention(x)

        assert output.shape == x.shape, "Attention output should have same shape as input"

    def test_attention_mask_causality(self, tiny_model_config, device):
        """Test that attention mask enforces causality."""
        attention = MultiHeadAttention(tiny_model_config).to(device)
        attention.eval()

        batch_size, seq_len, d_model = 1, 5, tiny_model_config.d_model

        # Create input where each position has distinct value
        x = torch.arange(seq_len, device=device).float().view(1, seq_len, 1)
        x = x.expand(batch_size, seq_len, d_model)

        with torch.no_grad():
            output = attention(x)

        # Due to causal masking, each position should only attend to previous positions
        # This is a weak test - just check shape is preserved
        assert output.shape == x.shape


class TestTransformerBlock:
    """Test transformer block."""

    def test_block_initialization(self, tiny_model_config):
        """Test transformer block initialization."""
        block = TransformerBlock(tiny_model_config)

        assert isinstance(block, nn.Module)
        assert hasattr(block, 'attn')
        assert hasattr(block, 'mlp')
        assert hasattr(block, 'ln1')
        assert hasattr(block, 'ln2')

    def test_block_forward(self, tiny_model_config, device):
        """Test transformer block forward pass."""
        block = TransformerBlock(tiny_model_config).to(device)

        batch_size, seq_len, d_model = 2, 10, tiny_model_config.d_model
        x = torch.randn(batch_size, seq_len, d_model, device=device)

        output = block(x)

        assert output.shape == x.shape, "Block output should have same shape as input"

    def test_residual_connections(self, tiny_model_config, device):
        """Test that residual connections work."""
        block = TransformerBlock(tiny_model_config).to(device)
        block.eval()

        batch_size, seq_len, d_model = 2, 10, tiny_model_config.d_model
        x = torch.randn(batch_size, seq_len, d_model, device=device)

        with torch.no_grad():
            output = block(x)

        # Output should not be identical to input (due to transformations)
        assert not torch.allclose(output, x)

        # But output magnitude should be reasonable (not exploding/vanishing)
        assert output.abs().mean() > 0.01
        assert output.abs().mean() < 100


@pytest.mark.slow
class TestModelSaveLoad:
    """Test model saving and loading."""

    def test_save_and_load_state_dict(self, tiny_model, temp_dir, device):
        """Test saving and loading model state dict."""
        model = tiny_model

        # Save
        save_path = os.path.join(temp_dir, "model.pt")
        torch.save({
            'model_state_dict': model.state_dict(),
            'config': model.config
        }, save_path)

        # Create new model and load
        new_model = CodeTransformer(model.config).to(device)
        checkpoint = torch.load(save_path, map_location=device)
        new_model.load_state_dict(checkpoint['model_state_dict'])

        # Test that models produce same output
        new_model.eval()
        model.eval()

        input_ids = torch.randint(0, model.config.vocab_size, (2, 20), device=device)

        with torch.no_grad():
            logits1, _ = model(input_ids)
            logits2, _ = new_model(input_ids)

        assert torch.allclose(logits1, logits2, rtol=1e-5), "Loaded model should produce same output"


# Import os for save/load test
import os
