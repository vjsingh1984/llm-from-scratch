"""
Test script to verify the transformer model works correctly.

This script creates a small model and runs a forward pass to ensure
all components are working properly.
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import mlx.core as mx
from model import GPTModel, GPTConfig


def test_model_forward():
    """Test basic forward pass."""
    print("="*60)
    print("Testing Model Forward Pass")
    print("="*60)

    # Create a tiny model for testing
    config = GPTConfig.tiny(vocab_size=1000)
    config.max_seq_len = 128
    config.n_layers = 4  # Even smaller for quick testing

    print(f"\nModel Configuration:")
    print(f"  Vocabulary size: {config.vocab_size}")
    print(f"  Layers: {config.n_layers}")
    print(f"  Hidden size: {config.d_model}")
    print(f"  Attention heads: {config.n_heads}")
    print(f"  FFN size: {config.d_ff}")
    print(f"  Max sequence length: {config.max_seq_len}")

    # Create model
    print("\nCreating model...")
    model = GPTModel(config)

    # Count parameters
    n_params = model.count_parameters()
    print(f"Total parameters: {n_params:,} ({n_params/1e6:.2f}M)")

    # Create dummy input
    batch_size = 4
    seq_len = 32
    print(f"\nInput shape: [{batch_size}, {seq_len}]")

    # Random token IDs
    input_ids = mx.random.randint(0, config.vocab_size, (batch_size, seq_len))

    # Forward pass
    print("\nRunning forward pass...")
    logits, _ = model(input_ids)

    print(f"Output shape: {logits.shape}")
    print(f"Expected shape: [{batch_size}, {seq_len}, {config.vocab_size}]")

    assert logits.shape == (batch_size, seq_len, config.vocab_size), \
        f"Unexpected output shape: {logits.shape}"

    print("\n✓ Forward pass successful!")

    return model


def test_model_with_loss():
    """Test forward pass with loss computation."""
    print("\n" + "="*60)
    print("Testing Model with Loss Computation")
    print("="*60)

    config = GPTConfig.tiny(vocab_size=1000)
    config.n_layers = 4
    model = GPTModel(config)

    batch_size = 4
    seq_len = 32

    # Create input and targets
    input_ids = mx.random.randint(0, config.vocab_size, (batch_size, seq_len))
    targets = mx.random.randint(0, config.vocab_size, (batch_size, seq_len))

    print(f"Input shape: {input_ids.shape}")
    print(f"Target shape: {targets.shape}")

    # Forward pass with loss
    print("\nComputing loss...")
    logits, loss = model(input_ids, targets=targets)

    print(f"Logits shape: {logits.shape}")
    print(f"Loss: {loss.item():.4f}")

    assert loss.shape == (), f"Loss should be scalar, got shape {loss.shape}"
    assert loss.item() > 0, "Loss should be positive"

    print("\n✓ Loss computation successful!")


def test_generation():
    """Test text generation."""
    print("\n" + "="*60)
    print("Testing Text Generation")
    print("="*60)

    config = GPTConfig.tiny(vocab_size=1000)
    config.n_layers = 4
    config.max_seq_len = 128
    model = GPTModel(config)

    # Starting tokens
    batch_size = 2
    start_tokens = mx.array([[1, 2, 3], [4, 5, 6]])  # [batch_size, 3]

    print(f"Starting tokens: {start_tokens.shape}")
    print(f"Generating 10 new tokens...")

    # Generate
    generated = model.generate(
        start_tokens,
        max_new_tokens=10,
        temperature=1.0
    )

    print(f"Generated shape: {generated.shape}")
    print(f"Expected shape: [{batch_size}, {start_tokens.shape[1] + 10}]")

    assert generated.shape == (batch_size, start_tokens.shape[1] + 10), \
        f"Unexpected generated shape: {generated.shape}"

    print(f"\nGenerated tokens:\n{generated}")
    print("\n✓ Generation successful!")


def test_different_sizes():
    """Test creating models of different sizes."""
    print("\n" + "="*60)
    print("Testing Different Model Sizes")
    print("="*60)

    sizes = {
        "Tiny (50M)": GPTConfig.tiny(vocab_size=8000),
        "GPT-2 Small (124M)": GPTConfig.gpt2_small(),
    }

    for name, config in sizes.items():
        print(f"\n{name}:")
        model = GPTModel(config)
        n_params = model.count_parameters()
        print(f"  Parameters: {n_params:,} ({n_params/1e6:.2f}M)")
        print(f"  Layers: {config.n_layers}")
        print(f"  Hidden size: {config.d_model}")
        print(f"  Attention heads: {config.n_heads}")

    print("\n✓ All model sizes created successfully!")


def test_attention_variants():
    """Test different attention mechanisms."""
    print("\n" + "="*60)
    print("Testing Attention Variants")
    print("="*60)

    # Standard multi-head attention
    config1 = GPTConfig.tiny(vocab_size=1000)
    config1.use_gqa = False
    model1 = GPTModel(config1)
    print(f"\nStandard Multi-Head Attention:")
    print(f"  Parameters: {model1.count_parameters():,}")

    # Grouped query attention
    config2 = GPTConfig.tiny(vocab_size=1000)
    config2.use_gqa = True
    config2.n_kv_heads = 4  # 8 query heads, 4 KV heads
    model2 = GPTModel(config2)
    print(f"\nGrouped Query Attention (n_kv_heads=4):")
    print(f"  Parameters: {model2.count_parameters():,}")

    # Test forward pass
    input_ids = mx.random.randint(0, 1000, (2, 16))
    logits1, _ = model1(input_ids)
    logits2, _ = model2(input_ids)

    print(f"\nOutput shapes:")
    print(f"  MHA: {logits1.shape}")
    print(f"  GQA: {logits2.shape}")

    print("\n✓ Attention variants working!")


def main():
    """Run all tests."""
    print("\n" + "="*60)
    print("TRANSFORMER MODEL TESTS")
    print("="*60)

    try:
        # Test basic functionality
        model = test_model_forward()
        test_model_with_loss()
        test_generation()

        # Test variants
        test_different_sizes()
        test_attention_variants()

        print("\n" + "="*60)
        print("ALL TESTS PASSED! ✓")
        print("="*60)
        print("\nThe transformer model is working correctly.")
        print("You can now proceed to implement the training loop.")

    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
