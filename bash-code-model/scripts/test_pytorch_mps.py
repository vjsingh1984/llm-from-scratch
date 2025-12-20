"""
Test PyTorch model with MPS (Metal Performance Shaders) backend.

Verifies that:
1. PyTorch is installed
2. MPS is available on M1 Max
3. Model can be created and run on MPS
4. Forward pass and generation work
"""

import sys
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn.functional as F


def test_pytorch_installation():
    """Test PyTorch is installed."""
    print("="*60)
    print("Testing PyTorch Installation")
    print("="*60)
    print(f"PyTorch version: {torch.__version__}")
    print(f"Python version: {sys.version}")
    print()


def test_mps_availability():
    """Test MPS backend is available."""
    print("="*60)
    print("Testing MPS (Metal) Availability")
    print("="*60)

    if torch.backends.mps.is_available():
        print("✓ MPS is available!")
        print("✓ Your M1 Max GPU can be used for training")

        # Test basic operation
        device = torch.device("mps")
        x = torch.randn(100, 100, device=device)
        y = x @ x.t()
        print(f"✓ Basic GPU operation works")
        print(f"  Result shape: {y.shape}")
    else:
        print("✗ MPS is not available")
        print("  You may need to update PyTorch or macOS")
        return False

    print()
    return True


def test_model_creation():
    """Test creating and running model on MPS."""
    print("="*60)
    print("Testing Model Creation")
    print("="*60)

    try:
        from model import create_model, CoderConfig

        # Create tiny model
        print("Creating tiny coder model...")
        model = create_model('tiny', vocab_size=1000, device='mps')

        print(f"✓ Model created successfully")
        print(f"  Parameters: {model.count_parameters():,}")
        print(f"  Device: {model.get_device()}")

        return model

    except Exception as e:
        print(f"✗ Model creation failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def test_forward_pass(model):
    """Test forward pass through model."""
    print("\n" + "="*60)
    print("Testing Forward Pass")
    print("="*60)

    try:
        device = model.get_device()

        # Create dummy input
        batch_size = 4
        seq_len = 32
        input_ids = torch.randint(0, 1000, (batch_size, seq_len), device=device)

        print(f"Input shape: {input_ids.shape}")

        # Forward pass
        with torch.no_grad():
            logits, _ = model(input_ids)

        print(f"✓ Forward pass successful")
        print(f"  Output shape: {logits.shape}")
        print(f"  Expected: [{batch_size}, {seq_len}, {model.config.vocab_size}]")

        assert logits.shape == (batch_size, seq_len, model.config.vocab_size)
        print("✓ Output shape correct")

        return True

    except Exception as e:
        print(f"✗ Forward pass failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_loss_computation(model):
    """Test loss computation."""
    print("\n" + "="*60)
    print("Testing Loss Computation")
    print("="*60)

    try:
        device = model.get_device()

        # Create dummy data
        batch_size = 4
        seq_len = 32
        input_ids = torch.randint(0, 1000, (batch_size, seq_len), device=device)
        targets = torch.randint(0, 1000, (batch_size, seq_len), device=device)

        # Compute loss
        logits, loss = model(input_ids, targets)

        print(f"✓ Loss computation successful")
        print(f"  Loss value: {loss.item():.4f}")
        print(f"  Loss shape: {loss.shape}")

        assert loss.shape == torch.Size([])
        print("✓ Loss is scalar")

        return True

    except Exception as e:
        print(f"✗ Loss computation failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_generation(model):
    """Test text generation."""
    print("\n" + "="*60)
    print("Testing Text Generation")
    print("="*60)

    try:
        device = model.get_device()

        # Start with a few tokens
        input_ids = torch.tensor([[1, 2, 3, 4, 5]], device=device)

        print(f"Starting tokens: {input_ids.shape}")

        # Generate
        output_ids = model.generate(
            input_ids,
            max_new_tokens=20,
            temperature=0.8,
            top_k=40
        )

        print(f"✓ Generation successful")
        print(f"  Generated shape: {output_ids.shape}")
        print(f"  Input length: {input_ids.shape[1]}")
        print(f"  Output length: {output_ids.shape[1]}")
        print(f"  New tokens: {output_ids.shape[1] - input_ids.shape[1]}")

        assert output_ids.shape[1] == input_ids.shape[1] + 20
        print("✓ Generated correct number of tokens")

        return True

    except Exception as e:
        print(f"✗ Generation failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_parameter_count():
    """Test parameter counting for different sizes."""
    print("\n" + "="*60)
    print("Testing Different Model Sizes")
    print("="*60)

    from model import CoderConfig

    sizes = {
        'tiny': CoderConfig.tiny_coder(),
        'small': CoderConfig.small_coder(),
        'medium': CoderConfig.medium_coder(),
    }

    for name, config in sizes.items():
        estimated = config.get_num_parameters()
        print(f"\n{name.capitalize()} Coder:")
        print(f"  Layers: {config.n_layers}")
        print(f"  Hidden: {config.d_model}")
        print(f"  Heads: {config.n_heads}")
        print(f"  Estimated params: {estimated:,} ({estimated/1e6:.1f}M)")


def benchmark_speed(model):
    """Benchmark inference speed."""
    print("\n" + "="*60)
    print("Benchmarking Speed on M1 Max")
    print("="*60)

    import time

    device = model.get_device()
    batch_size = 16
    seq_len = 256

    # Create dummy data
    input_ids = torch.randint(0, 1000, (batch_size, seq_len), device=device)

    # Warmup
    for _ in range(5):
        with torch.no_grad():
            _ = model(input_ids)

    # Benchmark
    num_iterations = 20
    start = time.time()

    with torch.no_grad():
        for _ in range(num_iterations):
            _ = model(input_ids)

    elapsed = time.time() - start

    tokens_processed = batch_size * seq_len * num_iterations
    tokens_per_sec = tokens_processed / elapsed

    print(f"Batch size: {batch_size}")
    print(f"Sequence length: {seq_len}")
    print(f"Iterations: {num_iterations}")
    print(f"Time: {elapsed:.2f} seconds")
    print(f"Speed: {tokens_per_sec:,.0f} tokens/second")
    print(f"\nThis is {tokens_per_sec/1000:.1f}K tokens/sec on your M1 Max!")


def main():
    """Run all tests."""
    print("\n" + "="*60)
    print("PyTorch + MPS Test Suite")
    print("="*60)
    print()

    # Test installation
    test_pytorch_installation()

    # Test MPS
    if not test_mps_availability():
        print("\n✗ MPS not available. Please install PyTorch with MPS support.")
        return

    # Test model
    model = test_model_creation()
    if model is None:
        print("\n✗ Model creation failed. Cannot continue.")
        return

    # Test forward pass
    if not test_forward_pass(model):
        return

    # Test loss
    if not test_loss_computation(model):
        return

    # Test generation
    if not test_generation(model):
        return

    # Test parameter counts
    test_parameter_count()

    # Benchmark
    benchmark_speed(model)

    print("\n" + "="*60)
    print("ALL TESTS PASSED! ✓")
    print("="*60)
    print("\nYour PyTorch setup is working correctly with MPS acceleration.")
    print("You're ready to train bash code generation models on M1 Max!")


if __name__ == '__main__':
    main()
