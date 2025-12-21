#!/usr/bin/env python3
"""
Model Quantization for Stage 9 (Model Optimization).

Reduces model size and increases inference speed:
- 8-bit quantization: 4x smaller, ~2x faster
- 4-bit quantization: 8x smaller, ~3x faster
"""

import torch
import torch.nn as nn
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))
from src.model import create_model_from_config
from src.model.config import get_config


def quantize_model_int8(model: nn.Module) -> nn.Module:
    """Quantize model to 8-bit integers."""
    print("Quantizing to INT8...")

    # Dynamic quantization (PyTorch built-in)
    quantized_model = torch.quantization.quantize_dynamic(
        model,
        {nn.Linear},  # Quantize Linear layers
        dtype=torch.qint8
    )

    return quantized_model


def quantize_model_int4(model: nn.Module) -> nn.Module:
    """Quantize model to 4-bit integers (simulated)."""
    print("⚠️  4-bit quantization requires bitsandbytes library")
    print("Install: pip install bitsandbytes")

    try:
        import bitsandbytes as bnb

        # Replace Linear with 4-bit quantized version
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                # Create 4-bit quantized layer
                quantized_layer = bnb.nn.Linear4bit(
                    module.in_features,
                    module.out_features,
                    bias=module.bias is not None
                )
                # Set in parent
                parent_name = '.'.join(name.split('.')[:-1])
                layer_name = name.split('.')[-1]
                parent = model
                if parent_name:
                    for part in parent_name.split('.'):
                        parent = getattr(parent, part)
                setattr(parent, layer_name, quantized_layer)

        return model

    except ImportError:
        print("bitsandbytes not installed, returning original model")
        return model


def measure_model_size(model: nn.Module) -> float:
    """Measure model size in MB."""
    param_size = sum(p.nelement() * p.element_size() for p in model.parameters())
    buffer_size = sum(b.nelement() * b.element_size() for b in model.buffers())
    size_mb = (param_size + buffer_size) / 1024 / 1024
    return size_mb


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Model Quantization (Stage 9)")
    parser.add_argument("--checkpoint", type=str, required=True,
                       help="Path to model checkpoint")
    parser.add_argument("--model-size", type=str, default="large",
                       choices=["tiny", "medium", "large", "xlarge"])
    parser.add_argument("--quantization", type=str, default="int8",
                       choices=["int8", "int4"],
                       help="Quantization precision")
    parser.add_argument("--device", type=str, default="cpu",
                       help="Device (quantization requires CPU)")
    args = parser.parse_args()

    print("=" * 60)
    print("Model Quantization (Stage 9)")
    print("=" * 60)

    # Load model
    print("\nLoading model...")
    config = get_config(args.model_size)
    config.use_rmsnorm = True
    config.use_rope = True

    model = create_model_from_config(config, architecture="dense", device="cpu")

    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    state_dict = checkpoint['model_state_dict']

    if any(key.startswith('_orig_mod.') for key in state_dict.keys()):
        state_dict = {
            key.replace('_orig_mod.', ''): value
            for key, value in state_dict.items()
        }

    model.load_state_dict(state_dict)
    model.eval()
    print("✓ Model loaded")

    # Measure original size
    original_size = measure_model_size(model)
    print(f"\nOriginal model size: {original_size:.2f} MB")

    # Quantize
    if args.quantization == "int8":
        quantized_model = quantize_model_int8(model)
    elif args.quantization == "int4":
        quantized_model = quantize_model_int4(model)

    # Measure quantized size
    quantized_size = measure_model_size(quantized_model)
    compression_ratio = original_size / quantized_size

    print(f"Quantized model size: {quantized_size:.2f} MB")
    print(f"Compression ratio: {compression_ratio:.2f}x")

    # Save
    project_root = Path(__file__).parent.parent
    output_path = project_root / "models" / f"model_{args.quantization}.pth"

    torch.save({
        'model_state_dict': quantized_model.state_dict(),
        'config': config.__dict__,
        'quantization': args.quantization
    }, output_path)

    print(f"\n✓ Saved quantized model: {output_path}")

    print("\n" + "=" * 60)
    print("✓ Quantization complete!")
    print("=" * 60)
    print(f"\nSize reduced from {original_size:.1f}MB to {quantized_size:.1f}MB")


if __name__ == "__main__":
    main()
