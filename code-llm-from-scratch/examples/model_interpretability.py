#!/usr/bin/env python3
"""
Model Interpretability Tools

Visualize and understand what the model has learned:
- Attention pattern visualization
- Token probability analysis
- Layer activation analysis
- Vocabulary usage statistics

Usage:
    python examples/model_interpretability.py --model models/code/code_model_final.pt
"""

import os
import sys
import argparse
from typing import List, Tuple
from collections import Counter

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.model.transformer import CodeTransformer
from src.tokenizer.bpe import BPETokenizer


class ModelInterpreter:
    """Tools for interpreting model behavior."""

    def __init__(self, model_path: str, tokenizer_path: str, device: torch.device):
        # Load model
        checkpoint = torch.load(model_path, map_location=device)
        self.config = checkpoint['config']
        self.model = CodeTransformer(self.config)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model = self.model.to(device)
        self.model.eval()

        # Load tokenizer
        self.tokenizer = BPETokenizer()
        self.tokenizer.load(tokenizer_path)

        self.device = device

    def analyze_token_probabilities(
        self,
        prompt: str,
        top_k: int = 10
    ) -> List[Tuple[str, float]]:
        """
        Analyze what tokens the model predicts next and their probabilities.

        Args:
            prompt: Input prompt
            top_k: Number of top predictions to show

        Returns:
            List of (token, probability) tuples
        """
        # Encode prompt
        input_ids = self.tokenizer.encode(prompt)
        input_tensor = torch.tensor([input_ids], dtype=torch.long, device=self.device)

        # Get predictions
        with torch.no_grad():
            logits, _ = self.model(input_tensor)
            next_token_logits = logits[0, -1, :]
            probs = F.softmax(next_token_logits, dim=-1)

        # Get top k predictions
        top_probs, top_indices = torch.topk(probs, top_k)

        results = []
        for prob, idx in zip(top_probs, top_indices):
            token = self.tokenizer.decode([idx.item()])
            results.append((token, prob.item()))

        return results

    def visualize_next_token_predictions(
        self,
        prompt: str,
        top_k: int = 15,
        save_path: str = None
    ):
        """Visualize top-k next token predictions."""
        predictions = self.analyze_token_probabilities(prompt, top_k)

        # Create bar chart
        tokens = [repr(t[0]) for t in predictions]  # Use repr to show special chars
        probs = [t[1] * 100 for t in predictions]

        plt.figure(figsize=(12, 6))
        plt.barh(range(len(tokens)), probs, color='skyblue')
        plt.yticks(range(len(tokens)), tokens)
        plt.xlabel('Probability (%)')
        plt.title(f'Top {top_k} Next Token Predictions\nPrompt: {prompt[:50]}...')
        plt.gca().invert_yaxis()
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved to {save_path}")
        else:
            plt.show()

    def analyze_generation_probabilities(
        self,
        prompt: str,
        num_tokens: int = 20
    ) -> List[Tuple[str, float]]:
        """
        Generate tokens and track their probabilities.

        Shows how confident the model is for each token it generates.
        """
        input_ids = self.tokenizer.encode(prompt)
        input_tensor = torch.tensor([input_ids], dtype=torch.long, device=self.device)

        generated = input_tensor.clone()
        token_probs = []

        with torch.no_grad():
            for _ in range(num_tokens):
                logits, _ = self.model(generated)
                next_logits = logits[0, -1, :]
                probs = F.softmax(next_logits, dim=-1)

                # Sample next token
                next_token = torch.multinomial(probs, num_samples=1)
                next_prob = probs[next_token.item()].item()

                # Decode token
                token_str = self.tokenizer.decode([next_token.item()])
                token_probs.append((token_str, next_prob))

                # Add to sequence
                generated = torch.cat([generated, next_token.unsqueeze(0)], dim=1)

        return token_probs

    def visualize_generation_confidence(
        self,
        prompt: str,
        num_tokens: int = 30,
        save_path: str = None
    ):
        """Visualize model confidence during generation."""
        token_probs = self.analyze_generation_probabilities(prompt, num_tokens)

        tokens = [repr(t[0])[:15] for t in token_probs]  # Truncate long tokens
        probs = [t[1] * 100 for t in token_probs]

        plt.figure(figsize=(14, 6))
        plt.plot(probs, marker='o', linewidth=2, markersize=6)
        plt.xlabel('Token Position')
        plt.ylabel('Probability (%)')
        plt.title(f'Model Confidence During Generation\nPrompt: {prompt[:50]}...')
        plt.grid(True, alpha=0.3)
        plt.axhline(y=50, color='r', linestyle='--', alpha=0.5, label='50% threshold')
        plt.legend()
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved to {save_path}")
        else:
            plt.show()

    def analyze_vocabulary_usage(
        self,
        prompts: List[str],
        num_generations: int = 3
    ) -> Counter:
        """Analyze which tokens the model uses most frequently."""
        all_tokens = []

        for prompt in prompts:
            for _ in range(num_generations):
                # Generate
                input_ids = self.tokenizer.encode(prompt)
                input_tensor = torch.tensor([input_ids], device=self.device)
                generated = input_tensor.clone()

                with torch.no_grad():
                    for _ in range(50):
                        logits, _ = self.model(generated)
                        next_logits = logits[0, -1, :]
                        probs = F.softmax(next_logits, dim=-1)
                        next_token = torch.multinomial(probs, num_samples=1)
                        generated = torch.cat([generated, next_token.unsqueeze(0)], dim=1)

                # Collect tokens
                generated_ids = generated[0].tolist()
                all_tokens.extend(generated_ids)

        return Counter(all_tokens)

    def visualize_vocabulary_usage(
        self,
        prompts: List[str],
        top_k: int = 30,
        save_path: str = None
    ):
        """Visualize most frequently generated tokens."""
        token_counts = self.analyze_vocabulary_usage(prompts)

        # Get top k
        top_tokens = token_counts.most_common(top_k)

        # Decode tokens
        tokens = [repr(self.tokenizer.decode([t[0]]))[:20] for t in top_tokens]
        counts = [t[1] for t in top_tokens]

        plt.figure(figsize=(12, 8))
        plt.barh(range(len(tokens)), counts, color='coral')
        plt.yticks(range(len(tokens)), tokens, fontsize=9)
        plt.xlabel('Frequency')
        plt.title(f'Top {top_k} Most Frequently Generated Tokens')
        plt.gca().invert_yaxis()
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved to {save_path}")
        else:
            plt.show()

    def compare_prompts(
        self,
        prompts: List[str],
        save_path: str = None
    ):
        """Compare model predictions for different prompts."""
        fig, axes = plt.subplots(len(prompts), 1, figsize=(10, 4*len(prompts)))

        if len(prompts) == 1:
            axes = [axes]

        for idx, prompt in enumerate(prompts):
            predictions = self.analyze_token_probabilities(prompt, top_k=10)

            tokens = [repr(t[0])[:15] for t in predictions]
            probs = [t[1] * 100 for t in predictions]

            axes[idx].barh(range(len(tokens)), probs, color='lightblue')
            axes[idx].set_yticks(range(len(tokens)))
            axes[idx].set_yticklabels(tokens)
            axes[idx].set_xlabel('Probability (%)')
            axes[idx].set_title(f'Prompt: {prompt[:60]}...')
            axes[idx].invert_yaxis()

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved to {save_path}")
        else:
            plt.show()

    def analyze_model_statistics(self):
        """Analyze overall model statistics."""
        stats = {
            'total_parameters': sum(p.numel() for p in self.model.parameters()),
            'trainable_parameters': sum(p.numel() for p in self.model.parameters() if p.requires_grad),
            'vocab_size': self.config.vocab_size,
            'num_layers': self.config.n_layers,
            'd_model': self.config.d_model,
            'n_heads': self.config.n_heads,
            'd_ff': self.config.d_ff,
            'max_seq_len': self.config.max_seq_len,
        }

        # Calculate memory usage
        param_size = sum(p.numel() * p.element_size() for p in self.model.parameters())
        buffer_size = sum(b.numel() * b.element_size() for b in self.model.buffers())
        total_size_mb = (param_size + buffer_size) / (1024 ** 2)

        stats['model_size_mb'] = total_size_mb

        return stats

    def print_model_summary(self):
        """Print comprehensive model summary."""
        stats = self.analyze_model_statistics()

        print("\n" + "="*60)
        print("MODEL SUMMARY")
        print("="*60)

        print(f"\nArchitecture:")
        print(f"  Layers: {stats['num_layers']}")
        print(f"  Hidden Size: {stats['d_model']}")
        print(f"  Attention Heads: {stats['n_heads']}")
        print(f"  FFN Size: {stats['d_ff']}")
        print(f"  Max Sequence Length: {stats['max_seq_len']}")

        print(f"\nVocabulary:")
        print(f"  Vocab Size: {stats['vocab_size']:,}")

        print(f"\nParameters:")
        print(f"  Total: {stats['total_parameters']:,}")
        print(f"  Trainable: {stats['trainable_parameters']:,}")
        print(f"  Model Size: {stats['model_size_mb']:.1f} MB")

        print("\n" + "="*60)


def main():
    parser = argparse.ArgumentParser(description='Model interpretability tools')
    parser.add_argument('--model', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--tokenizer', type=str,
                        default='models/language/language_tokenizer.json',
                        help='Path to tokenizer')
    parser.add_argument('--output-dir', type=str, default='interpretability_output',
                        help='Directory for output visualizations')

    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Setup device
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')

    print(f"Using device: {device}")

    # Initialize interpreter
    interpreter = ModelInterpreter(args.model, args.tokenizer, device)

    # Print model summary
    interpreter.print_model_summary()

    # Test prompts
    test_prompts = [
        "#!/bin/bash\n# Create a backup",
        "#!/bin/bash\n# Monitor system",
        "#!/bin/bash\n# Deploy application",
    ]

    print("\n" + "="*60)
    print("GENERATING VISUALIZATIONS")
    print("="*60)

    # 1. Next token predictions
    print("\n1. Analyzing next token predictions...")
    for i, prompt in enumerate(test_prompts):
        save_path = os.path.join(args.output_dir, f'next_token_{i+1}.png')
        interpreter.visualize_next_token_predictions(prompt, top_k=15, save_path=save_path)

    # 2. Generation confidence
    print("\n2. Analyzing generation confidence...")
    save_path = os.path.join(args.output_dir, 'generation_confidence.png')
    interpreter.visualize_generation_confidence(test_prompts[0], num_tokens=30, save_path=save_path)

    # 3. Vocabulary usage
    print("\n3. Analyzing vocabulary usage...")
    save_path = os.path.join(args.output_dir, 'vocabulary_usage.png')
    interpreter.visualize_vocabulary_usage(test_prompts, top_k=30, save_path=save_path)

    # 4. Prompt comparison
    print("\n4. Comparing prompts...")
    save_path = os.path.join(args.output_dir, 'prompt_comparison.png')
    interpreter.compare_prompts(test_prompts, save_path=save_path)

    print("\n" + "="*60)
    print(f"✓ All visualizations saved to: {args.output_dir}")
    print("="*60)

    # Print sample analysis
    print("\n" + "="*60)
    print("SAMPLE ANALYSIS")
    print("="*60)

    prompt = test_prompts[0]
    print(f"\nPrompt: {prompt}")
    print("\nTop 10 Next Token Predictions:")

    predictions = interpreter.analyze_token_probabilities(prompt, top_k=10)
    for i, (token, prob) in enumerate(predictions, 1):
        print(f"  {i}. {repr(token):20s} - {prob*100:5.2f}%")

    print("\n" + "="*60)
    print("✓ Analysis complete!")
    print("="*60)


if __name__ == '__main__':
    main()
