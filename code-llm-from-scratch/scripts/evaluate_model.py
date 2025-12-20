#!/usr/bin/env python3
"""
Model Evaluation Script

Comprehensive evaluation of trained models including:
- Syntax correctness
- Pattern matching
- Code quality
- Comparative analysis

Usage:
    python scripts/evaluate_model.py --model models/code/code_model_final.pt
    python scripts/evaluate_model.py --model models/code/code_model_final.pt --test-file data/test_prompts.json
"""

import os
import sys
import json
import argparse
import subprocess
import tempfile
from typing import List, Dict
from datetime import datetime

import torch
import torch.nn.functional as F
from tqdm import tqdm

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.model.transformer import CodeTransformer
from src.tokenizer.bpe import BPETokenizer


class ModelEvaluator:
    """Comprehensive model evaluation."""

    def __init__(self, model_path: str, tokenizer_path: str, device: torch.device):
        self.model_path = model_path
        self.tokenizer_path = tokenizer_path
        self.device = device

        # Load model and tokenizer
        print(f"Loading model from {model_path}...")
        checkpoint = torch.load(model_path, map_location=device)

        self.config = checkpoint['config']
        self.model = CodeTransformer(self.config)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model = self.model.to(device)
        self.model.eval()

        print(f"Loading tokenizer from {tokenizer_path}...")
        self.tokenizer = BPETokenizer()
        self.tokenizer.load(tokenizer_path)

        print("Model loaded successfully!\n")

    def generate_code(
        self,
        prompt: str,
        max_length: int = 200,
        temperature: float = 0.8,
        top_k: int = 50
    ) -> str:
        """Generate code from prompt."""
        input_ids = self.tokenizer.encode(prompt)
        if not input_ids:
            return prompt

        input_tensor = torch.tensor([input_ids], dtype=torch.long, device=self.device)
        generated = input_tensor.clone()

        with torch.no_grad():
            for _ in range(max_length):
                logits, _ = self.model(generated)
                next_logits = logits[0, -1, :] / temperature

                # Top-k filtering
                if top_k > 0:
                    indices_to_remove = next_logits < torch.topk(next_logits, top_k)[0][..., -1, None]
                    next_logits[indices_to_remove] = float('-inf')

                probs = F.softmax(next_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                generated = torch.cat([generated, next_token.unsqueeze(0)], dim=1)

                # Early stopping for very long generations
                if generated.shape[1] > 500:
                    break

        generated_ids = generated[0].tolist()
        return self.tokenizer.decode(generated_ids)

    def check_bash_syntax(self, code: str) -> bool:
        """Check if bash code is syntactically valid."""
        try:
            result = subprocess.run(
                ['bash', '-n'],
                input=code.encode(),
                capture_output=True,
                timeout=5
            )
            return result.returncode == 0
        except Exception:
            return False

    def check_patterns(self, code: str, prompt: str) -> Dict[str, bool]:
        """Check for common bash patterns."""
        patterns = {
            'has_shebang': code.startswith('#!/bin/bash') or code.startswith('#!/usr/bin/env bash'),
            'has_comments': '#' in code and not code.strip().startswith('#!'),
            'has_variables': '$' in code or '${' in code,
            'has_conditionals': any(kw in code for kw in ['if ', 'then', 'fi', '[ ', '[[ ']),
            'has_loops': any(kw in code for kw in ['for ', 'while ', 'do', 'done']),
            'has_functions': 'function ' in code or '() {' in code,
            'has_error_handling': any(kw in code for kw in ['set -e', 'trap', 'exit ', '|| ']),
            'has_output': any(kw in code for kw in ['echo', 'printf', 'cat']),
        }

        # Prompt-specific checks
        prompt_lower = prompt.lower()
        if 'backup' in prompt_lower:
            patterns['backup_relevant'] = any(kw in code.lower() for kw in ['tar', 'rsync', 'cp', 'backup', 'archive'])
        if 'monitor' in prompt_lower:
            patterns['monitor_relevant'] = any(kw in code.lower() for kw in ['top', 'df', 'free', 'ps', 'uptime'])
        if 'deploy' in prompt_lower:
            patterns['deploy_relevant'] = any(kw in code.lower() for kw in ['git', 'docker', 'systemctl', 'service', 'restart'])
        if 'log' in prompt_lower:
            patterns['log_relevant'] = any(kw in code.lower() for kw in ['log', 'journalctl', 'tail', 'grep'])

        return patterns

    def evaluate_syntax(self, prompts: List[str]) -> Dict:
        """Evaluate syntax correctness."""
        print("\n" + "="*60)
        print("SYNTAX EVALUATION")
        print("="*60)

        valid_count = 0
        results = []

        for prompt in tqdm(prompts, desc="Checking syntax"):
            code = self.generate_code(prompt)
            is_valid = self.check_bash_syntax(code)

            if is_valid:
                valid_count += 1

            results.append({
                'prompt': prompt,
                'code': code,
                'valid': is_valid,
                'length': len(code)
            })

        accuracy = valid_count / len(prompts) * 100 if prompts else 0

        print(f"\nSyntax Accuracy: {accuracy:.1f}% ({valid_count}/{len(prompts)})")

        return {
            'syntax_accuracy': accuracy,
            'valid_count': valid_count,
            'total_count': len(prompts),
            'results': results
        }

    def evaluate_patterns(self, prompts: List[str]) -> Dict:
        """Evaluate pattern matching."""
        print("\n" + "="*60)
        print("PATTERN EVALUATION")
        print("="*60)

        all_patterns = []

        for prompt in tqdm(prompts, desc="Checking patterns"):
            code = self.generate_code(prompt)
            patterns = self.check_patterns(code, prompt)
            all_patterns.append(patterns)

        # Aggregate statistics
        pattern_stats = {}
        for key in all_patterns[0].keys():
            count = sum(1 for p in all_patterns if p[key])
            pattern_stats[key] = {
                'percentage': count / len(all_patterns) * 100,
                'count': count,
                'total': len(all_patterns)
            }

        print("\nPattern Statistics:")
        for pattern, stats in pattern_stats.items():
            print(f"  {pattern}: {stats['percentage']:.1f}% ({stats['count']}/{stats['total']})")

        return {
            'pattern_stats': pattern_stats,
            'all_patterns': all_patterns
        }

    def evaluate_diversity(self, prompt: str, num_samples: int = 5) -> Dict:
        """Evaluate generation diversity."""
        print("\n" + "="*60)
        print("DIVERSITY EVALUATION")
        print("="*60)

        print(f"Generating {num_samples} samples for prompt:")
        print(f"  {prompt[:60]}...")

        generations = []
        for i in range(num_samples):
            code = self.generate_code(prompt, temperature=0.8)
            generations.append(code)

        # Calculate pairwise similarities
        import difflib
        similarities = []
        for i in range(len(generations)):
            for j in range(i+1, len(generations)):
                similarity = difflib.SequenceMatcher(None, generations[i], generations[j]).ratio()
                similarities.append(similarity * 100)

        avg_similarity = sum(similarities) / len(similarities) if similarities else 0
        diversity_score = 100 - avg_similarity

        print(f"\nDiversity Score: {diversity_score:.1f}")
        print(f"Average Similarity: {avg_similarity:.1f}%")
        print(f"Unique Generations: {len(set(generations))}/{num_samples}")

        return {
            'diversity_score': diversity_score,
            'avg_similarity': avg_similarity,
            'unique_count': len(set(generations)),
            'total_samples': num_samples,
            'generations': generations
        }

    def generate_report(self, results: Dict, output_path: str):
        """Generate evaluation report."""
        report = {
            'timestamp': datetime.now().isoformat(),
            'model_path': self.model_path,
            'tokenizer_path': self.tokenizer_path,
            'model_config': {
                'vocab_size': self.config.vocab_size,
                'n_layers': self.config.n_layers,
                'd_model': self.config.d_model,
                'n_heads': self.config.n_heads,
            },
            'results': results
        }

        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)

        print(f"\n{'='*60}")
        print(f"Report saved to: {output_path}")
        print(f"{'='*60}")


def load_test_prompts(test_file: str) -> List[str]:
    """Load test prompts from file."""
    if test_file.endswith('.json'):
        with open(test_file) as f:
            data = json.load(f)
            # Handle different JSON formats
            if isinstance(data, list):
                return data
            elif isinstance(data, dict):
                # Flatten all categories
                prompts = []
                for category in data.values():
                    if isinstance(category, list):
                        prompts.extend(category)
                return prompts
    else:
        # Plain text file, one prompt per line
        with open(test_file) as f:
            return [line.strip() for line in f if line.strip()]


def get_default_test_prompts() -> List[str]:
    """Get default test prompts."""
    return [
        "#!/bin/bash\n# Create a backup script",
        "#!/bin/bash\n# Monitor system resources",
        "#!/bin/bash\n# Deploy application to server",
        "#!/bin/bash\n# Check disk space and alert",
        "#!/bin/bash\n# Rotate log files",
        "#!/bin/bash\n# Database backup with compression",
        "#!/bin/bash\n# Update system packages",
        "#!/bin/bash\n# Create user accounts from file",
        "#!/bin/bash\n# Network connectivity test",
        "#!/bin/bash\n# Process cleanup script",
        "#!/bin/bash\n# Automated deployment pipeline",
        "#!/bin/bash\n# System health check",
        "#!/bin/bash\n# Certificate renewal script",
        "#!/bin/bash\n# Docker container management",
        "#!/bin/bash\n# Git repository backup",
    ]


def main():
    parser = argparse.ArgumentParser(description='Evaluate trained model')
    parser.add_argument('--model', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--tokenizer', type=str,
                        default='models/language/language_tokenizer.json',
                        help='Path to tokenizer')
    parser.add_argument('--test-file', type=str,
                        help='Path to test prompts file (JSON or text)')
    parser.add_argument('--output', type=str,
                        default='evaluation_results.json',
                        help='Output path for results')
    parser.add_argument('--eval-syntax', action='store_true', default=True,
                        help='Evaluate syntax correctness')
    parser.add_argument('--eval-patterns', action='store_true', default=True,
                        help='Evaluate pattern matching')
    parser.add_argument('--eval-diversity', action='store_true',
                        help='Evaluate generation diversity')

    args = parser.parse_args()

    # Setup device
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')

    print(f"Using device: {device}\n")

    # Initialize evaluator
    evaluator = ModelEvaluator(args.model, args.tokenizer, device)

    # Load test prompts
    if args.test_file:
        print(f"Loading test prompts from {args.test_file}...")
        test_prompts = load_test_prompts(args.test_file)
    else:
        print("Using default test prompts...")
        test_prompts = get_default_test_prompts()

    print(f"Loaded {len(test_prompts)} test prompts\n")

    # Run evaluations
    results = {}

    if args.eval_syntax:
        results['syntax'] = evaluator.evaluate_syntax(test_prompts)

    if args.eval_patterns:
        results['patterns'] = evaluator.evaluate_patterns(test_prompts)

    if args.eval_diversity:
        # Use first prompt for diversity test
        results['diversity'] = evaluator.evaluate_diversity(test_prompts[0], num_samples=5)

    # Generate report
    evaluator.generate_report(results, args.output)

    # Print summary
    print("\n" + "="*60)
    print("EVALUATION SUMMARY")
    print("="*60)

    if 'syntax' in results:
        print(f"\nSyntax Accuracy: {results['syntax']['syntax_accuracy']:.1f}%")

    if 'patterns' in results:
        print(f"\nKey Patterns:")
        stats = results['patterns']['pattern_stats']
        for pattern in ['has_shebang', 'has_comments', 'has_error_handling']:
            if pattern in stats:
                print(f"  {pattern}: {stats[pattern]['percentage']:.1f}%")

    if 'diversity' in results:
        print(f"\nDiversity Score: {results['diversity']['diversity_score']:.1f}")

    print("\n" + "="*60)
    print("✓ Evaluation complete!")
    print("="*60)


if __name__ == '__main__':
    main()
