#!/usr/bin/env python3
"""
Model Quality Testing
Automated tests for code generation quality benchmarking
"""

import os
import sys
import json
import time
from dataclasses import dataclass
from typing import List, Dict, Optional
import subprocess

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

@dataclass
class TestCase:
    """A test case for model evaluation"""
    name: str
    prompt: str
    expected_patterns: List[str]  # Patterns that should appear in output
    forbidden_patterns: List[str] = None  # Patterns that should NOT appear
    language: str = "bash"
    max_tokens: int = 256

@dataclass
class TestResult:
    """Result of a single test"""
    name: str
    passed: bool
    generated_text: str
    matched_patterns: List[str]
    missing_patterns: List[str]
    forbidden_found: List[str]
    generation_time_ms: float
    error: Optional[str] = None

class ModelQualityTester:
    """Tests model quality against benchmark test cases"""

    def __init__(self, model_path: str, tokenizer_path: str = "data/tokenizer/tokenizer.json"):
        self.model_path = model_path
        self.tokenizer_path = tokenizer_path
        self.model = None
        self.tokenizer = None
        self.device = None

    def load_model(self):
        """Load model and tokenizer"""
        import torch
        from tokenizers import Tokenizer
        from src.model import create_model
        from src.model.config import ModelConfig

        # Determine device
        if torch.backends.mps.is_available():
            self.device = 'mps'
        elif torch.cuda.is_available():
            self.device = 'cuda'
        else:
            self.device = 'cpu'

        print(f"Loading model on {self.device}...")

        # Load tokenizer
        self.tokenizer = Tokenizer.from_file(self.tokenizer_path)

        # Create model
        config = ModelConfig(
            vocab_size=self.tokenizer.get_vocab_size(),
            hidden_size=1024,
            num_layers=24,
            num_heads=16,
            intermediate_size=4096,
            max_position_embeddings=2048,
            use_rmsnorm=True,
            use_rope=True,
        )

        self.model = create_model(config, device=self.device)

        # Load checkpoint
        checkpoint = torch.load(self.model_path, map_location=self.device, weights_only=False)
        if 'model_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.model.load_state_dict(checkpoint)

        self.model.eval()
        print("Model loaded successfully")

    def generate(self, prompt: str, max_tokens: int = 256, temperature: float = 0.7) -> str:
        """Generate text from prompt"""
        import torch

        encoded = self.tokenizer.encode(prompt)
        input_ids = torch.tensor([encoded.ids], device=self.device)

        generated_ids = input_ids.clone()

        with torch.no_grad():
            for _ in range(max_tokens):
                outputs = self.model(generated_ids)
                next_token_logits = outputs[:, -1, :] / temperature

                # Sample
                probs = torch.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)

                generated_ids = torch.cat([generated_ids, next_token], dim=-1)

                # Check for EOS
                if next_token.item() == self.tokenizer.token_to_id("[EOS]"):
                    break

        generated_text = self.tokenizer.decode(generated_ids[0].tolist())
        return generated_text[len(prompt):]

    def run_test(self, test: TestCase) -> TestResult:
        """Run a single test case"""
        import re

        start_time = time.time()
        error = None
        generated_text = ""

        try:
            generated_text = self.generate(test.prompt, test.max_tokens)
        except Exception as e:
            error = str(e)

        generation_time = (time.time() - start_time) * 1000

        # Check expected patterns
        matched = []
        missing = []
        for pattern in test.expected_patterns:
            if re.search(pattern, generated_text, re.IGNORECASE):
                matched.append(pattern)
            else:
                missing.append(pattern)

        # Check forbidden patterns
        forbidden_found = []
        if test.forbidden_patterns:
            for pattern in test.forbidden_patterns:
                if re.search(pattern, generated_text, re.IGNORECASE):
                    forbidden_found.append(pattern)

        passed = len(missing) == 0 and len(forbidden_found) == 0 and error is None

        return TestResult(
            name=test.name,
            passed=passed,
            generated_text=generated_text,
            matched_patterns=matched,
            missing_patterns=missing,
            forbidden_found=forbidden_found,
            generation_time_ms=generation_time,
            error=error
        )

    def run_all_tests(self, tests: List[TestCase]) -> List[TestResult]:
        """Run all test cases"""
        results = []
        for i, test in enumerate(tests, 1):
            print(f"Running test {i}/{len(tests)}: {test.name}...")
            result = self.run_test(test)
            results.append(result)
            status = "PASS" if result.passed else "FAIL"
            print(f"  {status} ({result.generation_time_ms:.2f}ms)")
        return results

    def generate_report(self, results: List[TestResult]) -> Dict:
        """Generate test report"""
        passed = sum(1 for r in results if r.passed)
        failed = len(results) - passed

        return {
            "summary": {
                "total": len(results),
                "passed": passed,
                "failed": failed,
                "pass_rate": passed / len(results) if results else 0
            },
            "results": [
                {
                    "name": r.name,
                    "passed": r.passed,
                    "generation_time_ms": r.generation_time_ms,
                    "matched_patterns": r.matched_patterns,
                    "missing_patterns": r.missing_patterns,
                    "forbidden_found": r.forbidden_found,
                    "error": r.error
                }
                for r in results
            ]
        }


# ============================================================
# Test Cases
# ============================================================

def get_bash_test_cases() -> List[TestCase]:
    """Get Bash/Shell test cases"""
    return [
        TestCase(
            name="basic_echo",
            prompt="#!/bin/bash\n# Print hello world\necho",
            expected_patterns=[r"echo", r"hello|world|Hello|World"],
            language="bash"
        ),
        TestCase(
            name="for_loop",
            prompt="#!/bin/bash\n# Loop through numbers 1 to 5\nfor",
            expected_patterns=[r"for", r"do", r"done"],
            language="bash"
        ),
        TestCase(
            name="if_statement",
            prompt="#!/bin/bash\n# Check if file exists\nif",
            expected_patterns=[r"if", r"then", r"fi"],
            language="bash"
        ),
        TestCase(
            name="function_definition",
            prompt="#!/bin/bash\n# Define a function to print a message\nfunction",
            expected_patterns=[r"function|\(\)", r"\{", r"\}"],
            language="bash"
        ),
        TestCase(
            name="variable_assignment",
            prompt="#!/bin/bash\n# Set a variable with a name\nNAME=",
            expected_patterns=[r"NAME=", r"[\"']?\w+[\"']?"],
            language="bash"
        ),
        TestCase(
            name="file_operations",
            prompt="#!/bin/bash\n# Read a file line by line\nwhile read",
            expected_patterns=[r"while", r"read", r"do", r"done"],
            language="bash"
        ),
        TestCase(
            name="no_dangerous_commands",
            prompt="#!/bin/bash\n# Clean up temporary files\nrm",
            expected_patterns=[r"rm"],
            forbidden_patterns=[r"rm\s+-rf\s+/\s*$", r"rm\s+-rf\s+/\*"],
            language="bash"
        ),
    ]

def get_python_test_cases() -> List[TestCase]:
    """Get Python test cases (if model supports Python)"""
    return [
        TestCase(
            name="python_function",
            prompt="def hello():\n    \"\"\"Print hello world\"\"\"\n    print(",
            expected_patterns=[r"print\(", r"[\"'].*[\"']\)"],
            language="python"
        ),
        TestCase(
            name="python_class",
            prompt="class User:\n    def __init__(self",
            expected_patterns=[r"self", r"def __init__"],
            language="python"
        ),
        TestCase(
            name="python_list_comp",
            prompt="# Create a list of squares\nsquares = [",
            expected_patterns=[r"\[", r"for", r"in", r"\]"],
            language="python"
        ),
    ]

def get_syntax_test_cases() -> List[TestCase]:
    """Test cases for syntax correctness"""
    return [
        TestCase(
            name="balanced_braces",
            prompt="#!/bin/bash\nif [ $x -gt 0 ]; then\n    echo \"positive\"\n",
            expected_patterns=[r"fi"],
            language="bash"
        ),
        TestCase(
            name="balanced_quotes",
            prompt='#!/bin/bash\nMSG="Hello',
            expected_patterns=[r'".*"'],
            language="bash"
        ),
    ]


# ============================================================
# Main
# ============================================================

def main():
    import argparse

    parser = argparse.ArgumentParser(description="Test model quality")
    parser.add_argument("--model", default="models/code_model_best.pth",
                        help="Path to model checkpoint")
    parser.add_argument("--tokenizer", default="data/tokenizer/tokenizer.json",
                        help="Path to tokenizer")
    parser.add_argument("--output", default="test_results.json",
                        help="Output file for results")
    parser.add_argument("--language", choices=["bash", "python", "all"], default="bash",
                        help="Which language tests to run")
    parser.add_argument("--threshold", type=float, default=0.7,
                        help="Minimum pass rate to succeed (0.0-1.0)")
    args = parser.parse_args()

    # Check model exists
    if not os.path.exists(args.model):
        print(f"ERROR: Model not found: {args.model}")
        exit(1)

    if not os.path.exists(args.tokenizer):
        print(f"ERROR: Tokenizer not found: {args.tokenizer}")
        exit(1)

    # Load test cases
    tests = []
    if args.language in ["bash", "all"]:
        tests.extend(get_bash_test_cases())
    if args.language in ["python", "all"]:
        tests.extend(get_python_test_cases())
    tests.extend(get_syntax_test_cases())

    print(f"Loaded {len(tests)} test cases")

    # Initialize tester
    tester = ModelQualityTester(args.model, args.tokenizer)
    tester.load_model()

    # Run tests
    print("\n" + "=" * 60)
    print("RUNNING MODEL QUALITY TESTS")
    print("=" * 60 + "\n")

    results = tester.run_all_tests(tests)
    report = tester.generate_report(results)

    # Print summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    print(f"Total:  {report['summary']['total']}")
    print(f"Passed: {report['summary']['passed']}")
    print(f"Failed: {report['summary']['failed']}")
    print(f"Rate:   {report['summary']['pass_rate']*100:.1f}%")

    # Print failed tests
    failed_tests = [r for r in results if not r.passed]
    if failed_tests:
        print("\nFailed Tests:")
        for r in failed_tests:
            print(f"  - {r.name}")
            if r.missing_patterns:
                print(f"    Missing: {r.missing_patterns}")
            if r.forbidden_found:
                print(f"    Forbidden: {r.forbidden_found}")
            if r.error:
                print(f"    Error: {r.error}")

    # Save report
    with open(args.output, 'w') as f:
        json.dump(report, f, indent=2)
    print(f"\nDetailed results saved to: {args.output}")

    # Exit with status
    if report['summary']['pass_rate'] >= args.threshold:
        print(f"\nPASS: Pass rate {report['summary']['pass_rate']*100:.1f}% >= {args.threshold*100:.1f}% threshold")
        exit(0)
    else:
        print(f"\nFAIL: Pass rate {report['summary']['pass_rate']*100:.1f}% < {args.threshold*100:.1f}% threshold")
        exit(1)


if __name__ == "__main__":
    main()
