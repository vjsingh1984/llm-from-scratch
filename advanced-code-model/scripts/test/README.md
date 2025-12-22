# Testing Scripts

Quality assurance and testing scripts.

## Quick Start

```bash
# Test model quality
python ../test_model_quality.py

# Test Stage 2 specifically
python ../test_stage2.py

# Security scan
python ../security_scanner.py
```

## Available Scripts

| Script | Description |
|--------|-------------|
| `test_model_quality.py` | Comprehensive quality tests |
| `test_stage2.py` | Stage 2 specific tests |
| `security_scanner.py` | Scan for security issues |

## Quality Metrics

- **Syntactic Correctness**: Does generated code parse?
- **Semantic Correctness**: Does it do what was asked?
- **Style Compliance**: Follows best practices?
- **Security**: No obvious vulnerabilities?

## Running All Tests

```bash
# Run full test suite
python ../test_model_quality.py --comprehensive

# Quick smoke test
python ../test_model_quality.py --quick
```
