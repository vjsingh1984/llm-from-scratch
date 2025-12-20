# Testing Infrastructure

Comprehensive test suite for the Code LLM project.

## Quick Start

```bash
# Install test dependencies
pip install pytest pytest-cov pytest-mock

# Run all tests
pytest

# Run with coverage
pytest --cov=src --cov-report=html

# Run specific test file
pytest tests/test_tokenizer.py

# Run specific test
pytest tests/test_tokenizer.py::test_bpe_encode_decode
```

## Test Structure

```
tests/
├── README.md                    # This file
├── conftest.py                  # Shared fixtures
├── test_tokenizer.py            # Tokenizer tests
├── test_model.py                # Model architecture tests
├── test_training.py             # Training utilities tests
├── test_generation.py           # Generation tests
└── integration/
    ├── test_end_to_end.py       # End-to-end tests
    └── test_api.py              # API tests
```

## Test Coverage

### Unit Tests

- **Tokenizer** (`test_tokenizer.py`)
  - BPE encoding/decoding
  - Vocabulary management
  - Special tokens handling
  - Save/load functionality

- **Model** (`test_model.py`)
  - Architecture initialization
  - Forward pass
  - Parameter counts
  - Attention masking

- **Training** (`test_training.py`)
  - Data loading
  - Optimizer configuration
  - Loss computation

- **Generation** (`test_generation.py`)
  - Text generation
  - Sampling strategies
  - Temperature effects

### Integration Tests

- **End-to-End** (`test_end_to_end.py`)
  - Full training pipeline
  - Model save/load
  - Generation from trained model

- **API** (`test_api.py`)
  - API endpoints
  - Request validation
  - Error handling

## Running Tests

### All Tests

```bash
pytest
```

### With Verbose Output

```bash
pytest -v
```

### With Coverage Report

```bash
pytest --cov=src --cov-report=term-missing
```

### Generate HTML Coverage Report

```bash
pytest --cov=src --cov-report=html
open htmlcov/index.html
```

### Run Only Fast Tests

```bash
pytest -m "not slow"
```

### Run Only Integration Tests

```bash
pytest tests/integration/
```

## Continuous Integration

### GitHub Actions

Create `.github/workflows/tests.yml`:

```yaml
name: Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v2
    - name: Set up Python
      uses: actions/setup-python@v2
      with:
        python-version: '3.10'
    - name: Install dependencies
      run: |
        pip install -r requirements.txt
        pip install pytest pytest-cov
    - name: Run tests
      run: pytest --cov=src
```

## Test Data

Test fixtures use minimal data for fast execution:
- Small vocabulary (100-500 tokens)
- Tiny model (1-2 layers, 64-128 dimensions)
- Short sequences (10-50 tokens)

This ensures tests run quickly while still validating functionality.

## Writing New Tests

### Example Test

```python
import pytest
from src.tokenizer.bpe import BPETokenizer

def test_tokenizer_encode_decode():
    """Test that encoding and decoding are inverse operations."""
    tokenizer = BPETokenizer()

    # Train on sample data
    texts = ["hello world", "test data"]
    tokenizer.target_vocab_size = 100
    tokenizer.train(texts, verbose=False)

    # Test encode/decode
    text = "hello world"
    encoded = tokenizer.encode(text)
    decoded = tokenizer.decode(encoded)

    assert decoded == text, "Decoded text should match original"
```

### Using Fixtures

```python
@pytest.fixture
def trained_tokenizer():
    """Fixture providing a trained tokenizer."""
    tokenizer = BPETokenizer()
    texts = ["sample", "data", "for", "testing"]
    tokenizer.target_vocab_size = 50
    tokenizer.train(texts, verbose=False)
    return tokenizer

def test_with_fixture(trained_tokenizer):
    """Test using the fixture."""
    encoded = trained_tokenizer.encode("sample")
    assert len(encoded) > 0
```

## Best Practices

1. **Test Naming**: Use descriptive names (`test_<what>_<condition>_<expected>`)
2. **Isolation**: Each test should be independent
3. **Fast Tests**: Keep unit tests under 1 second
4. **Clear Assertions**: Use informative assertion messages
5. **Mock External Resources**: Don't rely on network or large files
6. **Coverage**: Aim for >80% code coverage

## Troubleshooting

### Tests Fail on Import

```bash
# Add src to PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
pytest
```

### Slow Tests

```bash
# Run with timing
pytest --durations=10
```

### Debug Failed Test

```bash
# Run with pdb
pytest --pdb

# Show print statements
pytest -s
```

---

**Happy testing!** Comprehensive tests ensure code quality and catch regressions early.
