"""
Pytest configuration and shared fixtures.

This module provides fixtures that can be used across all tests.
"""

import os
import sys
import tempfile
import pytest
import torch

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.tokenizer.bpe import BPETokenizer
from src.model.transformer import CodeTransformer
from src.model.config import CoderConfig


@pytest.fixture(scope="session")
def sample_texts():
    """Sample texts for testing."""
    return [
        "#!/bin/bash\necho 'Hello World'",
        "#!/bin/bash\nfor i in {1..10}; do\n  echo $i\ndone",
        "#!/bin/bash\nif [ -f /tmp/test ]; then\n  cat /tmp/test\nfi",
        "#!/bin/bash\nwhile true; do\n  sleep 1\ndone",
        "#!/bin/bash\nfunction backup() {\n  tar -czf backup.tar.gz /data\n}",
    ]


@pytest.fixture(scope="session")
def small_vocab_tokenizer(sample_texts):
    """
    Trained tokenizer with small vocabulary for testing.

    This fixture is session-scoped to avoid retraining for every test.
    """
    tokenizer = BPETokenizer()
    tokenizer.target_vocab_size = 200  # Small vocab for fast tests
    tokenizer.train(sample_texts, verbose=False)
    return tokenizer


@pytest.fixture
def temp_dir():
    """Temporary directory for test files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


@pytest.fixture
def device():
    """Get the best available device for testing."""
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif torch.backends.mps.is_available():
        # Note: MPS can be flaky in tests, use CPU for stability
        return torch.device('cpu')
    else:
        return torch.device('cpu')


@pytest.fixture
def tiny_model_config(small_vocab_tokenizer):
    """Tiny model configuration for fast tests."""
    return CoderConfig(
        vocab_size=len(small_vocab_tokenizer.vocab),
        n_layers=2,
        d_model=64,
        n_heads=2,
        d_ff=128,
        max_seq_len=128,
        dropout=0.1
    )


@pytest.fixture
def tiny_model(tiny_model_config, device):
    """Tiny transformer model for testing."""
    model = CodeTransformer(tiny_model_config)
    model = model.to(device)
    model.eval()
    return model


@pytest.fixture
def sample_batch(small_vocab_tokenizer, device):
    """Sample batch of tokenized data for testing."""
    texts = [
        "#!/bin/bash\necho test",
        "#!/bin/bash\nls -la"
    ]

    # Tokenize
    tokenized = [small_vocab_tokenizer.encode(text) for text in texts]

    # Pad to same length
    max_len = max(len(t) for t in tokenized)
    padded = []
    for tokens in tokenized:
        if len(tokens) < max_len:
            tokens = tokens + [0] * (max_len - len(tokens))
        padded.append(tokens[:max_len])

    # Convert to tensor
    batch = torch.tensor(padded, dtype=torch.long, device=device)
    return batch


# Pytest configuration
def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )
    config.addinivalue_line(
        "markers", "gpu: marks tests that require GPU"
    )


# Skip GPU tests if CUDA not available
def pytest_collection_modifyitems(config, items):
    """Automatically skip GPU tests if CUDA is not available."""
    if not torch.cuda.is_available():
        skip_gpu = pytest.mark.skip(reason="GPU not available")
        for item in items:
            if "gpu" in item.keywords:
                item.add_marker(skip_gpu)
