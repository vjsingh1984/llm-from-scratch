# Contributing to Code LLM from Scratch

Thank you for your interest in contributing! This guide will help you get started with contributing to the project.

## Table of Contents

1. [Getting Started](#getting-started)
2. [Development Setup](#development-setup)
3. [How to Contribute](#how-to-contribute)
4. [Code Style Guide](#code-style-guide)
5. [Testing Requirements](#testing-requirements)
6. [Documentation](#documentation)
7. [Pull Request Process](#pull-request-process)
8. [Community Guidelines](#community-guidelines)

---

## Getting Started

### Ways to Contribute

We welcome contributions of all kinds:

- 🐛 **Bug Reports**: Found a bug? Let us know!
- ✨ **Feature Requests**: Have an idea? We'd love to hear it!
- 📝 **Documentation**: Help improve our docs
- 🧪 **Tests**: Add test coverage
- 💻 **Code**: Fix bugs or implement features
- 🎨 **Examples**: Share interesting use cases
- 📊 **Datasets**: Contribute training data
- 🔬 **Research**: Share experiments and findings

### Good First Issues

Look for issues labeled `good first issue` or `help wanted`. These are great starting points for new contributors.

---

## Development Setup

### 1. Fork and Clone

```bash
# Fork the repository on GitHub, then clone your fork
git clone https://github.com/YOUR_USERNAME/code-llm-from-scratch.git
cd code-llm-from-scratch

# Add upstream remote
git remote add upstream https://github.com/ORIGINAL_OWNER/code-llm-from-scratch.git
```

### 2. Create Development Environment

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install development dependencies
pip install pytest pytest-cov black flake8 mypy
```

### 3. Create a Branch

```bash
# Update your main branch
git checkout main
git pull upstream main

# Create a feature branch
git checkout -b feature/your-feature-name
# or
git checkout -b fix/bug-description
```

### 4. Verify Setup

```bash
# Run tests to ensure everything works
pytest

# Run a quick generation test
python examples/basic_usage.py
```

---

## How to Contribute

### Reporting Bugs

When reporting bugs, please include:

1. **Clear title**: Describe the issue briefly
2. **Description**: What happened vs. what you expected
3. **Steps to reproduce**: How can we recreate the bug?
4. **Environment**: Python version, OS, PyTorch version
5. **Error messages**: Full stack trace if available

**Bug Report Template**:

```markdown
## Description
Brief description of the bug

## Steps to Reproduce
1. Step one
2. Step two
3. ...

## Expected Behavior
What should happen

## Actual Behavior
What actually happens

## Environment
- Python version:
- PyTorch version:
- OS:
- Device (CPU/CUDA/MPS):

## Error Messages
```
Full error trace
```
```

### Suggesting Features

When suggesting features:

1. **Use case**: Why is this feature useful?
2. **Proposed solution**: How should it work?
3. **Alternatives**: Other approaches considered
4. **Examples**: Code examples if applicable

**Feature Request Template**:

```markdown
## Feature Description
What feature would you like to see?

## Use Case
Why is this feature useful?

## Proposed Implementation
How should this work?

## Alternatives Considered
What other approaches did you consider?

## Additional Context
Any other information
```

### Contributing Code

1. **Check existing issues**: Avoid duplicate work
2. **Discuss major changes**: Open an issue first for large changes
3. **Write tests**: All code changes need tests
4. **Update docs**: Document new features
5. **Follow style guide**: See below
6. **Keep PRs focused**: One feature/fix per PR

---

## Code Style Guide

### Python Code Style

We follow **PEP 8** with some modifications.

#### Formatting

```bash
# Format code with Black
black src/ scripts/ examples/ tests/

# Check with flake8
flake8 src/ scripts/ examples/ tests/
```

#### Code Style Rules

```python
# ✅ Good
def train_model(
    model: CodeTransformer,
    data_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    num_epochs: int = 10
) -> Dict[str, float]:
    """
    Train the model.

    Args:
        model: Transformer model to train
        data_loader: Training data loader
        optimizer: Optimization algorithm
        num_epochs: Number of training epochs

    Returns:
        Dictionary with training metrics
    """
    metrics = {}

    for epoch in range(num_epochs):
        total_loss = 0.0

        for batch_idx, (input_ids, target_ids) in enumerate(data_loader):
            optimizer.zero_grad()
            logits, loss = model(input_ids, target_ids)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        metrics[f'epoch_{epoch}_loss'] = total_loss / len(data_loader)

    return metrics


# ❌ Bad
def train(m,d,o,e=10):  # No type hints, unclear names
    # No docstring
    for i in range(e):
        l=0
        for b,(x,y) in enumerate(d):
            o.zero_grad()
            _,loss=m(x,y)
            loss.backward()
            o.step()
            l+=loss.item()
    return l
```

#### Naming Conventions

- **Functions/methods**: `snake_case`
- **Classes**: `PascalCase`
- **Constants**: `UPPER_SNAKE_CASE`
- **Private methods**: `_leading_underscore`
- **Variables**: `snake_case`, descriptive names

#### Documentation

```python
# Module docstring
"""
Training utilities for the Code LLM.

This module provides functions and classes for training transformer models
on code generation tasks.
"""

# Class docstring
class CodeTransformer(nn.Module):
    """
    GPT-style transformer for code generation.

    This model uses a decoder-only architecture with causal masking,
    suitable for autoregressive code generation.

    Args:
        config: Model configuration with architecture hyperparameters

    Attributes:
        token_embedding: Token embedding layer
        pos_embedding: Positional embedding layer
        blocks: List of transformer blocks
        lm_head: Language modeling head
    """

# Function docstring
def generate_code(
    model: CodeTransformer,
    tokenizer: BPETokenizer,
    prompt: str,
    max_length: int = 200
) -> str:
    """
    Generate code from a natural language prompt.

    Args:
        model: Trained transformer model
        tokenizer: BPE tokenizer for encoding/decoding
        prompt: Natural language description of desired code
        max_length: Maximum number of tokens to generate

    Returns:
        Generated code as a string

    Example:
        >>> code = generate_code(model, tokenizer, "Create a backup script")
        >>> print(code)
        #!/bin/bash
        tar -czf backup.tar.gz /data
    """
```

### Type Hints

Always use type hints for function signatures:

```python
from typing import List, Dict, Optional, Tuple

def process_data(
    texts: List[str],
    max_length: Optional[int] = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Process text data into tensors."""
    ...
```

### Imports

```python
# Standard library imports
import os
import sys
import json
from typing import List, Dict

# Third-party imports
import torch
import numpy as np

# Local imports
from src.model.transformer import CodeTransformer
from src.tokenizer.bpe import BPETokenizer
```

---

## Testing Requirements

### Writing Tests

All new code must include tests. We aim for **>80% code coverage**.

#### Test Structure

```python
# tests/test_new_feature.py
import pytest
import torch
from src.new_feature import new_function


class TestNewFeature:
    """Test suite for new feature."""

    def test_basic_functionality(self):
        """Test basic use case."""
        result = new_function(input_data)
        assert result is not None
        assert isinstance(result, expected_type)

    def test_edge_case(self):
        """Test edge case handling."""
        with pytest.raises(ValueError):
            new_function(invalid_input)

    def test_with_fixture(self, trained_model):
        """Test using a fixture."""
        result = new_function(trained_model)
        assert result.shape == expected_shape


# Use fixtures for common setup
@pytest.fixture
def sample_data():
    """Provide sample data for tests."""
    return ["sample", "data", "for", "testing"]
```

#### Running Tests

```bash
# Run all tests
pytest

# Run specific test file
pytest tests/test_tokenizer.py

# Run with coverage
pytest --cov=src --cov-report=html

# Run specific test
pytest tests/test_tokenizer.py::test_encode_decode
```

#### Test Requirements

- ✅ Test normal use cases
- ✅ Test edge cases
- ✅ Test error handling
- ✅ Use descriptive test names
- ✅ Keep tests fast (< 1 second each)
- ✅ Use fixtures for common setup
- ✅ Mock external dependencies

---

## Documentation

### Types of Documentation

1. **Code comments**: Explain *why*, not *what*
2. **Docstrings**: Document all public functions/classes
3. **README updates**: For new features
4. **Guides**: For major features (see `docs/`)
5. **Examples**: Show how to use new features

### Writing Good Documentation

```python
# ✅ Good comment - explains WHY
# Use lower learning rate for fine-tuning to prevent catastrophic forgetting
learning_rate = 1e-4

# ❌ Bad comment - just repeats the code
# Set learning rate to 1e-4
learning_rate = 1e-4
```

### Documentation Structure

```
docs/
├── ARCHITECTURE.md     # Model architecture details
├── ADVANCED_TOPICS.md  # Advanced usage
├── DEPLOYMENT.md       # Production deployment
├── EVALUATION.md       # Model evaluation
└── MONITORING.md       # Production monitoring
```

### Adding New Guides

1. Create markdown file in `docs/`
2. Follow existing format
3. Include code examples
4. Add table of contents
5. Link from main README

---

## Pull Request Process

### 1. Prepare Your Changes

```bash
# Make sure tests pass
pytest

# Format code
black src/ tests/

# Check linting
flake8 src/ tests/

# Update documentation if needed
```

### 2. Commit Your Changes

```bash
# Stage changes
git add .

# Commit with descriptive message
git commit -m "feat: Add support for custom tokenizer configurations

- Add TokenizerConfig dataclass
- Update BPETokenizer to use config
- Add tests for new configuration options
- Update documentation in README.md

Closes #123"
```

#### Commit Message Format

```
<type>: <short summary>

<detailed description>

<footer>
```

**Types**:
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `test`: Adding or updating tests
- `refactor`: Code refactoring
- `style`: Formatting changes
- `chore`: Maintenance tasks

### 3. Push and Create PR

```bash
# Push to your fork
git push origin feature/your-feature-name

# Create PR on GitHub
```

### 4. PR Template

```markdown
## Description
Brief description of changes

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Documentation update
- [ ] Code refactoring
- [ ] Performance improvement

## Changes Made
- Change 1
- Change 2
- Change 3

## Testing
- [ ] Unit tests added/updated
- [ ] All tests pass
- [ ] Manual testing completed

## Documentation
- [ ] Code comments added
- [ ] Docstrings updated
- [ ] README updated (if needed)
- [ ] Guides updated (if needed)

## Checklist
- [ ] Code follows style guide
- [ ] Tests pass locally
- [ ] Documentation updated
- [ ] No breaking changes (or documented)
```

### 5. Review Process

1. Maintainers will review your PR
2. Address feedback and comments
3. Update PR as needed
4. Once approved, it will be merged

### 6. After Merge

```bash
# Update your main branch
git checkout main
git pull upstream main

# Delete feature branch
git branch -d feature/your-feature-name
```

---

## Community Guidelines

### Code of Conduct

Be respectful and inclusive:
- ✅ Be welcoming and friendly
- ✅ Respect different viewpoints
- ✅ Accept constructive criticism
- ✅ Focus on what's best for the community
- ❌ No harassment or discrimination
- ❌ No trolling or insulting comments

### Getting Help

- **Documentation**: Check `docs/` first
- **Issues**: Search existing issues
- **Discussions**: Use GitHub Discussions for questions
- **Email**: Contact maintainers for private inquiries

### Recognition

Contributors will be:
- Listed in CONTRIBUTORS.md
- Mentioned in release notes
- Credited in presentations (with permission)

---

## Development Tips

### Local Testing

```bash
# Test specific component
pytest tests/test_tokenizer.py -v

# Test with print output
pytest -s

# Test with debugging
pytest --pdb
```

### Performance Testing

```python
import time

def benchmark_generation():
    """Benchmark generation speed."""
    start = time.time()

    for _ in range(100):
        generate_code(model, tokenizer, "#!/bin/bash\n# test")

    elapsed = time.time() - start
    print(f"Average: {elapsed / 100:.3f}s per generation")
```

### Debugging Tips

```python
# Use logging instead of print
import logging
logger = logging.getLogger(__name__)

logger.debug(f"Input shape: {input_tensor.shape}")
logger.info(f"Training epoch {epoch}")
logger.warning(f"Low confidence: {prob}")
logger.error(f"Generation failed: {e}")
```

---

## Questions?

- **Issues**: https://github.com/yourusername/code-llm-from-scratch/issues
- **Discussions**: https://github.com/yourusername/code-llm-from-scratch/discussions
- **Email**: your.email@example.com

Thank you for contributing! 🎉
