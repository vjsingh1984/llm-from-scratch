# Advanced Topics Guide

This guide covers advanced topics for users who have completed the basic tutorial and want to dive deeper into customization, deployment, testing, and production use.

## Navigation

- [Prerequisites](#prerequisites)
- [Custom Fine-Tuning](#1-custom-fine-tuning)
- [API Deployment](#2-api-deployment)
- [Testing and Quality Assurance](#3-testing-and-quality-assurance)
- [Performance Optimization](#4-performance-optimization)
- [Interactive Development](#5-interactive-development)
- [Production Best Practices](#6-production-best-practices)
- [Research and Experimentation](#7-research-and-experimentation)

---

## Prerequisites

Before diving into advanced topics, you should:

1. ✅ Complete `GETTING_STARTED.md` (Levels 1-7)
2. ✅ Successfully train both language and code models
3. ✅ Generate code using `scripts/generate.py`
4. ✅ Understand the basic architecture from `docs/ARCHITECTURE.md`

If you haven't completed these, start there first!

---

## 1. Custom Fine-Tuning

**Goal**: Train the model on your own code or scripts.

### When to Use Custom Fine-Tuning

- You have domain-specific scripts (e.g., deployment scripts for your company)
- You want the model to learn your coding style
- You need to generate code for a specific framework or tool
- You have 50+ example scripts in your domain

### Step-by-Step Guide

#### 1.1 Prepare Your Data

Organize your scripts in a directory:

```bash
my_scripts/
├── deploy_001.sh
├── deploy_002.sh
├── monitor_001.sh
└── ...
```

Or create a JSON file:

```json
{
  "scripts": [
    "#!/bin/bash\n# My script 1\n...",
    "#!/bin/bash\n# My script 2\n...",
    ...
  ]
}
```

#### 1.2 Run Fine-Tuning

```bash
python examples/fine_tuning.py \
    --data-path my_scripts/ \
    --pretrained-model models/language/language_model_final.pt \
    --output-dir models/custom/ \
    --num-epochs 30 \
    --batch-size 8
```

**Key Parameters**:
- `--data-path`: Your scripts directory or JSON file
- `--num-epochs`: More epochs for better learning (30-50 recommended)
- `--learning-rate`: Lower for fine-tuning (default: 1e-4)
- `--validation-split`: Fraction for validation (default: 0.1)

#### 1.3 Monitor Training

The script shows:
- Data validation results (warnings for missing shebangs, etc.)
- Training progress with loss curves
- Validation metrics
- Best model checkpoints

Expected output:
```
Epoch 1/30: Train Loss: 2.1, Val Loss: 2.0
...
Epoch 30/30: Train Loss: 0.8, Val Loss: 1.1
✓ Best model saved
```

#### 1.4 Test Your Custom Model

```bash
python scripts/generate.py \
    --model models/custom/custom_model_final.pt \
    --prompt "#!/bin/bash\n# Your custom task" \
    --max-length 300
```

### Tips for Best Results

1. **Data Quality**:
   - Use well-formatted, working scripts
   - Include comments explaining what code does
   - Ensure consistent style across scripts

2. **Data Quantity**:
   - Minimum: 50 scripts
   - Good: 100-200 scripts
   - Excellent: 500+ scripts

3. **Training Duration**:
   - Short (10-20 epochs): Quick adaptation
   - Medium (30-40 epochs): Good quality
   - Long (50+ epochs): Best quality (risk of overfitting)

4. **Avoid Overfitting**:
   - Use validation split (10-20%)
   - Stop if validation loss increases
   - Use more diverse training data

### Complete Example Workflow

See `examples/fine_tuning.py` for the full implementation with:
- Data validation
- Custom dataset creation
- Training loop with checkpoints
- Evaluation metrics

---

## 2. API Deployment

**Goal**: Deploy your model as a production REST API.

### Local Development

#### 2.1 Install Dependencies

```bash
pip install fastapi uvicorn pydantic
```

#### 2.2 Run API Server

```bash
python examples/deployment_api.py
```

Access at:
- API: http://localhost:8000
- Docs: http://localhost:8000/docs
- Health: http://localhost:8000/health

#### 2.3 Test the API

Python client:
```python
import requests

response = requests.post(
    "http://localhost:8000/generate",
    json={
        "prompt": "#!/bin/bash\n# Create a backup script",
        "max_length": 200,
        "temperature": 0.8,
        "top_k": 50
    }
)

result = response.json()
print(result['generated_code'])
```

curl:
```bash
curl -X POST http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "#!/bin/bash\n# Monitor disk space",
    "max_length": 200,
    "temperature": 0.8
  }'
```

### Docker Deployment

#### 2.4 Build Docker Image

```bash
docker build -t code-llm-api .
```

#### 2.5 Run Container

```bash
docker run -d \
  --name code-llm-api \
  -p 8000:8000 \
  code-llm-api
```

#### 2.6 Or Use Docker Compose

```bash
docker-compose up -d
docker-compose logs -f
```

### Cloud Deployment

For production deployment on AWS, GCP, or Azure, see the complete guide:

**📖 [docs/DEPLOYMENT.md](DEPLOYMENT.md)** - Comprehensive deployment guide covering:
- AWS EC2 deployment
- Google Cloud Run
- Azure Container Instances
- Kubernetes deployment
- Security best practices
- Load balancing and scaling
- Monitoring and logging

### API Features

The deployment API includes:
- ✅ Request validation with Pydantic
- ✅ Error handling
- ✅ Health checks
- ✅ Model information endpoint
- ✅ Configurable via environment variables
- ✅ CORS support
- ✅ Production-ready

---

## 3. Testing and Quality Assurance

**Goal**: Ensure code quality and prevent regressions.

### Running Tests

#### 3.1 Install Test Dependencies

```bash
pip install pytest pytest-cov pytest-mock
```

#### 3.2 Run All Tests

```bash
# All tests
pytest

# With coverage
pytest --cov=src --cov-report=html

# Specific test file
pytest tests/test_tokenizer.py

# Specific test
pytest tests/test_tokenizer.py::test_bpe_encode_decode
```

#### 3.3 Test Categories

**Unit Tests** (fast, ~1-2 seconds each):
```bash
# Tokenizer tests
pytest tests/test_tokenizer.py

# Model tests
pytest tests/test_model.py

# Generation tests
pytest tests/test_generation.py
```

**Integration Tests** (slower, ~10-30 seconds):
```bash
pytest tests/integration/test_end_to_end.py
```

**Exclude Slow Tests**:
```bash
pytest -m "not slow"
```

### Test Coverage

Generate HTML coverage report:

```bash
pytest --cov=src --cov-report=html
open htmlcov/index.html
```

Target coverage: **>80%** for production code.

### Writing Your Own Tests

Example test structure:

```python
import pytest
from src.tokenizer.bpe import BPETokenizer

def test_custom_functionality():
    """Test description."""
    # Setup
    tokenizer = BPETokenizer()
    tokenizer.target_vocab_size = 100
    tokenizer.train(["sample", "data"], verbose=False)

    # Execute
    result = tokenizer.encode("sample")

    # Assert
    assert len(result) > 0, "Should produce tokens"
```

### Continuous Integration

GitHub Actions workflow (`.github/workflows/tests.yml`):

```yaml
name: Tests
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - uses: actions/setup-python@v2
        with:
          python-version: '3.10'
      - run: pip install -r requirements.txt
      - run: pip install pytest pytest-cov
      - run: pytest --cov=src
```

For complete testing documentation, see:

**📖 [tests/README.md](../tests/README.md)** - Complete testing guide

---

## 4. Performance Optimization

### Model Optimization

#### 4.1 Model Quantization

Reduce model size and improve inference speed:

```python
import torch

# Load model
checkpoint = torch.load('models/code/code_model_final.pt')
model = CodeTransformer(checkpoint['config'])
model.load_state_dict(checkpoint['model_state_dict'])

# Quantize to int8
model_quantized = torch.quantization.quantize_dynamic(
    model,
    {torch.nn.Linear},
    dtype=torch.qint8
)

# Save quantized model (50% smaller, 2x faster)
torch.save({
    'model_state_dict': model_quantized.state_dict(),
    'config': checkpoint['config']
}, 'models/code/code_model_quantized.pt')
```

Benefits:
- 50% smaller model size
- 2x faster inference
- Minimal quality loss (< 1%)

#### 4.2 Batch Generation

Generate multiple scripts in parallel:

```python
def generate_batch(prompts, model, tokenizer, max_length=200):
    """Generate from multiple prompts efficiently."""
    # Tokenize all prompts
    input_ids_list = [tokenizer.encode(p) for p in prompts]

    # Pad to same length
    max_len = max(len(ids) for ids in input_ids_list)
    padded = [ids + [0] * (max_len - len(ids)) for ids in input_ids_list]

    # Generate in batch
    batch = torch.tensor(padded, device=device)
    generated = generate_tokens(batch, model, max_length)

    # Decode all results
    return [tokenizer.decode(seq) for seq in generated]
```

#### 4.3 Caching Common Prompts

```python
from functools import lru_cache

@lru_cache(maxsize=100)
def cached_generate(prompt: str, max_length: int = 200):
    return generate_code(model, tokenizer, prompt, max_length)
```

### Server Optimization

#### 4.4 Multiple Workers

```bash
# Run with 4 worker processes
uvicorn examples.deployment_api:app \
    --host 0.0.0.0 \
    --port 8000 \
    --workers 4
```

#### 4.5 Response Compression

Add to `deployment_api.py`:

```python
from fastapi.middleware.gzip import GZipMiddleware

app.add_middleware(GZipMiddleware, minimum_size=1000)
```

### Benchmarking

Measure generation speed:

```python
import time

start = time.time()
for _ in range(100):
    generate_code(model, tokenizer, "#!/bin/bash\n# test")
elapsed = time.time() - start

print(f"Average: {elapsed / 100:.3f}s per generation")
print(f"Throughput: {100 / elapsed:.1f} generations/second")
```

---

## 5. Interactive Development

### Jupyter Notebook Tutorial

For hands-on learning and experimentation:

**📖 [presentation/interactive_demo.ipynb](../presentation/interactive_demo.ipynb)**

This notebook covers:

**Part 1: Foundations**
- Understanding tokenization
- BPE vs character vs word tokenization
- Training a tokenizer
- Vocabulary analysis

**Part 2: Architecture**
- Model configuration
- Parameter breakdown
- Attention mechanism visualization
- Layer-by-layer explanation

**Part 3: Training**
- Mini training run
- Loss curve visualization
- Training loop explanation

**Part 4: Generation**
- Loading trained models
- Generation function implementation
- Temperature effects
- Sampling strategies

**Part 5: Advanced**
- Token probability analysis
- Model size comparison
- Training data distribution
- Performance optimization

### Running the Notebook

```bash
# Install Jupyter
pip install jupyter matplotlib

# Launch
jupyter notebook presentation/interactive_demo.ipynb
```

### Experimentation Tips

1. **Start Small**: Use tiny model (2 layers, 64 dims) for quick experiments
2. **Iterate Quickly**: Test on small data first, then scale up
3. **Visualize**: Plot loss curves, attention patterns, token distributions
4. **Compare**: Test different hyperparameters side-by-side
5. **Document**: Save your findings in notebooks

---

## 6. Production Best Practices

### Security

#### 6.1 Rate Limiting

```python
from slowapi import Limiter
from slowapi.util import get_remote_address

limiter = Limiter(key_func=get_remote_address)

@app.post("/generate")
@limiter.limit("10/minute")
async def generate_code(request: Request, ...):
    # Your code here
```

#### 6.2 Input Validation

Already implemented via Pydantic, but add domain-specific checks:

```python
BLOCKED_PATTERNS = ['rm -rf /', 'dd if=/dev/zero', 'fork bomb']

def validate_prompt(prompt: str):
    for pattern in BLOCKED_PATTERNS:
        if pattern in prompt.lower():
            raise ValueError(f"Blocked pattern: {pattern}")
```

#### 6.3 API Authentication

```python
from fastapi.security import APIKeyHeader

api_key_header = APIKeyHeader(name="X-API-Key")

async def verify_api_key(api_key: str = Depends(api_key_header)):
    if api_key != os.getenv("API_KEY"):
        raise HTTPException(status_code=403)
```

### Monitoring

#### 6.4 Logging

```python
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@app.post("/generate")
async def generate_code(...):
    logger.info(f"Generation request: {request.prompt[:50]}")
    # ... generate code
    logger.info(f"Generated {num_tokens} tokens in {time:.2f}s")
```

#### 6.5 Metrics

```python
from prometheus_fastapi_instrumentator import Instrumentator

Instrumentator().instrument(app).expose(app)
```

Access metrics at `/metrics` for Prometheus/Grafana.

### Error Handling

```python
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Error: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"error": "Internal server error"}
    )
```

### Production Checklist

Before deploying:

- [ ] Enable HTTPS/TLS
- [ ] Add API authentication
- [ ] Configure rate limiting
- [ ] Set up monitoring and logging
- [ ] Test with production-like load
- [ ] Configure auto-scaling
- [ ] Set up automated backups
- [ ] Document API endpoints
- [ ] Perform security audit
- [ ] Set up CI/CD pipeline

---

## 7. Research and Experimentation

### Model Variants

#### 7.1 Experiment with Different Architectures

Modify `src/model/config.py`:

```python
# Deeper but narrower
config = CoderConfig(
    n_layers=12,    # More layers
    d_model=256,    # Smaller width
    n_heads=8,
    d_ff=1024
)

# Wider but shallower
config = CoderConfig(
    n_layers=4,     # Fewer layers
    d_model=768,    # Larger width
    n_heads=12,
    d_ff=3072
)
```

#### 7.2 Add New Features

**Relative Positional Encoding**:
Research shows relative positions may work better for code.

**Mixture of Experts (MoE)**:
Scale model by adding expert layers.

**Custom Attention Patterns**:
Try local attention, sparse attention, etc.

### Training Strategies

#### 7.3 Learning Rate Schedules

```python
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

scheduler = CosineAnnealingWarmRestarts(
    optimizer,
    T_0=1000,  # Restart every 1000 steps
    T_mult=2,
    eta_min=1e-6
)
```

#### 7.4 Data Augmentation

For code:
- Variable renaming
- Comment insertion/removal
- Code style variations
- Synthetic data generation

### Evaluation Metrics

#### 7.5 Automatic Evaluation

```python
def evaluate_syntax(generated_code):
    """Check if generated code is syntactically valid."""
    import subprocess
    result = subprocess.run(
        ['bash', '-n', '-c', generated_code],
        capture_output=True
    )
    return result.returncode == 0

# Test on validation set
valid_scripts = 0
for prompt in test_prompts:
    generated = generate_code(model, tokenizer, prompt)
    if evaluate_syntax(generated):
        valid_scripts += 1

print(f"Syntax correctness: {valid_scripts / len(test_prompts) * 100:.1f}%")
```

---

## Summary

This guide covered:

1. ✅ **Custom Fine-Tuning**: Train on your own data
2. ✅ **API Deployment**: Deploy to production
3. ✅ **Testing**: Ensure code quality
4. ✅ **Performance**: Optimize for speed and efficiency
5. ✅ **Interactive Development**: Experiment with Jupyter
6. ✅ **Production**: Best practices for deployment
7. ✅ **Research**: Explore new ideas

## Next Steps

1. **Fine-tune** on your own scripts using `examples/fine_tuning.py`
2. **Deploy** your API using Docker or cloud services
3. **Test** your changes with the comprehensive test suite
4. **Experiment** with the interactive notebook
5. **Contribute** your improvements back to the project!

## Additional Resources

- **Architecture Deep-Dive**: [docs/ARCHITECTURE.md](ARCHITECTURE.md)
- **Deployment Guide**: [docs/DEPLOYMENT.md](DEPLOYMENT.md)
- **Testing Guide**: [tests/README.md](../tests/README.md)
- **Presentation Materials**: [presentation/PRESENTATION_GUIDE.md](../presentation/PRESENTATION_GUIDE.md)
- **Interactive Tutorial**: [presentation/interactive_demo.ipynb](../presentation/interactive_demo.ipynb)

---

**Ready to build something amazing!** These advanced topics will help you customize, deploy, and optimize the Code LLM for your specific needs.

For questions or contributions, open an issue on GitHub or submit a pull request.
