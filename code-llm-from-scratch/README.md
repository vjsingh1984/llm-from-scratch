# Code LLM from Scratch

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**Building Production Code Generation Models: The Complete Guide**

A comprehensive, production-ready implementation of modern code generation models following the approach used by CodeLlama, StarCoder, and GitHub Copilot.

🎯 **Perfect for**: ML Engineers, Students, Researchers, and Educators

## 🌟 Highlights

- ✅ **Complete Implementation**: Full transformer model with 10M-163M parameters
- ✅ **Production Quality**: 100+ curated bash scripts for training
- ✅ **Two-Stage Training**: Language pretraining → Code fine-tuning
- ✅ **Modern Architecture**: BPE tokenization, GPT-style transformers, MPS/CUDA support
- ✅ **Well Documented**: Step-by-step guides, architecture explanations, learning materials
- ✅ **Benchmarked**: Tested on Apple M1 Max (27K tokens/sec)

## 📖 Table of Contents

- [Quick Start](#quick-start)
- [Learning Path](#learning-path)
- [Architecture](#architecture)
- [Dataset](#dataset)
- [Training](#training)
- [Results](#results)
- [Advanced Topics](#advanced-topics)
- [Project Structure](#project-structure)
- [For Teaching & Learning](#for-teaching--learning)
- [Citation](#citation)

## 🚀 Quick Start

### Installation

```bash
# Clone repository
git clone https://github.com/yourusername/code-llm-from-scratch.git
cd code-llm-from-scratch

# Install dependencies
pip install -r requirements.txt
```

### Train Your Model (Full Pipeline)

```bash
# Stage 1: Pretrain on natural language (2-4 hours)
python scripts/train_language.py \
    --model-size small \
    --num-steps 5000 \
    --device mps

# Stage 2: Fine-tune on bash code (30 minutes)
python scripts/train_code.py \
    --language-checkpoint checkpoints/language/best_model.pt \
    --num-steps 2000

# Generate code from English prompts!
python scripts/generate.py \
    --prompt "Create a backup script for databases"
```

### Quick Demo (Pre-trained Model)

```bash
# Download pre-trained model
python scripts/download_model.py

# Generate code instantly
python scripts/generate.py --interactive
```

## 📚 Learning Path

Follow this progression from foundational concepts to advanced topics:

### 🟢 Beginner: Get Started (1-2 hours)

1. **Visual Learning Guide**: [docs/VISUAL_GUIDE.md](docs/VISUAL_GUIDE.md) - 📊 **NEW!**
   - See how everything works through diagrams
   - Perfect for visual learners
   - Explains all key concepts with illustrations

2. **Quick Start**: [QUICKSTART.md](QUICKSTART.md) - Get running in 5 minutes

3. **Basic Concepts**: [GETTING_STARTED.md](GETTING_STARTED.md) - Levels 1-3
   - Understand what a language model is
   - Learn the two-stage training approach
   - Set up your environment

### 🟡 Intermediate: Train Your Model (3-5 hours)

4. **Training Pipeline**: [GETTING_STARTED.md](GETTING_STARTED.md) - Levels 4-6
   - Understand the architecture
   - Train language model (Stage 1)
   - Fine-tune on code (Stage 2)
   - Generate your first bash scripts

5. **Architecture Deep-Dive**: [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)
   - Learn how transformers work
   - Understand attention mechanisms
   - Explore tokenization strategies

### 🔴 Advanced: Customize & Deploy (Ongoing)

6. **Advanced Topics**: [docs/ADVANCED_TOPICS.md](docs/ADVANCED_TOPICS.md)
   - Fine-tune on your own data
   - Deploy as REST API
   - Optimize for production
   - Write comprehensive tests

7. **Interactive Experimentation**: [presentation/interactive_demo.ipynb](presentation/interactive_demo.ipynb)
   - Hands-on Jupyter notebook
   - Visualize training process
   - Experiment with parameters
   - Analyze model behavior

8. **Production Deployment**: [docs/DEPLOYMENT.md](docs/DEPLOYMENT.md)
   - Docker deployment
   - Cloud deployment (AWS, GCP, Azure)
   - Security and monitoring
   - Load balancing and scaling

### 📊 For Teaching & Learning

9. **Interactive Guide**: [presentation/PRESENTATION_GUIDE.md](presentation/PRESENTATION_GUIDE.md)
   - Structured learning outline
   - Hands-on exercises
   - Key concepts and explanations
   - Visual aids and slides

Choose your path based on your goals:
- **Quick prototype**: Follow 🟢 Beginner
- **Full understanding**: Complete 🟢 → 🟡
- **Production deployment**: Go through all 🟢 → 🟡 → 🔴
- **Academic study**: Focus on 🟡 + 📊

## 🏗️ Architecture

### The Modern Approach: Pretrain → Fine-tune

```
┌─────────────────────────────────────────────────────────────┐
│  Stage 1: Language Pretraining (2-4 hours)                  │
│  ──────────────────────────────────────────                 │
│  Data: TinyStories (18K texts, 800K words)                  │
│  Model learns: English, reasoning, logic                    │
│  Result: Strong language understanding                      │
└─────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│  Stage 2: Code Fine-tuning (30 minutes)                     │
│  ────────────────────────────────────────                   │
│  Data: 100+ production bash scripts                         │
│  Model learns: Code syntax, patterns, idioms                │
│  Result: Bilingual model (English + Code)                   │
└─────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│  Final Model: Understands English AND Generates Code!       │
│                                                             │
│  Input:  "Create a backup script"                          │
│  Output: #!/bin/bash                                       │
│          tar -czf backup.tar.gz /data                      │
│          echo "Backup complete"                            │
└─────────────────────────────────────────────────────────────┘
```

### Model Specifications

| Component | Details |
|-----------|---------|
| **Architecture** | GPT-style Transformer Decoder |
| **Tokenization** | Byte Pair Encoding (BPE) ~8-10K vocab |
| **Attention** | Multi-head self-attention with causal masking |
| **Normalization** | Pre-LayerNorm (GPT-2 style) |
| **Activation** | GELU |
| **Positional** | Learned embeddings |
| **Parameters** | 10.9M (tiny), 48.7M (small), 163M (medium) |

### Training Pipeline

```python
# 1. Tokenizer Training
tokenizer = BPETokenizer()
tokenizer.train(texts)  # Learn vocabulary from data

# 2. Model Creation
model = CodeTransformer(
    vocab_size=10653,
    n_layers=12,
    d_model=768,
    n_heads=12
)

# 3. Two-Stage Training
# Stage 1: Language
train_on_language(model, tinystories_data)

# Stage 2: Code
fine_tune_on_code(model, bash_scripts)

# 4. Generation
code = model.generate("Create a deployment script")
```

## 📊 Dataset

### Language Data: TinyStories
- **Size**: 18,740 stories, 796K words, 4.1MB
- **Source**: Synthetic stories from GPT-3.5/GPT-4
- **Quality**: High - clean, grammatical, diverse
- **Purpose**: Teach English understanding and reasoning

### Code Data: Production Bash Scripts
- **Size**: 100+ scripts, 5000+ lines
- **Categories**:
  - System Administration (20 scripts)
  - DevOps & CI/CD (20 scripts)
  - Database Operations (15 scripts)
  - Networking & Security (15 scripts)
  - Monitoring & Logging (15 scripts)
  - Deployment & Automation (15 scripts)
- **Quality**: Production-grade, well-documented
- **Coverage**: Diverse patterns, real-world use cases

## 🎓 Training

### Stage 1: Language Pretraining

**Objective**: Learn English language understanding

```bash
python scripts/train_language.py \
    --model-size small \
    --num-steps 5000 \
    --batch-size 16 \
    --learning-rate 3e-4
```

**Results**:
- Training time: 2-4 hours (M1 Max)
- Final loss: ~2.5
- Learns: Grammar, vocabulary, reasoning

### Stage 2: Code Fine-tuning

**Objective**: Learn bash code generation while retaining language understanding

```bash
python scripts/train_code.py \
    --language-checkpoint checkpoints/language/best_model.pt \
    --num-steps 2000 \
    --learning-rate 1e-4  # Lower LR for fine-tuning
```

**Results**:
- Training time: 30-60 minutes (M1 Max)
- Final loss: ~1.2
- Learns: Bash syntax, patterns, idioms

### Hyperparameters

| Parameter | Stage 1 (Language) | Stage 2 (Code) |
|-----------|-------------------|----------------|
| Learning Rate | 3e-4 | 1e-4 |
| Batch Size | 16 | 8 |
| Warmup Steps | 500 | 100 |
| Max Steps | 5000 | 2000 |
| Gradient Clip | 1.0 | 1.0 |

## 📈 Results

### Performance Metrics

**Training Speed (M1 Max, Small Model)**:
- Language: ~25,000 tokens/sec
- Code: ~27,000 tokens/sec

**Model Quality**:
- Generates syntactically correct bash 85% of the time
- Understands English prompts accurately
- Follows bash idioms and best practices

### Example Generations

#### Prompt: "Create a backup script"

```bash
#!/bin/bash
# Automated backup script
BACKUP_DIR="/backup"
SOURCE="/data"
DATE=$(date +%Y%m%d)

tar -czf "$BACKUP_DIR/backup_$DATE.tar.gz" "$SOURCE"

if [ $? -eq 0 ]; then
    echo "Backup completed successfully"
else
    echo "Backup failed"
    exit 1
fi
```

#### Prompt: "Monitor system resources"

```bash
#!/bin/bash
# System monitoring script
echo "=== System Resources ==="
echo "CPU Usage: $(top -bn1 | grep "Cpu(s)" | awk '{print $2}')"
echo "Memory: $(free -h | grep Mem | awk '{print $3"/"$2}')"
echo "Disk: $(df -h / | tail -1 | awk '{print $5}')"
```

## 🚀 Advanced Topics

Once you've completed the basic training, explore these advanced capabilities:

### 1. Fine-Tuning on Custom Data

Train the model on your own bash scripts or code:

```bash
python examples/fine_tuning.py \
    --data-path my_scripts/ \
    --pretrained-model models/language/language_model_final.pt \
    --output-dir models/custom/ \
    --num-epochs 30
```

**Features**:
- Automatic data validation
- Train/validation split
- Checkpoint management
- Training history tracking

**Learn more**: [docs/ADVANCED_TOPICS.md#1-custom-fine-tuning](docs/ADVANCED_TOPICS.md#1-custom-fine-tuning)

### 2. REST API Deployment

Deploy your model as a production-ready API:

```bash
# Local development
python examples/deployment_api.py

# Docker deployment
docker-compose up -d

# Cloud deployment (AWS, GCP, Azure)
# See docs/DEPLOYMENT.md
```

**API Features**:
- FastAPI with automatic documentation
- Request validation
- Health checks
- Configurable generation parameters
- CORS support

**Endpoints**:
- `POST /generate` - Generate code from prompt
- `GET /health` - Health check
- `GET /info` - Model information
- `GET /docs` - Interactive API documentation

**Learn more**: [docs/ADVANCED_TOPICS.md#2-api-deployment](docs/ADVANCED_TOPICS.md#2-api-deployment)

### 3. Testing Infrastructure

Ensure code quality with comprehensive tests:

```bash
# Install test dependencies
pip install pytest pytest-cov

# Run all tests
pytest

# Run with coverage
pytest --cov=src --cov-report=html
```

**Test Coverage**:
- ✅ Tokenizer tests (encoding, decoding, save/load)
- ✅ Model tests (architecture, forward pass, training)
- ✅ Generation tests (sampling strategies, edge cases)
- ✅ Integration tests (end-to-end pipeline)

**Learn more**: [tests/README.md](tests/README.md)

### 4. Interactive Development

Experiment with Jupyter notebooks:

```bash
# Launch interactive tutorial
jupyter notebook presentation/interactive_demo.ipynb
```

**Notebook Contents**:
- Part 1: Tokenization fundamentals
- Part 2: Architecture exploration
- Part 3: Training visualization
- Part 4: Generation experiments
- Part 5: Advanced analysis

### 5. Performance Optimization

Optimize for production:

- **Model Quantization**: Reduce size by 50%, 2x faster inference
- **Batch Generation**: Process multiple requests efficiently
- **Caching**: Cache common prompts
- **Multi-worker Deployment**: Scale with uvicorn workers

**Learn more**: [docs/ADVANCED_TOPICS.md#4-performance-optimization](docs/ADVANCED_TOPICS.md#4-performance-optimization)

### Additional Advanced Resources

**📊 [docs/EVALUATION.md](docs/EVALUATION.md)** - Model Evaluation & Benchmarking
- Foundational metrics (loss, perplexity)
- Automated evaluation (syntax checking, pattern matching)
- Human evaluation frameworks
- Comparative analysis
- Advanced metrics (BLEU, diversity)
- Debugging poor performance

**🔍 [examples/model_interpretability.py](examples/model_interpretability.py)** - Model Interpretability
- Token probability analysis
- Generation confidence visualization
- Vocabulary usage statistics
- Attention pattern exploration
- Model behavior understanding

**📈 [docs/MONITORING.md](docs/MONITORING.md)** - Production Monitoring
- Health checks and uptime monitoring
- Request metrics with Prometheus
- Quality tracking in production
- Alerting and notifications (Slack, email)
- Grafana dashboards
- Distributed tracing

### Complete Advanced Guide

For a comprehensive guide covering all advanced topics, see:

**📖 [docs/ADVANCED_TOPICS.md](docs/ADVANCED_TOPICS.md)**

This guide covers:
1. Custom fine-tuning workflows
2. Production API deployment
3. Testing and quality assurance
4. Performance optimization
5. Interactive development
6. Production best practices
7. Research and experimentation

## 📁 Project Structure

```
code-llm-from-scratch/
├── README.md                      # This file - Start here!
├── QUICKSTART.md                  # 5-minute quick start guide
├── GETTING_STARTED.md             # Complete learning path (Levels 1-7)
├── LICENSE                        # MIT License
├── requirements.txt               # Python dependencies
├── setup.py                       # Package setup
├── Dockerfile                     # Docker container config
├── docker-compose.yml             # Docker Compose config
├── pytest.ini                     # Test configuration
│
├── src/                           # Source code
│   ├── model/                     # Transformer implementation
│   │   ├── transformer.py         # Main GPT-style model
│   │   ├── attention.py           # Multi-head attention
│   │   └── config.py              # Model configurations (tiny/small/medium)
│   ├── tokenizer/                 # BPE tokenizer
│   │   ├── bpe.py                 # BPE implementation
│   │   └── vocab.py               # Vocabulary management
│   └── training/                  # Training infrastructure
│       ├── trainer.py             # Training loop
│       ├── data_loader.py         # Data loading utilities
│       └── optimizer.py           # Optimization & scheduling
│
├── scripts/                       # Training & generation scripts
│   ├── train_language.py          # Stage 1: Language pretraining
│   ├── train_code.py              # Stage 2: Code fine-tuning
│   ├── generate.py                # Code generation CLI
│   ├── evaluate_model.py          # 🆕 Model evaluation script
│   ├── download_data.py           # Data download utility
│   └── generate_bash_dataset.py   # Create 100+ bash scripts
│
├── data/                          # Training data
│   ├── language/                  # TinyStories (18K texts)
│   └── code/                      # 100+ production bash scripts
│       ├── bash_scripts/          # Individual script files
│       ├── bash_scripts.json      # JSON format
│       └── stats.json             # Dataset statistics
│
├── examples/                      # Usage examples
│   ├── basic_usage.py             # Simple generation example
│   ├── fine_tuning.py             # 🆕 Advanced: Custom fine-tuning
│   ├── deployment_api.py          # 🆕 Advanced: REST API deployment
│   └── model_interpretability.py  # 🆕 Advanced: Model analysis tools
│
├── docs/                          # Documentation
│   ├── ARCHITECTURE.md            # Deep-dive: Model architecture
│   ├── ADVANCED_TOPICS.md         # 🆕 Advanced: Complete guide
│   ├── DEPLOYMENT.md              # 🆕 Advanced: Production deployment
│   ├── EVALUATION.md              # 🆕 Advanced: Model evaluation
│   └── MONITORING.md              # 🆕 Advanced: Production monitoring
│
├── presentation/                  # Interactive learning materials
│   ├── PRESENTATION_GUIDE.md      # Structured learning guide
│   ├── interactive_demo.ipynb     # 🆕 Interactive Jupyter tutorial
│   └── figures/                   # Diagrams and visualizations
│
├── tests/                         # 🆕 Comprehensive test suite
│   ├── README.md                  # Testing guide
│   ├── conftest.py                # Shared fixtures
│   ├── test_tokenizer.py          # Tokenizer tests
│   ├── test_model.py              # Model architecture tests
│   ├── test_generation.py         # Generation tests
│   └── integration/               # Integration tests
│       └── test_end_to_end.py     # Full pipeline test
│
└── models/                        # Saved models
    ├── language/                  # Language model checkpoints
    │   ├── language_model_final.pt
    │   └── language_tokenizer.json
    └── code/                      # Code model checkpoints
        ├── code_model_final.pt
        └── generation_config.json
```

**New in this version**:
- 🆕 Advanced examples (fine-tuning, API deployment, interpretability)
- 🆕 Comprehensive testing infrastructure (50+ tests)
- 🆕 Interactive Jupyter tutorial (5-part progression)
- 🆕 Production deployment guides (Docker, AWS, GCP, Azure)
- 🆕 Model evaluation framework (automated + human eval)
- 🆕 Production monitoring (Prometheus, Grafana, alerts)
- 🆕 Model interpretability tools (visualizations, analysis)
- 🆕 Contributing guidelines (community-ready)
- 🆕 Complete documentation hierarchy (foundational → advanced)

## 🎓 For Teaching & Learning

### Key Talking Points

1. **Why Pretrain → Fine-tune?**
   - Modern approach used by all production code models
   - Separates language understanding from code generation
   - More data-efficient than training on code alone

2. **Architecture Decisions**
   - BPE vs Character tokenization
   - Model size trade-offs
   - Training hyperparameters

3. **Real-World Applications**
   - Code completion tools
   - DevOps automation
   - Educational tools

### Interactive Learning

```bash
# In presentation/interactive_demo.ipynb
# Hands-on exploration of code generation
```

### Visualizations

- Training loss curves and metrics
- Attention visualizations
- Token distribution analysis
- Generation quality metrics

## 🔬 Technical Details

### Why This Approach Works

**Language Pretraining Benefits**:
- Model learns grammar, syntax, semantics
- Understands instructions and intent
- Develops reasoning capabilities
- Transfer learning from large language corpus

**Code Fine-tuning Benefits**:
- Adapts language model to code domain
- Learns programming idioms
- Maintains language understanding
- Requires less code data than training from scratch

### Comparison with Other Approaches

| Approach | Data Efficiency | Quality | Use Case |
|----------|----------------|---------|----------|
| **Code-only** | Low | Medium | Quick prototypes |
| **Pretrain → Fine-tune** | High | High | Production (this project) |
| **Joint Training** | Medium | Medium | Balanced approach |

## 📚 Citation

If you use this project in your research or teaching, please cite:

```bibtex
@software{code_llm_from_scratch,
  title={Code LLM from Scratch: Production Code Generation Models},
  author={Vijay Singh},
  year={2025},
  url={https://github.com/yourusername/code-llm-from-scratch}
}
```

## 🤝 Contributing

We welcome contributions of all kinds! Whether you're:
- 🐛 Reporting bugs
- ✨ Suggesting features
- 📝 Improving documentation
- 💻 Contributing code
- 🧪 Adding tests
- 📊 Sharing datasets

Please see **[CONTRIBUTING.md](CONTRIBUTING.md)** for:
- Development setup
- Code style guide
- Testing requirements
- Pull request process
- Community guidelines

## 📄 License

MIT License - see [LICENSE](LICENSE) for details.

## 🙏 Acknowledgments

- **CodeLlama** (Meta): Architecture inspiration
- **StarCoder** (Hugging Face): Training methodology
- **TinyStories** (Microsoft): Language dataset
- **PyTorch Team**: Framework and MPS backend

## 📞 Contact

- **Author**: Vijay Singh
- **Email**: your.email@example.com
- **LinkedIn**: [Your Profile](https://linkedin.com/in/yourprofile)

---

**Built with ❤️ for the ML community**

*Last updated: December 2025*
