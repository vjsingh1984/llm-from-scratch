# Learning Path

> **Start Here!** Choose your path based on what you want to accomplish.

![Learning Path](diagrams/learning-path.svg)

---

## Quick Navigation

| I want to... | Go to... | Time |
|--------------|----------|------|
| Run a model and generate code NOW | [01-QUICKSTART.md](01-QUICKSTART.md) | 5 min |
| Understand how LLMs work | [02-CONCEPTS.md](02-CONCEPTS.md) | 30 min |
| Compare model architectures | [03-ARCHITECTURE.md](03-ARCHITECTURE.md) | 20 min |
| Train my own model from scratch | [04-TRAINING-PIPELINE.md](04-TRAINING-PIPELINE.md) | Full guide |
| Deploy to production | [05-DEPLOYMENT.md](05-DEPLOYMENT.md) | 45 min |

---

## Learning Paths

### Path A: Just Want to Use It (5 minutes)

```
01-QUICKSTART.md → Generate your first code
```

Perfect for: Evaluating the model, quick demos, testing outputs.

### Path B: Understanding the Concepts (1-2 hours)

```
02-CONCEPTS.md → 03-ARCHITECTURE.md → Try examples/
```

Perfect for: ML engineers, students, anyone curious about how LLMs work.

### Path C: Training Your Own Model (Full experience)

```
02-CONCEPTS.md → 04-TRAINING-PIPELINE.md → 05-DEPLOYMENT.md
```

Perfect for: Researchers, developers building custom models, learning by doing.

---

## Training Pipeline Overview

![Training Pipeline](diagrams/training-pipeline.svg)

The model is trained in stages, each building on the previous:

| Stage | Purpose | Data | Duration |
|-------|---------|------|----------|
| 0. Tokenizer | Learn vocabulary | Code + Text | 5 min |
| 1. Language | Basic language understanding | TinyStories | 2-4 hours |
| 2. Code | Code generation | Bash, Python, etc. | 3-6 hours |
| 3. Tools | Function calling | Synthetic examples | 1-2 hours |
| 4. RLHF | Human preferences | Preference pairs | 2-4 hours |
| 5. Deploy | Serve the model | - | 30 min |

---

## Architecture Options

![Architecture Comparison](diagrams/architecture-comparison.svg)

This project supports **4 model architectures**:

- **Dense Transformer** - The classic, well-understood architecture
- **Mamba (SSM)** - Linear complexity, great for long context
- **MoE (Sparse)** - More parameters at same compute cost
- **Hybrid** - Combines attention and Mamba layers

See [03-ARCHITECTURE.md](03-ARCHITECTURE.md) for detailed comparison.

---

## File Structure Quick Reference

```
advanced-code-model/
├── docs/               # You are here
│   ├── 00-LEARNING-PATH.md      # Start here
│   ├── 01-QUICKSTART.md         # 5-min getting started
│   ├── 02-CONCEPTS.md           # Core concepts
│   ├── 03-ARCHITECTURE.md       # Model architectures
│   ├── 04-TRAINING-PIPELINE.md  # Full training guide
│   └── 05-DEPLOYMENT.md         # Production deployment
├── scripts/            # All training and utility scripts
├── src/                # Model source code
├── data/               # Training data
├── models/             # Saved checkpoints
├── configs/            # YAML configuration files
└── examples/           # Working code examples
```

---

## Getting Help

- **Script reference**: [reference/SCRIPT-REFERENCE.md](reference/SCRIPT-REFERENCE.md)
- **Configuration options**: [reference/CONFIG-REFERENCE.md](reference/CONFIG-REFERENCE.md)
- **Common issues**: [reference/TROUBLESHOOTING.md](reference/TROUBLESHOOTING.md)

---

**Next step**: Pick a path above and start learning!
