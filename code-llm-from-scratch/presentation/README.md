# Presentation Materials

Interactive materials for learning and presenting the code generation model.

## Contents

| File | Description |
|------|-------------|
| [interactive_demo.ipynb](interactive_demo.ipynb) | Interactive Jupyter notebook tutorial |
| [PRESENTATION_GUIDE.md](PRESENTATION_GUIDE.md) | Slide deck and presentation tips |

## Quick Start

```bash
# Navigate to presentation directory
cd presentation

# Start Jupyter
jupyter notebook interactive_demo.ipynb
```

## Interactive Demo Notebook

The notebook covers:

1. **Foundations** - Tokenization with BPE
2. **Architecture** - Transformer model structure
3. **Training** - Two-stage training process
4. **Generation** - Code generation from prompts
5. **Advanced** - Temperature effects, model scaling

### Visual Diagrams

The notebook references SVG diagrams from `../docs/diagrams/`:

- `tokenization-process.svg` - How tokenization works
- `transformer-architecture.svg` - Model architecture
- `attention-mechanism.svg` - Self-attention explained
- `two-stage-training.svg` - Training pipeline
- `training-loop.svg` - Training process
- `generation-process.svg` - Code generation

### Requirements

```python
# Required packages
torch
matplotlib
seaborn
numpy
pandas
```

## For Presenters

See [PRESENTATION_GUIDE.md](PRESENTATION_GUIDE.md) for:

- Complete slide deck outline
- Live demo scripts
- Presentation tips
- Backup slides
- Q&A preparation

## Related Documentation

- [Visual Guide](../docs/VISUAL_GUIDE.md) - All diagrams with explanations
- [Architecture](../docs/ARCHITECTURE.md) - Technical deep-dive
- [Getting Started](../GETTING_STARTED.md) - Setup guide
