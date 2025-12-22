# SVG Diagrams

Visual diagrams explaining key concepts of the code generation model.

## Available Diagrams

| Diagram | Description | Used In |
|---------|-------------|---------|
| [tokenization-process.svg](tokenization-process.svg) | How BPE tokenization converts text to tokens | Part 1: Foundations |
| [transformer-architecture.svg](transformer-architecture.svg) | Model architecture overview | Part 2: Architecture |
| [attention-mechanism.svg](attention-mechanism.svg) | Self-attention mechanism explained | Part 2: Architecture |
| [two-stage-training.svg](two-stage-training.svg) | Language + Code training pipeline | Part 3: Training |
| [training-loop.svg](training-loop.svg) | Training iteration process | Part 3: Training |
| [generation-process.svg](generation-process.svg) | Autoregressive code generation | Part 4: Generation |

## Usage

### In Markdown Files

```markdown
![Tokenization Process](diagrams/tokenization-process.svg)
```

### In Jupyter Notebooks

```markdown
![Tokenization Process](../docs/diagrams/tokenization-process.svg)
```

### In HTML

```html
<img src="diagrams/tokenization-process.svg" alt="Tokenization Process" />
```

## Viewing Diagrams

SVG files can be viewed in:
- Web browsers (drag and drop)
- GitHub (renders inline)
- VS Code (SVG Preview extension)
- Any image viewer that supports SVG

## Related

- [Visual Guide](../VISUAL_GUIDE.md) - Detailed explanations of each diagram
- [Interactive Demo](../../presentation/interactive_demo.ipynb) - Jupyter notebook with diagrams
