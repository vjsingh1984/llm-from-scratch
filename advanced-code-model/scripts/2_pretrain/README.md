# Stage 2: Language Pretraining

Pretrain the model on natural language data (TinyStories).

## Quick Start

```bash
# Train language model (large, recommended)
python ../train.py --stage language --model-size large

# Train with all optimizations
python ../train.py --stage language --model-size large \
    --use-rmsnorm --use-rope --use-compile --use-amp

# Quick test with tiny model
python ../train.py --stage language --model-size tiny --num-epochs 1
```

## Configuration

| Flag | Description | Default |
|------|-------------|---------|
| `--model-size` | tiny/medium/large/xlarge | large |
| `--batch-size` | Batch size | 2 |
| `--num-epochs` | Number of epochs | 3 |
| `--learning-rate` | Learning rate | 5e-6 |
| `--use-rmsnorm` | Use RMSNorm | False |
| `--use-rope` | Use RoPE embeddings | False |
| `--use-compile` | Use torch.compile | False |
| `--use-amp` | Use mixed precision | False |

## Output

Models saved to `models/`:
- `language_model_best.pth` - Best checkpoint
- `language_model_latest.pth` - Latest checkpoint
- `language_training_history.json` - Training metrics

## Duration

- Tiny: ~10 minutes
- Large: 2-4 hours on M1 Max

## Next Step

After language pretraining, proceed to [Stage 3: Code Fine-tuning](../3_finetune/).
