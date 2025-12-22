# Stage 3: Code Fine-tuning

Fine-tune the language model on code data.

## Quick Start

```bash
# Prepare code data
python ../prepare_code_data.py

# Fine-tune on code
python ../train.py --stage code --model-size large

# With all optimizations
python ../train.py --stage code --model-size large \
    --use-rmsnorm --use-rope --use-compile --use-amp
```

## Prepare Data

```bash
# Prepare main code corpus
python ../prepare_code_data.py

# Add additional code data
python ../prepare_additional_code_data.py

# Generate domain-specific data
python ../generate_domain_data.py
```

## Configuration

| Flag | Description | Default |
|------|-------------|---------|
| `--model-size` | Must match Stage 2 | large |
| `--resume` | Resume from language model | True |
| `--learning-rate` | Lower than Stage 2 | 1e-6 |

## Output

Models saved to `models/`:
- `code_model_best.pth` - Best code model
- `code_model_latest.pth` - Latest checkpoint
- `code_training_history.json` - Training metrics

## Duration

30-60 minutes on M1 Max (depends on data size)

## Next Step

After code fine-tuning:
- For tool calling: [Stage 4: Tools](../4_tools/)
- For deployment: [Stage 6: Deploy](../6_deploy/)
