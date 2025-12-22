# Stage 5: RLHF (Reinforcement Learning from Human Feedback)

Align the model using human preferences.

## Quick Start

```bash
# Prepare RLHF data (preference pairs)
python ../prepare_rlhf_data.py

# Train reward model
python ../train_reward_model.py

# Run RLHF training
python ../train_rlhf.py
```

## Pipeline

1. **Prepare Data** - Create preference pairs (chosen/rejected)
2. **Reward Model** - Train model to predict human preferences
3. **PPO Training** - Fine-tune with reinforcement learning

## Available Scripts

| Script | Description |
|--------|-------------|
| `prepare_rlhf_data.py` | Create preference dataset |
| `train_reward_model.py` | Train reward model |
| `train_rlhf.py` | PPO training loop |
| `continual_learning.py` | Continual learning updates |

## Data Format

```json
{
  "prompt": "Write a backup script",
  "chosen": "#!/bin/bash\ntar -czf...",
  "rejected": "backup stuff..."
}
```

## Duration

- Reward model: 30-60 minutes
- PPO training: 2-4 hours

## Next Step

Proceed to [Stage 6: Deployment](../6_deploy/).
