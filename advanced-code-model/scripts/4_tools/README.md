# Stage 4: Tool Calling

Train the model to use tools and function calling.

## Quick Start

```bash
# Prepare tool calling data
python ../prepare_tool_calling_data.py

# Train with tool calling
python ../train.py --stage tools --model-size large
```

## Test Tool Calling

```bash
# Run inference with tools
python ../inference_tool_calling.py

# Run agent mode
python ../run_agent.py
```

## Available Scripts

| Script | Description |
|--------|-------------|
| `prepare_tool_calling_data.py` | Prepare tool calling dataset |
| `inference_tool_calling.py` | Test tool calling inference |
| `run_agent.py` | Run agent with tools |

## Tool Format

```json
{
  "function": {
    "name": "search_code",
    "parameters": {
      "query": "find database connection"
    }
  }
}
```

## Next Step

- For RLHF alignment: [Stage 5: RLHF](../5_rlhf/)
- For deployment: [Stage 6: Deploy](../6_deploy/)
