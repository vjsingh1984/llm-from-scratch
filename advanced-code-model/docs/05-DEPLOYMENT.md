# Deployment Guide

> Deploy your trained model to production.

---

## Deployment Options

| Option | Best For | Setup Time |
|--------|----------|------------|
| Local API | Development, testing | 5 min |
| Docker | Portable deployment | 15 min |
| Cloud (Modal/RunPod) | Scalable production | 30 min |

---

## Option 1: Local API Server

The simplest way to serve your model.

### Start Server

```bash
python scripts/serve_api.py \
    --checkpoint models/code_model_best.pth \
    --port 8080
```

### Test with cURL

```bash
curl -X POST http://localhost:8080/generate \
    -H "Content-Type: application/json" \
    -d '{"prompt": "#!/bin/bash\n# List files\n", "max_tokens": 100}'
```

### Python Client

```python
import requests

response = requests.post(
    "http://localhost:8080/generate",
    json={
        "prompt": "#!/bin/bash\n# Backup directory\n",
        "max_tokens": 100,
        "temperature": 0.7
    }
)
print(response.json()["text"])
```

---

## Option 2: Docker Deployment

### Build Image

```bash
docker build -t code-model .
```

### Run Container

```bash
docker run -d \
    --name code-model \
    -p 8080:8080 \
    -v $(pwd)/models:/app/models \
    code-model
```

### Docker Compose

```bash
docker-compose up -d
```

Check `docker-compose.yml` for configuration options.

### Health Check

```bash
curl http://localhost:8080/health
# {"status": "healthy", "model": "loaded"}
```

---

## Option 3: Cloud Deployment

### Modal

```bash
# Install Modal
pip install modal

# Deploy
modal deploy scripts/modal_deploy.py
```

### RunPod

```bash
# Create a template with the Dockerfile
# Configure GPU type (A100 recommended)
# Deploy and get endpoint URL
```

---

## API Reference

### POST /generate

Generate text from a prompt.

**Request:**
```json
{
    "prompt": "#!/bin/bash\n# List files\n",
    "max_tokens": 100,
    "temperature": 0.7,
    "top_k": 50,
    "top_p": 0.95,
    "stop_sequences": ["###"]
}
```

**Response:**
```json
{
    "text": "#!/bin/bash\n# List files\nls -la",
    "tokens_generated": 12,
    "finish_reason": "length"
}
```

### POST /tool_call

Execute a function call.

**Request:**
```json
{
    "prompt": "What files are in the current directory?",
    "tools": [
        {
            "name": "list_files",
            "description": "List files in a directory",
            "parameters": {"path": {"type": "string"}}
        }
    ]
}
```

**Response:**
```json
{
    "tool_call": {
        "name": "list_files",
        "arguments": {"path": "."}
    }
}
```

### GET /health

Check server status.

**Response:**
```json
{
    "status": "healthy",
    "model": "loaded",
    "device": "mps",
    "memory_used": "8.5GB"
}
```

---

## Performance Optimization

### Quantization

Reduce memory and increase speed with INT8 quantization.

```bash
python scripts/quantize.py \
    --checkpoint models/code_model_best.pth \
    --output models/code_model_int8.pth \
    --bits 8
```

### KV-Cache

Enable KV-cache for faster generation:

```python
# In server config
config.use_kv_cache = True
```

### Batching

For high throughput, enable request batching:

```python
# In server config
config.max_batch_size = 8
config.batch_timeout_ms = 50
```

---

## Monitoring

### Prometheus Metrics

```bash
python scripts/serve_api.py --metrics-port 9090
```

Metrics available:
- `model_requests_total`
- `model_latency_seconds`
- `model_tokens_generated_total`
- `model_memory_bytes`

### Logging

```bash
# Enable structured logging
python scripts/serve_api.py --log-format json

# View logs
docker logs -f code-model
```

---

## Security

### API Key Authentication

```bash
# Set API key
export API_KEY=your-secret-key

# Start server with auth
python scripts/serve_api.py --require-auth
```

### Rate Limiting

```yaml
# In config
rate_limit:
    requests_per_minute: 60
    tokens_per_minute: 10000
```

### Input Validation

The server automatically:
- Limits prompt length to `max_seq_len`
- Sanitizes special characters
- Validates JSON schema

---

## Production Checklist

- [ ] Model checkpoint tested and working
- [ ] Docker image built and tested
- [ ] API endpoints tested with sample requests
- [ ] Health check endpoint working
- [ ] Metrics endpoint configured
- [ ] Logging configured
- [ ] API key authentication enabled
- [ ] Rate limiting configured
- [ ] Error handling tested
- [ ] Load testing completed

---

## Troubleshooting

### Server Won't Start

```bash
# Check if port is in use
lsof -i :8080

# Check model file exists
ls -la models/code_model_best.pth
```

### Out of Memory

```bash
# Use smaller model
--model-size medium

# Enable quantization
--quantize int8

# Reduce batch size
--max-batch-size 1
```

### Slow Responses

```bash
# Check device is correct
--device mps  # or cuda

# Enable compilation
--compile

# Enable KV-cache
--use-kv-cache
```

---

## Example Production Setup

```yaml
# docker-compose.prod.yml
version: '3.8'
services:
  model:
    image: code-model:latest
    ports:
      - "8080:8080"
      - "9090:9090"
    volumes:
      - ./models:/app/models
    environment:
      - API_KEY=${API_KEY}
      - LOG_LEVEL=info
    deploy:
      resources:
        reservations:
          devices:
            - capabilities: [gpu]
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8080/health"]
      interval: 30s
      timeout: 10s
      retries: 3
```

---

## Next Steps

- **Understanding the model**: [02-CONCEPTS.md](02-CONCEPTS.md)
- **Training improvements**: [04-TRAINING-PIPELINE.md](04-TRAINING-PIPELINE.md)
- **Troubleshooting**: [reference/TROUBLESHOOTING.md](reference/TROUBLESHOOTING.md)
