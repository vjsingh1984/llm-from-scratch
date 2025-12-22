# Stage 6: Deployment

Deploy the model for production use.

## Quick Start

```bash
# Quantize model (optional, for smaller size)
python ../quantize_model.py

# Start API server
python ../serve_api.py
```

## API Server

```bash
# Start with default settings
python ../serve_api.py

# With custom port
python ../serve_api.py --port 8080

# With CORS enabled
python ../serve_api.py --cors
```

### API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/generate` | POST | Generate code |
| `/health` | GET | Health check |
| `/metrics` | GET | Model metrics |

### Example Request

```bash
curl -X POST http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt": "#!/bin/bash\n# Backup script", "max_tokens": 200}'
```

## Quantization

```bash
# 8-bit quantization
python ../quantize_model.py --bits 8

# 4-bit quantization
python ../quantize_model.py --bits 4
```

## Docker Deployment

```bash
# Build image
docker build -t code-model .

# Run container
docker run -p 8000:8000 code-model
```

## See Also

- [Docker Compose](../../docker-compose.yml)
- [Production Guide](../../docs/05-DEPLOYMENT.md)
