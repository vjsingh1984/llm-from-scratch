# Deployment Guide

This guide covers deploying the Code LLM model in production environments, from local development to cloud deployment.

## Table of Contents

1. [Local Development API](#local-development-api)
2. [Docker Deployment](#docker-deployment)
3. [Cloud Deployment](#cloud-deployment)
4. [Performance Optimization](#performance-optimization)
5. [Monitoring and Logging](#monitoring-and-logging)
6. [Security Considerations](#security-considerations)

---

## Local Development API

### Quick Start

Run the API server locally for development:

```bash
# Install additional dependencies
pip install fastapi uvicorn pydantic

# Run the API
python examples/deployment_api.py
```

The API will be available at:
- **API**: http://localhost:8000
- **Interactive Docs**: http://localhost:8000/docs
- **Alternative Docs**: http://localhost:8000/redoc

### Testing the API

Using curl:

```bash
# Health check
curl http://localhost:8000/health

# Model info
curl http://localhost:8000/info

# Generate code
curl -X POST http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "#!/bin/bash\n# Create a backup script",
    "max_length": 200,
    "temperature": 0.8,
    "top_k": 50,
    "top_p": 0.9
  }'
```

Using Python:

```python
import requests

# Generate code
response = requests.post(
    "http://localhost:8000/generate",
    json={
        "prompt": "#!/bin/bash\n# Monitor disk space",
        "max_length": 200,
        "temperature": 0.8
    }
)

result = response.json()
print(result['generated_code'])
```

Using JavaScript:

```javascript
// Generate code
fetch('http://localhost:8000/generate', {
  method: 'POST',
  headers: {'Content-Type': 'application/json'},
  body: JSON.stringify({
    prompt: '#!/bin/bash\n# Deploy application',
    max_length: 200,
    temperature: 0.8
  })
})
.then(res => res.json())
.then(data => console.log(data.generated_code));
```

---

## Docker Deployment

### Basic Docker Deployment

1. **Build the Docker image**:

```bash
docker build -t code-llm-api .
```

2. **Run the container**:

```bash
docker run -d \
  --name code-llm-api \
  -p 8000:8000 \
  code-llm-api
```

3. **Check logs**:

```bash
docker logs -f code-llm-api
```

4. **Stop the container**:

```bash
docker stop code-llm-api
docker rm code-llm-api
```

### Docker Compose Deployment

For easier management, use Docker Compose:

1. **Start services**:

```bash
docker-compose up -d
```

2. **View logs**:

```bash
docker-compose logs -f
```

3. **Stop services**:

```bash
docker-compose down
```

4. **Restart services**:

```bash
docker-compose restart
```

### Environment Variables

Configure the API using environment variables:

```bash
docker run -d \
  --name code-llm-api \
  -p 8000:8000 \
  -e MODEL_PATH=models/code/code_model_final.pt \
  -e TOKENIZER_PATH=models/language/language_tokenizer.json \
  -e MAX_GENERATION_LENGTH=500 \
  -e DEFAULT_TEMPERATURE=0.8 \
  code-llm-api
```

Available environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `MODEL_PATH` | `models/code/code_model_final.pt` | Path to model checkpoint |
| `TOKENIZER_PATH` | `models/language/language_tokenizer.json` | Path to tokenizer |
| `MAX_PROMPT_LENGTH` | `200` | Maximum prompt length in tokens |
| `MAX_GENERATION_LENGTH` | `500` | Maximum generation length |
| `DEFAULT_TEMPERATURE` | `0.8` | Default sampling temperature |
| `DEFAULT_TOP_K` | `50` | Default top-k parameter |
| `DEFAULT_TOP_P` | `0.9` | Default top-p parameter |

---

## Cloud Deployment

### AWS Deployment (EC2)

1. **Launch EC2 instance**:
   - Instance type: t3.medium or larger (for CPU) or g4dn.xlarge (for GPU)
   - AMI: Ubuntu 22.04 LTS
   - Storage: 20GB+ EBS volume
   - Security group: Allow inbound traffic on port 8000

2. **SSH into instance**:

```bash
ssh -i your-key.pem ubuntu@your-ec2-ip
```

3. **Install dependencies**:

```bash
# Update system
sudo apt update && sudo apt upgrade -y

# Install Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh
sudo usermod -aG docker ubuntu

# Install Docker Compose
sudo curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
sudo chmod +x /usr/local/bin/docker-compose
```

4. **Deploy application**:

```bash
# Clone repository
git clone <your-repo-url>
cd code-llm-from-scratch

# Start with Docker Compose
docker-compose up -d
```

5. **Setup Nginx reverse proxy** (optional but recommended):

```bash
# Install Nginx
sudo apt install nginx -y

# Configure
sudo nano /etc/nginx/sites-available/code-llm
```

Nginx configuration:

```nginx
server {
    listen 80;
    server_name your-domain.com;

    location / {
        proxy_pass http://localhost:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
    }
}
```

Enable and restart:

```bash
sudo ln -s /etc/nginx/sites-available/code-llm /etc/nginx/sites-enabled/
sudo nginx -t
sudo systemctl restart nginx
```

### Google Cloud Platform (Cloud Run)

1. **Build and push container**:

```bash
# Configure gcloud
gcloud auth configure-docker

# Build for Cloud Run
docker build -t gcr.io/your-project-id/code-llm-api .

# Push to Container Registry
docker push gcr.io/your-project-id/code-llm-api
```

2. **Deploy to Cloud Run**:

```bash
gcloud run deploy code-llm-api \
  --image gcr.io/your-project-id/code-llm-api \
  --platform managed \
  --region us-central1 \
  --memory 2Gi \
  --cpu 2 \
  --timeout 300 \
  --allow-unauthenticated
```

### Azure (Container Instances)

1. **Login to Azure**:

```bash
az login
```

2. **Create resource group**:

```bash
az group create --name code-llm-rg --location eastus
```

3. **Deploy container**:

```bash
az container create \
  --resource-group code-llm-rg \
  --name code-llm-api \
  --image code-llm-api:latest \
  --cpu 2 \
  --memory 4 \
  --ports 8000 \
  --dns-name-label code-llm-api-unique \
  --restart-policy Always
```

---

## Performance Optimization

### Model Optimization

1. **Model Quantization** (reduce model size and improve speed):

```python
import torch

# Load model
model = torch.load('models/code/code_model_final.pt')

# Quantize to int8
model_quantized = torch.quantization.quantize_dynamic(
    model,
    {torch.nn.Linear},
    dtype=torch.qint8
)

# Save quantized model
torch.save(model_quantized, 'models/code/code_model_quantized.pt')
```

2. **Batch Processing** (process multiple requests together):

```python
# Modify deployment_api.py to support batch generation
@app.post("/generate/batch")
async def generate_batch(requests: List[GenerateRequest]):
    results = []
    for req in requests:
        result = await generate_code(req)
        results.append(result)
    return results
```

3. **Caching** (cache common prompts):

```python
from functools import lru_cache

@lru_cache(maxsize=100)
def cached_generate(prompt: str, max_length: int, temperature: float):
    return model_manager.generate(prompt, max_length, temperature)
```

### Server Optimization

1. **Use Multiple Workers**:

```bash
# Run with 4 worker processes
uvicorn examples.deployment_api:app \
  --host 0.0.0.0 \
  --port 8000 \
  --workers 4
```

2. **Enable HTTP/2**:

```bash
uvicorn examples.deployment_api:app \
  --host 0.0.0.0 \
  --port 8000 \
  --http h11
```

3. **Add Response Compression**:

```python
from fastapi.middleware.gzip import GZipMiddleware

app.add_middleware(GZipMiddleware, minimum_size=1000)
```

---

## Monitoring and Logging

### Application Logging

Add structured logging to `deployment_api.py`:

```python
import logging
import json
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Log requests
@app.middleware("http")
async def log_requests(request: Request, call_next):
    start_time = time.time()

    response = await call_next(request)

    duration = time.time() - start_time
    logger.info(json.dumps({
        'timestamp': datetime.now().isoformat(),
        'method': request.method,
        'path': request.url.path,
        'status_code': response.status_code,
        'duration': duration
    }))

    return response
```

### Prometheus Metrics

Add Prometheus metrics:

```bash
pip install prometheus-fastapi-instrumentator
```

```python
from prometheus_fastapi_instrumentator import Instrumentator

app = FastAPI()

# Add metrics endpoint
Instrumentator().instrument(app).expose(app)
```

Access metrics at: http://localhost:8000/metrics

### Health Monitoring

Set up automated health checks:

```bash
# Using cron
*/5 * * * * curl -f http://localhost:8000/health || echo "API is down!"
```

---

## Security Considerations

### 1. Rate Limiting

Add rate limiting to prevent abuse:

```bash
pip install slowapi
```

```python
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

@app.post("/generate")
@limiter.limit("5/minute")
async def generate_code(request: Request, gen_request: GenerateRequest):
    # ... existing code
```

### 2. Input Validation

Already implemented via Pydantic models, but add additional checks:

```python
# Check for malicious patterns
BLOCKED_PATTERNS = ['rm -rf', 'dd if=', 'fork bomb']

def validate_prompt(prompt: str):
    for pattern in BLOCKED_PATTERNS:
        if pattern in prompt.lower():
            raise ValueError(f"Blocked pattern detected: {pattern}")
```

### 3. HTTPS/TLS

Use Let's Encrypt for free SSL certificates:

```bash
# Install certbot
sudo apt install certbot python3-certbot-nginx

# Get certificate
sudo certbot --nginx -d your-domain.com

# Auto-renewal
sudo certbot renew --dry-run
```

### 4. API Authentication

Add API key authentication:

```python
from fastapi.security import APIKeyHeader

API_KEY_HEADER = APIKeyHeader(name="X-API-Key")

async def verify_api_key(api_key: str = Depends(API_KEY_HEADER)):
    if api_key != os.getenv("API_KEY"):
        raise HTTPException(status_code=403, detail="Invalid API key")

@app.post("/generate")
async def generate_code(
    request: GenerateRequest,
    api_key: str = Depends(verify_api_key)
):
    # ... existing code
```

### 5. CORS Configuration

Restrict CORS in production:

```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://your-frontend.com"],  # Specific domains only
    allow_credentials=True,
    allow_methods=["POST", "GET"],
    allow_headers=["*"],
)
```

---

## Load Balancing and Scaling

### Kubernetes Deployment

1. **Create deployment.yaml**:

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: code-llm-api
spec:
  replicas: 3
  selector:
    matchLabels:
      app: code-llm-api
  template:
    metadata:
      labels:
        app: code-llm-api
    spec:
      containers:
      - name: api
        image: code-llm-api:latest
        ports:
        - containerPort: 8000
        resources:
          requests:
            memory: "2Gi"
            cpu: "1"
          limits:
            memory: "4Gi"
            cpu: "2"
---
apiVersion: v1
kind: Service
metadata:
  name: code-llm-api-service
spec:
  selector:
    app: code-llm-api
  ports:
  - port: 80
    targetPort: 8000
  type: LoadBalancer
```

2. **Deploy**:

```bash
kubectl apply -f deployment.yaml
kubectl get services
```

---

## Troubleshooting

### Common Issues

1. **Out of Memory**:
   - Reduce model size or use quantization
   - Increase container memory limit
   - Reduce batch size

2. **Slow Generation**:
   - Use GPU instance if available
   - Reduce `max_length` parameter
   - Enable model caching

3. **Connection Timeout**:
   - Increase timeout in uvicorn: `--timeout-keep-alive 300`
   - Check firewall rules

4. **Model Not Loading**:
   - Verify MODEL_PATH is correct
   - Check file permissions
   - Ensure sufficient disk space

---

## Production Checklist

Before deploying to production:

- [ ] Enable HTTPS/TLS
- [ ] Add API authentication
- [ ] Configure rate limiting
- [ ] Set up monitoring and alerting
- [ ] Configure proper CORS policy
- [ ] Add logging and metrics
- [ ] Test with production-like load
- [ ] Set up automated backups
- [ ] Document API endpoints
- [ ] Configure auto-scaling
- [ ] Set up CI/CD pipeline
- [ ] Perform security audit

---

## Support

For deployment issues:
- Check logs: `docker-compose logs -f`
- Health check: `curl http://localhost:8000/health`
- API docs: `http://localhost:8000/docs`
- GitHub Issues: [Create an issue](https://github.com/your-repo/issues)

---

**Ready for production!** Follow this guide to deploy your Code LLM API securely and efficiently.
