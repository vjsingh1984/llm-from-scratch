# Production Monitoring Guide

Learn how to monitor your deployed model in production, track performance, and detect issues early.

## Table of Contents

1. [Foundational Monitoring](#1-foundational-monitoring)
2. [Application Metrics](#2-application-metrics)
3. [Model Performance Tracking](#3-model-performance-tracking)
4. [Alerting and Notifications](#4-alerting-and-notifications)
5. [Advanced Observability](#5-advanced-observability)
6. [Debugging Production Issues](#6-debugging-production-issues)

---

## 1. Foundational Monitoring

Start with these basic metrics for any production system.

### 1.1 Health Checks

**Implementation in API**:

```python
# In deployment_api.py

from datetime import datetime

@app.get("/health")
async def health_check():
    """Basic health check endpoint."""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "model_loaded": model_manager.model is not None,
        "device": str(model_manager.device)
    }


@app.get("/health/detailed")
async def detailed_health():
    """Detailed health check with more information."""
    try:
        # Test model inference
        test_prompt = "#!/bin/bash\n# test"
        start_time = time.time()
        model_manager.generate(test_prompt, max_length=10)
        inference_time = time.time() - start_time

        return {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "model": {
                "loaded": True,
                "vocab_size": len(model_manager.tokenizer.vocab),
                "inference_test_ms": inference_time * 1000
            },
            "system": {
                "device": str(model_manager.device),
                "python_version": sys.version
            }
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }
```

**Monitor with curl**:

```bash
# Simple monitoring script
#!/bin/bash

while true; do
    response=$(curl -s http://localhost:8000/health)
    status=$(echo $response | jq -r '.status')

    if [ "$status" != "healthy" ]; then
        echo "[$(date)] ALERT: Service unhealthy - $response"
        # Send alert
    else
        echo "[$(date)] OK: Service healthy"
    fi

    sleep 60  # Check every minute
done
```

### 1.2 Uptime Monitoring

```python
import time

# Track service start time
SERVICE_START_TIME = time.time()

@app.get("/metrics/uptime")
async def uptime():
    """Get service uptime."""
    uptime_seconds = time.time() - SERVICE_START_TIME
    return {
        "uptime_seconds": uptime_seconds,
        "uptime_hours": uptime_seconds / 3600,
        "uptime_days": uptime_seconds / 86400,
        "started_at": datetime.fromtimestamp(SERVICE_START_TIME).isoformat()
    }
```

### 1.3 Request Logging

```python
import logging
import json

# Configure structured logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log all HTTP requests with timing."""
    start_time = time.time()

    # Process request
    response = await call_next(request)

    # Calculate duration
    duration = time.time() - start_time

    # Log request details
    log_data = {
        "timestamp": datetime.now().isoformat(),
        "method": request.method,
        "path": request.url.path,
        "status_code": response.status_code,
        "duration_ms": duration * 1000,
        "client_ip": request.client.host if request.client else None
    }

    logger.info(json.dumps(log_data))

    # Add duration header
    response.headers["X-Process-Time"] = str(duration)

    return response
```

---

## 2. Application Metrics

Track key performance indicators (KPIs) for your API.

### 2.1 Request Metrics with Prometheus

```bash
pip install prometheus-fastapi-instrumentator
```

```python
from prometheus_fastapi_instrumentator import Instrumentator
from prometheus_client import Counter, Histogram, Gauge

# Initialize Prometheus
instrumentator = Instrumentator()

# Custom metrics
generation_requests = Counter(
    'generation_requests_total',
    'Total number of generation requests'
)

generation_tokens = Histogram(
    'generation_tokens',
    'Number of tokens generated per request',
    buckets=[10, 25, 50, 100, 200, 300, 500]
)

generation_latency = Histogram(
    'generation_latency_seconds',
    'Time taken to generate code',
    buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
)

active_requests = Gauge(
    'active_generation_requests',
    'Number of currently processing requests'
)


@app.post("/generate")
async def generate_code(request: GenerateRequest):
    """Generate code with metrics."""
    generation_requests.inc()  # Increment counter
    active_requests.inc()      # Increment active requests

    try:
        start_time = time.time()

        # Generate
        code, num_tokens, gen_time = model_manager.generate(
            request.prompt,
            request.max_length,
            request.temperature,
            request.top_k,
            request.top_p
        )

        # Record metrics
        generation_latency.observe(gen_time)
        generation_tokens.observe(num_tokens)

        return GenerateResponse(
            generated_code=code,
            prompt=request.prompt,
            num_tokens_generated=num_tokens,
            generation_time=gen_time,
            model_info={
                'temperature': request.temperature,
                'top_k': request.top_k,
                'top_p': request.top_p,
            }
        )

    finally:
        active_requests.dec()  # Decrement active requests


# Expose metrics endpoint
instrumentator.instrument(app).expose(app, endpoint="/metrics")
```

**Access metrics**:

```bash
curl http://localhost:8000/metrics
```

**Sample output**:

```
# HELP generation_requests_total Total number of generation requests
# TYPE generation_requests_total counter
generation_requests_total 1523.0

# HELP generation_latency_seconds Time taken to generate code
# TYPE generation_latency_seconds histogram
generation_latency_seconds_bucket{le="0.5"} 1245.0
generation_latency_seconds_bucket{le="1.0"} 1480.0
generation_latency_seconds_sum 856.3
generation_latency_seconds_count 1523

# HELP generation_tokens Number of tokens generated per request
# TYPE generation_tokens histogram
generation_tokens_bucket{le="50.0"} 234.0
generation_tokens_bucket{le="100.0"} 987.0
generation_tokens_sum 145678.0
generation_tokens_count 1523
```

### 2.2 Error Tracking

```python
from prometheus_client import Counter

# Error counters
generation_errors = Counter(
    'generation_errors_total',
    'Total number of generation errors',
    ['error_type']
)


@app.post("/generate")
async def generate_code(request: GenerateRequest):
    """Generate code with error tracking."""
    try:
        code, num_tokens, gen_time = model_manager.generate(...)
        return GenerateResponse(...)

    except ValueError as e:
        generation_errors.labels(error_type='validation').inc()
        raise HTTPException(status_code=400, detail=str(e))

    except torch.cuda.OutOfMemoryError:
        generation_errors.labels(error_type='oom').inc()
        raise HTTPException(status_code=503, detail="Out of memory")

    except Exception as e:
        generation_errors.labels(error_type='unknown').inc()
        logger.error(f"Generation failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")
```

### 2.3 Quality Metrics

Track generation quality in production:

```python
from prometheus_client import Histogram

# Quality metrics
syntax_validity_rate = Gauge(
    'syntax_validity_rate',
    'Percentage of syntactically valid generations'
)

prompt_similarity = Histogram(
    'prompt_similarity_score',
    'How well generated code matches the prompt',
    buckets=[0.1, 0.3, 0.5, 0.7, 0.9, 1.0]
)


async def check_syntax_async(code: str) -> bool:
    """Async syntax check (non-blocking)."""
    try:
        result = await asyncio.create_subprocess_exec(
            'bash', '-n',
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        await result.communicate(input=code.encode())
        return result.returncode == 0
    except:
        return False


@app.post("/generate")
async def generate_code(request: GenerateRequest):
    """Generate with quality tracking."""
    code, num_tokens, gen_time = model_manager.generate(...)

    # Check syntax (async, don't block response)
    asyncio.create_task(check_and_log_quality(code, request.prompt))

    return GenerateResponse(...)


async def check_and_log_quality(code: str, prompt: str):
    """Background task to check and log quality metrics."""
    is_valid = await check_syntax_async(code)

    # Update syntax validity rate
    # (This is simplified - you'd want a sliding window)
    if is_valid:
        logger.info("Generated valid syntax")
    else:
        logger.warning("Generated invalid syntax")
```

---

## 3. Model Performance Tracking

Monitor model-specific metrics.

### 3.1 Token Statistics

```python
# Track token statistics
@app.post("/generate")
async def generate_code(request: GenerateRequest):
    """Track token-level statistics."""
    code, num_tokens, gen_time = model_manager.generate(...)

    # Log token statistics
    logger.info(json.dumps({
        "event": "generation_complete",
        "prompt_tokens": len(model_manager.tokenizer.encode(request.prompt)),
        "generated_tokens": num_tokens,
        "total_tokens": len(model_manager.tokenizer.encode(code)),
        "tokens_per_second": num_tokens / gen_time if gen_time > 0 else 0,
        "generation_time_ms": gen_time * 1000
    }))

    return GenerateResponse(...)
```

### 3.2 Sampling Parameter Tracking

```python
# Track what sampling parameters are being used
param_usage = Counter(
    'sampling_parameters',
    'Usage of different sampling parameters',
    ['parameter', 'value_range']
)


@app.post("/generate")
async def generate_code(request: GenerateRequest):
    """Track sampling parameter usage."""

    # Track temperature ranges
    if request.temperature < 0.5:
        param_usage.labels(parameter='temperature', value_range='low').inc()
    elif request.temperature < 1.0:
        param_usage.labels(parameter='temperature', value_range='medium').inc()
    else:
        param_usage.labels(parameter='temperature', value_range='high').inc()

    # Track top_k ranges
    if request.top_k < 25:
        param_usage.labels(parameter='top_k', value_range='low').inc()
    elif request.top_k < 75:
        param_usage.labels(parameter='top_k', value_range='medium').inc()
    else:
        param_usage.labels(parameter='top_k', value_range='high').inc()

    code, num_tokens, gen_time = model_manager.generate(...)
    return GenerateResponse(...)
```

### 3.3 Output Length Distribution

```python
# Track generated code lengths
output_length = Histogram(
    'output_length_chars',
    'Length of generated code in characters',
    buckets=[100, 250, 500, 1000, 2000, 5000]
)


@app.post("/generate")
async def generate_code(request: GenerateRequest):
    """Track output length."""
    code, num_tokens, gen_time = model_manager.generate(...)

    # Record length
    output_length.observe(len(code))

    return GenerateResponse(...)
```

---

## 4. Alerting and Notifications

Set up alerts for critical issues.

### 4.1 Slack Notifications

```python
import requests

SLACK_WEBHOOK_URL = os.getenv("SLACK_WEBHOOK_URL")


def send_slack_alert(message: str, severity: str = "warning"):
    """Send alert to Slack."""
    if not SLACK_WEBHOOK_URL:
        return

    color = {
        "info": "#36a64f",
        "warning": "#ff9800",
        "error": "#f44336"
    }.get(severity, "#808080")

    payload = {
        "attachments": [{
            "color": color,
            "title": f"Code LLM API Alert [{severity.upper()}]",
            "text": message,
            "ts": int(time.time())
        }]
    }

    try:
        requests.post(SLACK_WEBHOOK_URL, json=payload, timeout=5)
    except Exception as e:
        logger.error(f"Failed to send Slack alert: {e}")


# Use in error handlers
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Handle exceptions and send alerts."""
    error_msg = f"Unhandled exception: {exc}\nPath: {request.url.path}"

    logger.error(error_msg, exc_info=True)
    send_slack_alert(error_msg, severity="error")

    return JSONResponse(
        status_code=500,
        content={"error": "Internal server error"}
    )


# Alert on high error rate
error_count = 0
ALERT_THRESHOLD = 10

@app.middleware("http")
async def error_rate_monitor(request: Request, call_next):
    """Monitor error rate."""
    global error_count

    response = await call_next(request)

    if response.status_code >= 500:
        error_count += 1

        if error_count >= ALERT_THRESHOLD:
            send_slack_alert(
                f"High error rate detected: {error_count} errors in recent window",
                severity="error"
            )
            error_count = 0  # Reset

    return response
```

### 4.2 Email Alerts

```python
import smtplib
from email.mime.text import MIMEText

def send_email_alert(subject: str, body: str):
    """Send email alert."""
    smtp_server = os.getenv("SMTP_SERVER")
    smtp_port = int(os.getenv("SMTP_PORT", 587))
    smtp_user = os.getenv("SMTP_USER")
    smtp_password = os.getenv("SMTP_PASSWORD")
    alert_email = os.getenv("ALERT_EMAIL")

    if not all([smtp_server, smtp_user, smtp_password, alert_email]):
        return

    msg = MIMEText(body)
    msg['Subject'] = f"Code LLM API: {subject}"
    msg['From'] = smtp_user
    msg['To'] = alert_email

    try:
        with smtplib.SMTP(smtp_server, smtp_port) as server:
            server.starttls()
            server.login(smtp_user, smtp_password)
            server.send_message(msg)
    except Exception as e:
        logger.error(f"Failed to send email alert: {e}")
```

### 4.3 Threshold-Based Alerts

```python
# Monitor latency
LATENCY_THRESHOLD = 5.0  # seconds

@app.post("/generate")
async def generate_code(request: GenerateRequest):
    """Monitor generation latency."""
    start_time = time.time()

    code, num_tokens, gen_time = model_manager.generate(...)

    if gen_time > LATENCY_THRESHOLD:
        send_slack_alert(
            f"Slow generation detected: {gen_time:.2f}s (threshold: {LATENCY_THRESHOLD}s)\n"
            f"Prompt: {request.prompt[:100]}...",
            severity="warning"
        )

    return GenerateResponse(...)
```

---

## 5. Advanced Observability

### 5.1 Grafana Dashboards

**Connect Prometheus to Grafana**:

1. Add Prometheus data source in Grafana
2. Create dashboard with panels:

```json
{
  "dashboard": {
    "title": "Code LLM API Monitoring",
    "panels": [
      {
        "title": "Request Rate",
        "targets": [
          {
            "expr": "rate(generation_requests_total[5m])"
          }
        ]
      },
      {
        "title": "Latency (p50, p95, p99)",
        "targets": [
          {
            "expr": "histogram_quantile(0.50, generation_latency_seconds)"
          },
          {
            "expr": "histogram_quantile(0.95, generation_latency_seconds)"
          },
          {
            "expr": "histogram_quantile(0.99, generation_latency_seconds)"
          }
        ]
      },
      {
        "title": "Token Generation Rate",
        "targets": [
          {
            "expr": "rate(generation_tokens_sum[5m]) / rate(generation_tokens_count[5m])"
          }
        ]
      },
      {
        "title": "Error Rate",
        "targets": [
          {
            "expr": "rate(generation_errors_total[5m])"
          }
        ]
      }
    ]
  }
}
```

### 5.2 Distributed Tracing

```bash
pip install opentelemetry-api opentelemetry-sdk opentelemetry-instrumentation-fastapi
```

```python
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor

# Setup tracing
trace.set_tracer_provider(TracerProvider())
tracer = trace.get_tracer(__name__)

# Add exporter
span_processor = BatchSpanProcessor(ConsoleSpanExporter())
trace.get_tracer_provider().add_span_processor(span_processor)

# Instrument FastAPI
FastAPIInstrumentor.instrument_app(app)


@app.post("/generate")
async def generate_code(request: GenerateRequest):
    """Generate with tracing."""
    with tracer.start_as_current_span("generate_code") as span:
        # Add attributes
        span.set_attribute("prompt.length", len(request.prompt))
        span.set_attribute("max_length", request.max_length)
        span.set_attribute("temperature", request.temperature)

        # Tokenization span
        with tracer.start_as_current_span("tokenize"):
            input_ids = model_manager.tokenizer.encode(request.prompt)
            span.set_attribute("input_tokens", len(input_ids))

        # Generation span
        with tracer.start_as_current_span("model_inference"):
            code, num_tokens, gen_time = model_manager.generate(...)
            span.set_attribute("output_tokens", num_tokens)
            span.set_attribute("generation_time_ms", gen_time * 1000)

        return GenerateResponse(...)
```

### 5.3 Custom Dashboards

```python
# Create simple web dashboard
@app.get("/dashboard", response_class=HTMLResponse)
async def dashboard():
    """Simple monitoring dashboard."""
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Code LLM Monitoring</title>
        <meta http-equiv="refresh" content="30">
        <style>
            body { font-family: Arial; padding: 20px; }
            .metric { padding: 15px; margin: 10px 0; background: #f5f5f5; border-radius: 5px; }
            .metric h3 { margin: 0 0 10px 0; }
            .value { font-size: 24px; font-weight: bold; color: #2196F3; }
            .healthy { color: #4CAF50; }
            .warning { color: #FF9800; }
            .error { color: #F44336; }
        </style>
    </head>
    <body>
        <h1>Code LLM API Monitoring</h1>

        <div class="metric">
            <h3>Service Status</h3>
            <div class="value healthy">HEALTHY</div>
        </div>

        <div class="metric">
            <h3>Uptime</h3>
            <div class="value">{{ uptime_hours }} hours</div>
        </div>

        <div class="metric">
            <h3>Total Requests</h3>
            <div class="value">{{ total_requests }}</div>
        </div>

        <div class="metric">
            <h3>Average Latency</h3>
            <div class="value">{{ avg_latency }} ms</div>
        </div>

        <p><small>Auto-refreshes every 30 seconds</small></p>
    </body>
    </html>
    """
    return html_content
```

---

## 6. Debugging Production Issues

### 6.1 Structured Logging

```python
import logging
import json

# Custom JSON logger
class JSONFormatter(logging.Formatter):
    """Format logs as JSON."""
    def format(self, record):
        log_data = {
            'timestamp': datetime.now().isoformat(),
            'level': record.levelname,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno
        }

        # Add exception info if present
        if record.exc_info:
            log_data['exception'] = self.formatException(record.exc_info)

        return json.dumps(log_data)


# Setup JSON logging
handler = logging.StreamHandler()
handler.setFormatter(JSONFormatter())
logger = logging.getLogger()
logger.addHandler(handler)
logger.setLevel(logging.INFO)
```

### 6.2 Debug Mode

```python
DEBUG_MODE = os.getenv("DEBUG_MODE", "false").lower() == "true"


@app.post("/generate")
async def generate_code(request: GenerateRequest):
    """Generate with optional debug info."""
    start_time = time.time()

    code, num_tokens, gen_time = model_manager.generate(...)

    response_data = {
        "generated_code": code,
        "prompt": request.prompt,
        "num_tokens_generated": num_tokens,
        "generation_time": gen_time
    }

    # Add debug info if enabled
    if DEBUG_MODE:
        response_data["debug"] = {
            "input_tokens": len(model_manager.tokenizer.encode(request.prompt)),
            "model_config": {
                "vocab_size": model_manager.config.vocab_size,
                "n_layers": model_manager.config.n_layers,
                "d_model": model_manager.config.d_model
            },
            "sampling_params": {
                "temperature": request.temperature,
                "top_k": request.top_k,
                "top_p": request.top_p
            },
            "total_processing_time": time.time() - start_time
        }

    return response_data
```

---

## Summary

### Monitoring Checklist

- [ ] **Health Checks**: `/health` endpoint implemented and monitored
- [ ] **Request Logging**: All requests logged with timing
- [ ] **Metrics**: Prometheus metrics exposed
- [ ] **Errors**: Error tracking and alerting configured
- [ ] **Quality**: Syntax validity and quality metrics tracked
- [ ] **Alerts**: Slack/email alerts for critical issues
- [ ] **Dashboard**: Grafana or custom dashboard set up
- [ ] **Debugging**: Structured logging and debug mode available

### Key Metrics to Track

| Metric | Type | Threshold |
|--------|------|-----------|
| Request Rate | Counter | - |
| Latency (p95) | Histogram | < 2s |
| Error Rate | Counter | < 1% |
| Syntax Validity | Gauge | > 85% |
| Active Requests | Gauge | < 100 |
| Memory Usage | Gauge | < 80% |

---

**Next**: See [docs/EVALUATION.md](EVALUATION.md) for quality evaluation strategies.
