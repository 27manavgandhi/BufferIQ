\# BufferIQ API Deployment Guide



\## Overview



This guide covers deploying the BufferIQ API in various environments: local development, Docker, and Kubernetes.



\## Table of Contents



\- \[Prerequisites](#prerequisites)

\- \[Local Development](#local-development)

\- \[Docker Deployment](#docker-deployment)

\- \[Kubernetes Deployment](#kubernetes-deployment)

\- \[Environment Variables](#environment-variables)

\- \[Monitoring \& Logging](#monitoring--logging)

\- \[Scaling](#scaling)

\- \[Troubleshooting](#troubleshooting)



\## Prerequisites



\### System Requirements



\- \*\*CPU:\*\* 2+ cores recommended

\- \*\*Memory:\*\* 4GB+ RAM (8GB+ for production)

\- \*\*Storage:\*\* 10GB+ available space

\- \*\*OS:\*\* Linux, macOS, or Windows with WSL2



\### Software Dependencies



\- Python 3.10+

\- Redis 6.0+

\- Docker 20.10+ (for containerized deployment)

\- Kubernetes 1.20+ (for K8s deployment)



\## Local Development



\### 1. Install Dependencies



```bash

cd backend

pip install -r requirements.txt

```



\### 2. Start Redis



```bash

\# Using Docker

docker run -d -p 6379:6379 redis:7-alpine



\# Or using local Redis

redis-server

```



\### 3. Train Models (if not already done)



```bash

\# Ensure models exist

python scripts/train\_models.py

```



\### 4. Start API Server



```bash

\# Using script

python scripts/start\_api.py --config configs/api/development.yaml



\# Or using uvicorn directly

uvicorn bufferiq.api.app:app --reload --host 127.0.0.1 --port 8000

```



\### 5. Verify Deployment



```bash

\# Check health

curl http://localhost:8000/health



\# Test prediction

curl -X POST http://localhost:8000/api/v1/predict \\

&#x20; -H "Content-Type: application/json" \\

&#x20; -d '{"content":"Test","platform":"linkedin"}'

```



\## Docker Deployment



\### 1. Build Image



Create `Dockerfile`:



```dockerfile

FROM python:3.10-slim



WORKDIR /app



\# Install system dependencies

RUN apt-get update \&\& apt-get install -y \\

&#x20;   gcc \\

&#x20;   \&\& rm -rf /var/lib/apt/lists/\*



\# Copy requirements

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt



\# Copy application code

COPY bufferiq/ ./bufferiq/

COPY configs/ ./configs/



\# Copy models

COPY outputs/models/ /app/models/



\# Expose port

EXPOSE 8000



\# Start server

CMD \["uvicorn", "bufferiq.api.app:app", "--host", "0.0.0.0", "--port", "8000"]

```



Build the image:



```bash

docker build -t bufferiq-api:latest .

```



\### 2. Docker Compose



Create `docker-compose.yml`:



```yaml

version: '3.8'



services:

&#x20; api:

&#x20;   image: bufferiq-api:latest

&#x20;   ports:

&#x20;     - "8000:8000"

&#x20;   environment:

&#x20;     - REDIS\_URL=redis://redis:6379

&#x20;     - LOG\_LEVEL=INFO

&#x20;   depends\_on:

&#x20;     - redis

&#x20;   volumes:

&#x20;     - ./outputs/models:/app/models:ro

&#x20;   healthcheck:

&#x20;     test: \["CMD", "curl", "-f", "http://localhost:8000/health"]

&#x20;     interval: 30s

&#x20;     timeout: 10s

&#x20;     retries: 3



&#x20; redis:

&#x20;   image: redis:7-alpine

&#x20;   ports:

&#x20;     - "6379:6379"

&#x20;   volumes:

&#x20;     - redis-data:/data

&#x20;   command: redis-server --appendonly yes



volumes:

&#x20; redis-data:

```



\### 3. Start Services



```bash

docker-compose up -d

```



\### 4. View Logs



```bash

docker-compose logs -f api

```



\### 5. Stop Services



```bash

docker-compose down

```



\## Kubernetes Deployment



\### 1. Create Namespace



```yaml

\# namespace.yaml

apiVersion: v1

kind: Namespace

metadata:

&#x20; name: bufferiq

```



```bash

kubectl apply -f namespace.yaml

```



\### 2. Create ConfigMap



```yaml

\# configmap.yaml

apiVersion: v1

kind: ConfigMap

metadata:

&#x20; name: bufferiq-config

&#x20; namespace: bufferiq

data:

&#x20; config.yaml: |

&#x20;   app:

&#x20;     name: "BufferIQ API"

&#x20;     version: "1.0.0"

&#x20;     debug: false

&#x20;     workers: 4

&#x20;   cache:

&#x20;     enabled: true

&#x20;     host: "redis"

&#x20;     port: "6379"

&#x20;   logging:

&#x20;     level: "INFO"

```



\### 3. Create Deployment



```yaml

\# deployment.yaml

apiVersion: apps/v1

kind: Deployment

metadata:

&#x20; name: bufferiq-api

&#x20; namespace: bufferiq

spec:

&#x20; replicas: 3

&#x20; selector:

&#x20;   matchLabels:

&#x20;     app: bufferiq-api

&#x20; template:

&#x20;   metadata:

&#x20;     labels:

&#x20;       app: bufferiq-api

&#x20;   spec:

&#x20;     containers:

&#x20;     - name: api

&#x20;       image: bufferiq-api:latest

&#x20;       ports:

&#x20;       - containerPort: 8000

&#x20;       env:

&#x20;       - name: REDIS\_URL

&#x20;         value: "redis://redis:6379"

&#x20;       resources:

&#x20;         requests:

&#x20;           memory: "512Mi"

&#x20;           cpu: "500m"

&#x20;         limits:

&#x20;           memory: "2Gi"

&#x20;           cpu: "2000m"

&#x20;       livenessProbe:

&#x20;         httpGet:

&#x20;           path: /health/live

&#x20;           port: 8000

&#x20;         initialDelaySeconds: 30

&#x20;         periodSeconds: 10

&#x20;       readinessProbe:

&#x20;         httpGet:

&#x20;           path: /health/ready

&#x20;           port: 8000

&#x20;         initialDelaySeconds: 15

&#x20;         periodSeconds: 5

```



\### 4. Create Service



```yaml

\# service.yaml

apiVersion: v1

kind: Service

metadata:

&#x20; name: bufferiq-api

&#x20; namespace: bufferiq

spec:

&#x20; selector:

&#x20;   app: bufferiq-api

&#x20; ports:

&#x20; - protocol: TCP

&#x20;   port: 80

&#x20;   targetPort: 8000

&#x20; type: LoadBalancer

```



\### 5. Create Ingress



```yaml

\# ingress.yaml

apiVersion: networking.k8s.io/v1

kind: Ingress

metadata:

&#x20; name: bufferiq-ingress

&#x20; namespace: bufferiq

&#x20; annotations:

&#x20;   cert-manager.io/cluster-issuer: "letsencrypt-prod"

spec:

&#x20; tls:

&#x20; - hosts:

&#x20;   - api.bufferiq.com

&#x20;   secretName: bufferiq-tls

&#x20; rules:

&#x20; - host: api.bufferiq.com

&#x20;   http:

&#x20;     paths:

&#x20;     - path: /

&#x20;       pathType: Prefix

&#x20;       backend:

&#x20;         service:

&#x20;           name: bufferiq-api

&#x20;           port:

&#x20;             number: 80

```



\### 6. Deploy to Kubernetes



```bash

kubectl apply -f configmap.yaml

kubectl apply -f deployment.yaml

kubectl apply -f service.yaml

kubectl apply -f ingress.yaml

```



\### 7. Verify Deployment



```bash

\# Check pods

kubectl get pods -n bufferiq



\# Check logs

kubectl logs -f deployment/bufferiq-api -n bufferiq



\# Check service

kubectl get svc -n bufferiq

```



\## Environment Variables



\### Required Variables



\- `REDIS\_URL`: Redis connection URL (default: `redis://localhost:6379`)



\### Optional Variables



\- `LOG\_LEVEL`: Logging level (default: `INFO`)

\- `WORKERS`: Number of worker processes (default: `4`)

\- `MAX\_BATCH\_SIZE`: Maximum batch size (default: `100`)

\- `CACHE\_TTL`: Cache TTL in seconds (default: `3600`)

\- `RATE\_LIMIT\_RPM`: Rate limit per minute (default: `1000`)



\### Example `.env` File



```env

\# Redis

REDIS\_URL=redis://redis:6379



\# Logging

LOG\_LEVEL=INFO



\# Performance

WORKERS=4

MAX\_BATCH\_SIZE=100



\# Caching

CACHE\_TTL=7200



\# Rate Limiting

RATE\_LIMIT\_RPM=1000

```



\## Monitoring \& Logging



\### Prometheus Metrics



Metrics available at `/metrics`:



```bash

curl http://localhost:8000/metrics

```



Key metrics:

\- `bufferiq\_requests\_total`: Total requests

\- `bufferiq\_request\_duration\_seconds`: Request latency

\- `bufferiq\_errors\_total`: Total errors

\- `bufferiq\_cache\_hits\_total`: Cache hits



\### Grafana Dashboard



Import dashboard JSON from `deployment/grafana/dashboard.json`



\### Logging



Logs are output in JSON format to stdout:



```json

{

&#x20; "timestamp": "2026-04-27T10:30:00Z",

&#x20; "level": "INFO",

&#x20; "message": "Request completed",

&#x20; "request\_id": "abc123",

&#x20; "method": "POST",

&#x20; "path": "/api/v1/predict",

&#x20; "status\_code": 200,

&#x20; "duration\_ms": 45.2

}

```



\## Scaling



\### Horizontal Scaling (Kubernetes)



```bash

\# Scale to 5 replicas

kubectl scale deployment/bufferiq-api --replicas=5 -n bufferiq



\# Auto-scaling

kubectl autoscale deployment/bufferiq-api \\

&#x20; --min=3 --max=10 --cpu-percent=70 -n bufferiq

```



\### Vertical Scaling



Update resource limits in deployment:



```yaml

resources:

&#x20; requests:

&#x20;   memory: "1Gi"

&#x20;   cpu: "1000m"

&#x20; limits:

&#x20;   memory: "4Gi"

&#x20;   cpu: "4000m"

```



\### Load Balancing



Use NGINX or cloud load balancer:



```nginx

upstream bufferiq\_api {

&#x20;   least\_conn;

&#x20;   server api-1.bufferiq.com:8000;

&#x20;   server api-2.bufferiq.com:8000;

&#x20;   server api-3.bufferiq.com:8000;

}



server {

&#x20;   listen 80;

&#x20;   server\_name api.bufferiq.com;



&#x20;   location / {

&#x20;       proxy\_pass http://bufferiq\_api;

&#x20;       proxy\_set\_header Host $host;

&#x20;       proxy\_set\_header X-Real-IP $remote\_addr;

&#x20;   }

}

```



\## Troubleshooting



\### API Not Starting



\*\*Check logs:\*\*

```bash

docker-compose logs api

\# or

kubectl logs deployment/bufferiq-api -n bufferiq

```



\*\*Common issues:\*\*

\- Models not found: Ensure models are in correct path

\- Redis connection failed: Check Redis is running

\- Port already in use: Change port in config



\### Slow Predictions



\*\*Check:\*\*

1\. Model warmup completed: Look for "models loaded" in logs

2\. Redis latency: Run `redis-cli --latency`

3\. Resource usage: Check CPU/memory with `docker stats`



\*\*Solutions:\*\*

\- Increase workers

\- Enable response caching

\- Scale horizontally



\### High Error Rate



\*\*Check:\*\*

1\. Request validation errors: Review error logs

2\. Model loading errors: Check model files exist

3\. Memory issues: Increase container memory



\### Cache Not Working



\*\*Verify Redis:\*\*

```bash

redis-cli ping

\# Should return: PONG

```



\*\*Check cache stats:\*\*

```bash

curl http://localhost:8000/api/v1/cache/stats

```



\## Best Practices



1\. \*\*Always use health checks\*\* in production

2\. \*\*Enable monitoring\*\* with Prometheus/Grafana

3\. \*\*Set resource limits\*\* to prevent resource exhaustion

4\. \*\*Use rolling updates\*\* for zero-downtime deployments

5\. \*\*Keep models in persistent storage\*\* (not in container)

6\. \*\*Enable HTTPS\*\* in production

7\. \*\*Implement proper logging\*\* for debugging

8\. \*\*Use secrets management\*\* for sensitive data

9\. \*\*Regular backups\*\* of model artifacts

10\. \*\*Load testing\*\* before production deployment

