\# BufferIQ API Monitoring Guide



\## Overview



This guide covers monitoring, observability, and performance tuning for the BufferIQ API.



\## Table of Contents



\- \[Metrics](#metrics)

\- \[Logging](#logging)

\- \[Alerting](#alerting)

\- \[Performance Tuning](#performance-tuning)

\- \[Dashboards](#dashboards)



\## Metrics



\### Prometheus Metrics



The API exposes Prometheus metrics at `/metrics`:



```bash

curl http://localhost:8000/metrics

```



\### Available Metrics



\#### Request Metrics



\*\*`bufferiq\_requests\_total`\*\* (Counter)

\- Description: Total number of requests

\- Labels: `endpoint`, `platform`



```promql

\# Total requests

sum(bufferiq\_requests\_total)



\# Requests by endpoint

sum by (endpoint) (bufferiq\_requests\_total)



\# Requests by platform

sum by (platform) (bufferiq\_requests\_total)

```



\*\*`bufferiq\_request\_duration\_seconds`\*\* (Histogram)

\- Description: Request duration in seconds

\- Labels: `endpoint`

\- Buckets: 0.01, 0.025, 0.05, 0.075, 0.1, 0.25, 0.5, 1.0



```promql

\# Average latency

rate(bufferiq\_request\_duration\_seconds\_sum\[5m]) / 

rate(bufferiq\_request\_duration\_seconds\_count\[5m])



\# P95 latency

histogram\_quantile(0.95, 

&#x20; rate(bufferiq\_request\_duration\_seconds\_bucket\[5m]))



\# P99 latency

histogram\_quantile(0.99, 

&#x20; rate(bufferiq\_request\_duration\_seconds\_bucket\[5m]))

```



\#### Error Metrics



\*\*`bufferiq\_errors\_total`\*\* (Counter)

\- Description: Total number of errors

\- Labels: `endpoint`, `error\_type`



```promql

\# Error rate

rate(bufferiq\_errors\_total\[5m])



\# Error rate by type

sum by (error\_type) (rate(bufferiq\_errors\_total\[5m]))

```



\#### Cache Metrics



\*\*`bufferiq\_cache\_hits\_total`\*\* (Counter)

\- Description: Total cache hits



\*\*`bufferiq\_cache\_misses\_total`\*\* (Counter)

\- Description: Total cache misses



```promql

\# Cache hit rate

sum(rate(bufferiq\_cache\_hits\_total\[5m])) / 

(sum(rate(bufferiq\_cache\_hits\_total\[5m])) + 

&#x20;sum(rate(bufferiq\_cache\_misses\_total\[5m])))

```



\## Logging



\### Log Format



Structured JSON logging:



```json

{

&#x20; "timestamp": "2026-04-27T10:30:00.123Z",

&#x20; "level": "INFO",

&#x20; "logger": "bufferiq.api.routers.prediction",

&#x20; "message": "Request completed",

&#x20; "request\_id": "550e8400-e29b-41d4-a716-446655440000",

&#x20; "method": "POST",

&#x20; "path": "/api/v1/predict",

&#x20; "status\_code": 200,

&#x20; "duration\_ms": 45.2,

&#x20; "platform": "linkedin",

&#x20; "model\_version": "ensemble\_v1.0.0"

}

```



\### Log Levels



\- \*\*DEBUG\*\*: Detailed diagnostic information

\- \*\*INFO\*\*: General informational messages

\- \*\*WARNING\*\*: Warning messages

\- \*\*ERROR\*\*: Error messages

\- \*\*CRITICAL\*\*: Critical errors



\### Log Aggregation



\#### ELK Stack



```yaml

\# filebeat.yml

filebeat.inputs:

\- type: log

&#x20; enabled: true

&#x20; paths:

&#x20;   - /var/log/bufferiq/\*.log

&#x20; json.keys\_under\_root: true

&#x20; json.add\_error\_key: true



output.elasticsearch:

&#x20; hosts: \["elasticsearch:9200"]

```



\#### CloudWatch Logs



```python

import watchtower

import logging



logger = logging.getLogger()

logger.addHandler(watchtower.CloudWatchLogHandler(

&#x20;   log\_group='bufferiq-api',

&#x20;   stream\_name='production'

))

```



\## Alerting



\### Prometheus Alerting Rules



```yaml

\# alerts.yml

groups:

\- name: bufferiq\_api

&#x20; interval: 30s

&#x20; rules:

&#x20; 

&#x20; # High error rate

&#x20; - alert: HighErrorRate

&#x20;   expr: |

&#x20;     rate(bufferiq\_errors\_total\[5m]) > 0.05

&#x20;   for: 5m

&#x20;   labels:

&#x20;     severity: warning

&#x20;   annotations:

&#x20;     summary: "High error rate detected"

&#x20;     description: "Error rate is {{ $value }} errors/second"

&#x20; 

&#x20; # High latency

&#x20; - alert: HighLatency

&#x20;   expr: |

&#x20;     histogram\_quantile(0.95, 

&#x20;       rate(bufferiq\_request\_duration\_seconds\_bucket\[5m])) > 0.5

&#x20;   for: 5m

&#x20;   labels:

&#x20;     severity: warning

&#x20;   annotations:

&#x20;     summary: "High API latency"

&#x20;     description: "P95 latency is {{ $value }}s"

&#x20; 

&#x20; # Low cache hit rate

&#x20; - alert: LowCacheHitRate

&#x20;   expr: |

&#x20;     sum(rate(bufferiq\_cache\_hits\_total\[5m])) / 

&#x20;     (sum(rate(bufferiq\_cache\_hits\_total\[5m])) + 

&#x20;      sum(rate(bufferiq\_cache\_misses\_total\[5m]))) < 0.5

&#x20;   for: 10m

&#x20;   labels:

&#x20;     severity: info

&#x20;   annotations:

&#x20;     summary: "Low cache hit rate"

&#x20;     description: "Cache hit rate is {{ $value }}"

&#x20; 

&#x20; # API down

&#x20; - alert: APIDown

&#x20;   expr: up{job="bufferiq-api"} == 0

&#x20;   for: 1m

&#x20;   labels:

&#x20;     severity: critical

&#x20;   annotations:

&#x20;     summary: "BufferIQ API is down"

&#x20;     description: "API has been down for more than 1 minute"

```



\### Alertmanager Configuration



```yaml

\# alertmanager.yml

global:

&#x20; slack\_api\_url: 'YOUR\_SLACK\_WEBHOOK\_URL'



route:

&#x20; group\_by: \['alertname', 'severity']

&#x20; group\_wait: 30s

&#x20; group\_interval: 5m

&#x20; repeat\_interval: 4h

&#x20; receiver: 'slack-notifications'



receivers:

\- name: 'slack-notifications'

&#x20; slack\_configs:

&#x20; - channel: '#bufferiq-alerts'

&#x20;   title: '{{ .GroupLabels.alertname }}'

&#x20;   text: '{{ range .Alerts }}{{ .Annotations.description }}{{ end }}'

```



\## Performance Tuning



\### API Configuration



\*\*Worker Processes:\*\*

```yaml

\# Production

workers: 4  # Number of CPU cores



\# Development

workers: 1

```



\*\*Timeout Settings:\*\*

```python

\# uvicorn settings

timeout\_keep\_alive: 5

timeout\_notify: 30

```



\### Caching



\*\*Redis Configuration:\*\*

```yaml

cache:

&#x20; ttl: 7200  # 2 hours

&#x20; max\_size: 10000

```



\*\*Cache Key Strategy:\*\*

\- Include content hash

\- Include platform

\- Include model version



\### Model Loading



\*\*Lazy Loading:\*\*

```python

\# Load models on first request

warmup\_on\_startup: false



\# Load all models at startup (recommended)

warmup\_on\_startup: true

```



\*\*Model Cache Size:\*\*

```python

\# Keep 3 models in memory

cache\_size: 3



\# Keep 5 models (higher memory usage)

cache\_size: 5

```



\### Database Connections



\*\*Connection Pool:\*\*

```python

pool\_size: 10

max\_overflow: 20

pool\_timeout: 30

```



\## Dashboards



\### Grafana Dashboard



```json

{

&#x20; "dashboard": {

&#x20;   "title": "BufferIQ API",

&#x20;   "panels": \[

&#x20;     {

&#x20;       "title": "Request Rate",

&#x20;       "targets": \[

&#x20;         {

&#x20;           "expr": "sum(rate(bufferiq\_requests\_total\[5m]))"

&#x20;         }

&#x20;       ]

&#x20;     },

&#x20;     {

&#x20;       "title": "P95 Latency",

&#x20;       "targets": \[

&#x20;         {

&#x20;           "expr": "histogram\_quantile(0.95, rate(bufferiq\_request\_duration\_seconds\_bucket\[5m]))"

&#x20;         }

&#x20;       ]

&#x20;     },

&#x20;     {

&#x20;       "title": "Error Rate",

&#x20;       "targets": \[

&#x20;         {

&#x20;           "expr": "sum(rate(bufferiq\_errors\_total\[5m]))"

&#x20;         }

&#x20;       ]

&#x20;     },

&#x20;     {

&#x20;       "title": "Cache Hit Rate",

&#x20;       "targets": \[

&#x20;         {

&#x20;           "expr": "sum(rate(bufferiq\_cache\_hits\_total\[5m])) / (sum(rate(bufferiq\_cache\_hits\_total\[5m])) + sum(rate(bufferiq\_cache\_misses\_total\[5m])))"

&#x20;         }

&#x20;       ]

&#x20;     }

&#x20;   ]

&#x20; }

}

```



\### Key Performance Indicators (KPIs)



Monitor these metrics:



1\. \*\*Availability\*\*: 99.9%+ uptime

2\. \*\*Latency\*\*: P95 < 100ms

3\. \*\*Error Rate\*\*: < 0.1%

4\. \*\*Throughput\*\*: 100+ req/s

5\. \*\*Cache Hit Rate\*\*: > 70%



\## Troubleshooting



\### High CPU Usage



\*\*Check:\*\*

```bash

top

htop

```



\*\*Solutions:\*\*

\- Scale horizontally

\- Optimize model inference

\- Enable caching



\### High Memory Usage



\*\*Check:\*\*

```bash

free -h

docker stats

```



\*\*Solutions:\*\*

\- Reduce model cache size

\- Limit worker processes

\- Use smaller models



\### Slow Queries



\*\*Check logs:\*\*

```bash

grep "duration\_ms" logs/api.log | sort -t: -k6 -rn | head -20

```



\*\*Solutions:\*\*

\- Enable query caching

\- Optimize feature extraction

\- Profile slow endpoints



