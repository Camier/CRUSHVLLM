# Crush + vLLM Monitoring Stack

Production-ready monitoring solution for Crush applications with vLLM inference backend.

## Features

- **Real-time Metrics**: GPU utilization, memory, temperature, and performance
- **Performance Monitoring**: Token throughput, request latency, cache efficiency
- **Intelligent Alerting**: Multi-tier alerts with configurable thresholds
- **Visual Dashboards**: Comprehensive Grafana dashboards for all metrics
- **Auto-scaling Ready**: Metrics designed for auto-scaling decisions

## Architecture

```
┌─────────────────┐    ┌──────────────┐    ┌─────────────────┐
│   vLLM Server   │───▶│  Prometheus  │───▶│    Grafana      │
│   (port 8000)  │    │  (port 9090) │    │  (port 3000)    │
└─────────────────┘    └──────────────┘    └─────────────────┘
                              │
┌─────────────────┐           │            ┌─────────────────┐
│  DCGM Exporter  │───────────┴────────────│  Alertmanager   │
│   (port 9400)  │                        │  (port 9093)    │
└─────────────────┘                        └─────────────────┘
                                                    │
┌─────────────────┐                                 ▼
│  Node Exporter  │                        ┌─────────────────┐
│   (port 9100)  │                        │  Notifications  │
└─────────────────┘                        │  (Email/Slack)  │
                                          └─────────────────┘
```

## Quick Start

### Prerequisites

- Docker & Docker Compose
- NVIDIA Docker runtime (for GPU metrics)
- vLLM server exposing metrics on port 8000

### Installation

```bash
# Clone and setup
cd monitoring/
chmod +x setup.sh
./setup.sh
```

### Manual Setup

```bash
# Start all services
docker-compose up -d

# Check status
docker-compose ps

# View logs
docker-compose logs -f prometheus
```

## Configuration

### vLLM Integration

Ensure your vLLM server exposes metrics:

```python
# In your vLLM startup
import vllm
engine = vllm.LLMEngine.from_engine_args(
    engine_args,
    enable_metrics=True,
    metrics_port=8000
)
```

### GPU Metrics (Alternative to Docker)

If not using Docker DCGM exporter:

```bash
# Install DCGM
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/datacenter-gpu-manager_3.1.8_amd64.deb
sudo dpkg -i datacenter-gpu-manager_3.1.8_amd64.deb

# Start DCGM exporter
docker run -d --gpus all --rm -p 9400:9400 nvcr.io/nvidia/k8s/dcgm-exporter:3.1.8-3.1.5-ubuntu20.04
```

## Dashboards

### vLLM Performance Dashboard

Access: http://localhost:3000/d/vllm-crush-dashboard

**Panels:**
- Request & Token Throughput
- Latency Percentiles (P50, P95, P99)
- GPU Utilization & Memory
- Cache Hit Rates
- Batch Processing Efficiency
- Queue Status & Model Performance

### Key Metrics

| Metric | Description | Threshold |
|--------|-------------|-----------|
| `vllm:token_throughput_rate5m` | Tokens processed per second | >100 tokens/s |
| `vllm:request_latency_p99_5m` | 99th percentile latency | <10s |
| `vllm:gpu_utilization_avg` | Average GPU utilization | <95% |
| `vllm:gpu_memory_usage_percent` | GPU memory usage | <95% |
| `vllm:cache_hit_rate_5m` | Cache efficiency | >70% |

## Alerting

### Alert Levels

**Critical Alerts** (Immediate attention):
- Service down
- P99 latency >10s
- GPU memory >95%
- Hardware errors
- Failure rate >15%

**Warning Alerts** (Monitor closely):
- High latency trends
- GPU utilization >95%
- Low cache hit rates
- Queue depth issues

### Notification Setup

Edit `alertmanager.yml`:

```yaml
global:
  smtp_smarthost: 'your-smtp-server:587'
  smtp_from: 'alerts@yourcompany.com'

receivers:
  - name: 'critical-alerts'
    email_configs:
      - to: 'oncall@yourcompany.com'
```

### Slack Integration

```yaml
receivers:
  - name: 'slack-alerts'
    slack_configs:
      - api_url: 'YOUR_WEBHOOK_URL'
        channel: '#ml-ops'
        title: 'vLLM Alert'
```

## Performance Tuning

### Prometheus Optimization

```yaml
# prometheus.yml
global:
  scrape_interval: 15s     # Reduce for real-time needs
  evaluation_interval: 15s

storage:
  tsdb:
    retention.time: 30d      # Adjust based on storage
    retention.size: 50GB
```

### vLLM Metrics Configuration

```python
# Optimize metric collection
VLLM_METRICS_CONFIG = {
    "enable_gpu_memory_profiling": True,
    "enable_cache_metrics": True,
    "metrics_export_interval": 5,  # seconds
}
```

## Troubleshooting

### Common Issues

**GPU Metrics Missing:**
```bash
# Check NVIDIA runtime
docker info | grep nvidia

# Test DCGM exporter
curl http://localhost:9400/metrics | grep DCGM_FI_DEV_GPU_UTIL
```

**High Memory Usage:**
```bash
# Check Prometheus retention
docker exec prometheus-container promtool query series --match='{__name__=~".+"}'

# Adjust retention in prometheus.yml
storage:
  tsdb:
    retention.time: 7d  # Reduce retention
```

**vLLM Metrics Not Appearing:**
```bash
# Test vLLM metrics endpoint
curl http://localhost:8000/metrics

# Check Prometheus targets
curl http://localhost:9090/api/v1/targets
```

### Service Debugging

```bash
# Check all services
docker-compose ps

# View specific logs
docker-compose logs -f grafana
docker-compose logs -f prometheus

# Restart specific service
docker-compose restart alertmanager
```

## Customization

### Adding Custom Metrics

1. **Extend Prometheus Config:**
```yaml
# prometheus.yml
scrape_configs:
  - job_name: 'custom-app'
    static_configs:
      - targets: ['localhost:8080']
```

2. **Create Custom Alerts:**
```yaml
# alerts.yml
- alert: CustomMetricHigh
  expr: custom_metric > 100
  for: 5m
```

3. **Update Dashboard:**
- Import custom dashboard JSON
- Or modify existing `vllm-dashboard.json`

### Production Deployment

**Security Hardening:**
```yaml
# docker-compose.override.yml
services:
  grafana:
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=${GRAFANA_PASSWORD}
      - GF_USERS_ALLOW_SIGN_UP=false
      - GF_AUTH_ANONYMOUS_ENABLED=false
```

**Resource Limits:**
```yaml
services:
  prometheus:
    deploy:
      resources:
        limits:
          memory: 4G
          cpus: '2'
```

**Backup Configuration:**
```bash
# Backup Grafana dashboards
docker exec grafana-container grafana-cli admin export-dashboard > backup.json

# Backup Prometheus data
docker run --rm -v prometheus-data:/data -v $(pwd):/backup alpine tar czf /backup/prometheus-backup.tar.gz /data
```

## Scaling

### Multi-Instance Setup

```yaml
# prometheus.yml - for multiple vLLM instances
scrape_configs:
  - job_name: 'vllm'
    static_configs:
      - targets: 
        - 'vllm-1:8000'
        - 'vllm-2:8000'
        - 'vllm-3:8000'
```

### Federation Setup

```yaml
# For multiple Prometheus instances
scrape_configs:
  - job_name: 'federate'
    scrape_interval: 15s
    honor_labels: true
    metrics_path: '/federate'
    params:
      'match[]':
        - '{job=~"vllm.*"}'
    static_configs:
      - targets:
        - 'prometheus-region-1:9090'
        - 'prometheus-region-2:9090'
```

## Support

### Useful Commands

```bash
# Health check all services
./setup.sh

# Update configuration
docker-compose restart prometheus

# Scale specific service
docker-compose up -d --scale prometheus=2

# Check metrics availability
curl -s http://localhost:9090/api/v1/label/__name__/values | jq -r '.data[]' | grep vllm
```

### Log Locations

- Prometheus: `docker-compose logs prometheus`
- Grafana: `docker-compose logs grafana`
- Alertmanager: `docker-compose logs alertmanager`
- DCGM Exporter: `docker-compose logs dcgm-exporter`

---

**Version:** 1.0  
**Last Updated:** 2025-01-25  
**Compatibility:** vLLM 0.2+, Prometheus 2.45+, Grafana 10.0+