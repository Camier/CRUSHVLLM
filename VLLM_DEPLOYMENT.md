# Crush + vLLM Deployment Guide

<p align="center">
    <img width="450" alt="Crush + vLLM Integration" src="https://github.com/user-attachments/assets/adc1a6f4-b284-4603-836c-59038caa2e8b" />
</p>

<p align="center">Complete deployment guide for running Crush with local vLLM inference servers</p>

## Table of Contents

- [Overview](#overview)
- [Prerequisites](#prerequisites)
- [Quick Start](#quick-start)
- [Deployment Methods](#deployment-methods)
  - [Local Development Setup](#local-development-setup)
  - [Docker Deployment](#docker-deployment)
  - [Kubernetes Deployment](#kubernetes-deployment)
  - [Bare Metal Setup](#bare-metal-setup)
- [GPU Configuration](#gpu-configuration)
- [Model Configuration](#model-configuration)
- [Performance Tuning](#performance-tuning)
- [Monitoring & Observability](#monitoring--observability)
- [Troubleshooting](#troubleshooting)
- [Example Use Cases](#example-use-cases)
- [Advanced Configuration](#advanced-configuration)

## Overview

This guide covers deploying Crush with vLLM (vLLM: Easy, fast, and cheap LLM serving) for local, high-performance large language model inference. vLLM provides:

- **High throughput serving** with continuous batching and optimized attention kernels
- **OpenAI-compatible API** that works seamlessly with Crush
- **GPU acceleration** with support for NVIDIA GPUs and tensor parallelism
- **Memory efficiency** with PagedAttention and various quantization methods
- **Production ready** with monitoring, load balancing, and auto-scaling capabilities

### Architecture Overview

```
┌─────────────┐    ┌──────────────┐    ┌─────────────┐
│             │    │              │    │             │
│   Crush     │◄──►│    vLLM      │◄──►│   GPU(s)    │
│   Client    │    │   Server     │    │  Hardware   │
│             │    │              │    │             │
└─────────────┘    └──────────────┘    └─────────────┘
       │                   │                   │
       │            ┌─────────────┐           │
       └───────────►│ Monitoring  │◄──────────┘
                    │ & Metrics   │
                    └─────────────┘
```

## Prerequisites

### System Requirements

#### Minimum Requirements
- **OS**: Linux (Ubuntu 20.04+, CentOS 8+, RHEL 8+), macOS 12+, Windows 10+ (WSL2)
- **GPU**: NVIDIA GPU with 8GB+ VRAM (RTX 3070, A4000, or equivalent)
- **RAM**: 16GB system RAM
- **Storage**: 50GB+ free space for models and cache
- **Python**: 3.8+ (recommended: 3.11)
- **CUDA**: 11.8+ or 12.0+ (for NVIDIA GPUs)

#### Recommended Requirements
- **GPU**: NVIDIA A100, H100, RTX 4090, or Tesla V100 with 24GB+ VRAM
- **RAM**: 32GB+ system RAM
- **Storage**: NVMe SSD with 200GB+ free space
- **Network**: High-bandwidth internet for model downloads

### Software Dependencies

#### Core Dependencies
```bash
# NVIDIA drivers and CUDA toolkit
nvidia-driver-535+
cuda-toolkit-12-0+

# Python and package managers
python3.11+
pip3
conda (optional, recommended)

# Container runtime (for Docker deployment)
docker 24.0+
docker-compose 2.0+

# Monitoring tools
nvidia-ml-py3
prometheus-client
```

#### Development Tools
```bash
# Build tools
gcc/g++ 9+
cmake 3.18+
git

# Monitoring (optional)
prometheus
grafana
node-exporter
dcgm-exporter
```

### Hardware Compatibility

#### GPU Support Matrix

| GPU Model | VRAM | Recommended Models | Tensor Parallel |
|-----------|------|-------------------|-----------------|
| RTX 3070/3080 | 8-10GB | 7B models (quantized) | 1 |
| RTX 3090/4080 | 12-16GB | 7B-13B models | 1 |
| RTX 4090 | 24GB | 7B-70B models | 1-2 |
| A40/A100 40GB | 40GB | Up to 70B models | 1-4 |
| A100 80GB | 80GB | Up to 175B models | 1-8 |
| H100 | 80GB | Any model size | 1-8 |

## Quick Start

### 1. Install Crush
```bash
# Using Homebrew (macOS/Linux)
brew install charmbracelet/tap/crush

# Using npm
npm install -g @charmland/crush

# Using Go
go install github.com/charmbracelet/crush@latest

# Using package managers (Ubuntu/Debian)
sudo mkdir -p /etc/apt/keyrings
curl -fsSL https://repo.charm.sh/apt/gpg.key | sudo gpg --dearmor -o /etc/apt/keyrings/charm.gpg
echo "deb [signed-by=/etc/apt/keyrings/charm.gpg] https://repo.charm.sh/apt/ * *" | sudo tee /etc/apt/sources.list.d/charm.list
sudo apt update && sudo apt install crush
```

### 2. Quick vLLM Setup with Docker
```bash
# Clone the repository
git clone https://github.com/charmbracelet/crush.git
cd crush

# Set environment variables
export VLLM_MODEL="Qwen/Qwen2.5-Coder-32B-Instruct-AWQ"
export HF_TOKEN="your_huggingface_token"  # Optional, for gated models
export GPU_MEMORY=0.9

# Start vLLM server
docker-compose up vllm
```

### 3. Configure Crush
Create or update your `crush.json`:
```json
{
  "$schema": "https://charm.land/crush.json",
  "providers": {
    "vllm": {
      "name": "vLLM Local",
      "type": "openai", 
      "base_url": "http://localhost:8000/v1",
      "models": [
        {
          "id": "qwen2.5-coder-32b-instruct",
          "name": "Qwen 2.5 Coder 32B (Local)",
          "context_window": 32768,
          "default_max_tokens": 4096,
          "supports_tools": true
        }
      ]
    }
  }
}
```

### 4. Start Crush
```bash
crush
```

Select the vLLM provider when prompted, and you're ready to go!

## Deployment Methods

### Local Development Setup

#### Method 1: Native Installation

1. **Install vLLM**
```bash
# Create virtual environment
python -m venv ~/venvs/vllm
source ~/venvs/vllm/bin/activate

# Install vLLM with CUDA support
pip install vllm torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Verify installation
python -c "import vllm; print(vllm.__version__)"
```

2. **Download and configure model**
```bash
# Set Hugging Face cache directory (optional)
export HF_HOME=~/.cache/huggingface

# Pre-download model (optional)
python -c "
from transformers import AutoTokenizer, AutoModelForCausalLM
model_name = 'Qwen/Qwen2.5-Coder-32B-Instruct-AWQ'
tokenizer = AutoTokenizer.from_pretrained(model_name)
print(f'Downloaded {model_name}')
"
```

3. **Start vLLM server**
```bash
# Use the provided script
./scripts/start_vllm.sh

# Or manually
python -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen2.5-Coder-32B-Instruct-AWQ \
  --port 8000 \
  --gpu-memory-utilization 0.9 \
  --max-model-len 32768 \
  --quantization awq \
  --trust-remote-code \
  --enable-prefix-caching
```

#### Method 2: Conda Environment

1. **Setup Conda environment**
```bash
# Create environment
conda create -n vllm python=3.11 -y
conda activate vllm

# Install CUDA toolkit
conda install cudatoolkit=12.0 -c nvidia

# Install vLLM
pip install vllm

# Install monitoring dependencies
pip install prometheus-client nvidia-ml-py3
```

2. **Create startup script**
```bash
cat > start_vllm_conda.sh << 'EOF'
#!/bin/bash
set -e

# Activate conda environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate vllm

# Start vLLM server
python -m vllm.entrypoints.openai.api_server \
  --model ${VLLM_MODEL:-Qwen/Qwen2.5-Coder-32B-Instruct-AWQ} \
  --port ${VLLM_PORT:-8000} \
  --gpu-memory-utilization ${GPU_MEMORY:-0.9} \
  --max-model-len ${MAX_MODEL_LEN:-32768} \
  --quantization ${QUANTIZATION:-awq} \
  --trust-remote-code \
  --enable-prefix-caching \
  --max-num-seqs 256
EOF

chmod +x start_vllm_conda.sh
```

### Docker Deployment

#### Single Container Setup

1. **Basic Docker run**
```bash
# Pull vLLM image
docker pull vllm/vllm-openai:latest

# Run with GPU support
docker run -d \
  --name crush-vllm \
  --runtime=nvidia \
  -p 8000:8000 \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  -v ./models:/models \
  -e NVIDIA_VISIBLE_DEVICES=all \
  -e CUDA_VISIBLE_DEVICES=0 \
  vllm/vllm-openai:latest \
  --model Qwen/Qwen2.5-Coder-32B-Instruct-AWQ \
  --port 8000 \
  --gpu-memory-utilization 0.9 \
  --max-model-len 32768 \
  --quantization awq \
  --trust-remote-code \
  --enable-prefix-caching
```

2. **Custom Dockerfile for Crush**
```dockerfile
# Use the provided Dockerfile
FROM vllm/vllm-openai:latest AS vllm-base

# Add Crush binary
FROM golang:1.23-alpine AS crush-builder
WORKDIR /build
COPY . .
RUN go build -o crush .

FROM vllm-base
COPY --from=crush-builder /build/crush /usr/local/bin/crush
COPY configs/ /app/configs/

# Startup script
COPY <<'EOF' /start.sh
#!/bin/bash
# Start vLLM in background
python -m vllm.entrypoints.openai.api_server \
  --model $VLLM_MODEL \
  --port 8000 \
  --gpu-memory-utilization $GPU_MEMORY \
  --max-model-len $MAX_MODEL_LEN \
  --quantization $QUANTIZATION \
  --trust-remote-code \
  --enable-prefix-caching &

# Wait for vLLM to start
sleep 30

# Start Crush
crush --config /app/configs/vllm_setup.json
EOF

RUN chmod +x /start.sh
ENTRYPOINT ["/start.sh"]
```

#### Docker Compose Setup

The provided `docker-compose.yml` includes:
- vLLM server with GPU support
- Prometheus monitoring
- Grafana dashboards
- Health checks and auto-restart

```bash
# Start full stack
docker-compose up -d

# Start only vLLM
docker-compose up -d vllm

# View logs
docker-compose logs -f vllm

# Scale for multiple models (if you have multiple GPUs)
docker-compose up --scale vllm=2
```

#### Multi-Model Docker Setup

```yaml
# docker-compose.multi-model.yml
version: '3.8'

services:
  vllm-coder:
    image: vllm/vllm-openai:latest
    container_name: crush-vllm-coder
    runtime: nvidia
    ports:
      - "8000:8000"
    environment:
      - NVIDIA_VISIBLE_DEVICES=0
    command: >
      --model Qwen/Qwen2.5-Coder-32B-Instruct-AWQ
      --port 8000
      --gpu-memory-utilization 0.9
      --quantization awq
      --trust-remote-code
      
  vllm-chat:
    image: vllm/vllm-openai:latest
    container_name: crush-vllm-chat
    runtime: nvidia
    ports:
      - "8001:8000"
    environment:
      - NVIDIA_VISIBLE_DEVICES=1
    command: >
      --model mistralai/Mistral-7B-Instruct-v0.2
      --port 8000
      --gpu-memory-utilization 0.85
      --trust-remote-code

  nginx-lb:
    image: nginx:alpine
    ports:
      - "8080:80"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf
    depends_on:
      - vllm-coder
      - vllm-chat
```

### Kubernetes Deployment

#### Prerequisites
```bash
# Install kubectl
curl -LO "https://dl.k8s.io/release/$(curl -L -s https://dl.k8s.io/release/stable.txt)/bin/linux/amd64/kubectl"
sudo install -o root -g root -m 0755 kubectl /usr/local/bin/kubectl

# Install NVIDIA GPU Operator
kubectl apply -f https://raw.githubusercontent.com/NVIDIA/gpu-operator/v23.9.1/deployments/gpu-operator/values.yaml

# Verify GPU nodes
kubectl get nodes -o custom-columns="NAME:.metadata.name,GPU:.status.allocatable.nvidia\.com/gpu"
```

#### Basic Deployment

1. **Create namespace and configmap**
```yaml
# k8s/namespace.yaml
apiVersion: v1
kind: Namespace
metadata:
  name: crush-vllm
---
apiVersion: v1
kind: ConfigMap
metadata:
  name: vllm-config
  namespace: crush-vllm
data:
  model: "Qwen/Qwen2.5-Coder-32B-Instruct-AWQ"
  port: "8000"
  gpu_memory: "0.9"
  max_model_len: "32768"
  quantization: "awq"
```

2. **Create deployment**
```yaml
# k8s/deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: vllm-server
  namespace: crush-vllm
spec:
  replicas: 1
  selector:
    matchLabels:
      app: vllm-server
  template:
    metadata:
      labels:
        app: vllm-server
    spec:
      containers:
      - name: vllm
        image: vllm/vllm-openai:latest
        ports:
        - containerPort: 8000
        command:
        - "python"
        - "-m" 
        - "vllm.entrypoints.openai.api_server"
        args:
        - "--model"
        - "$(MODEL_NAME)"
        - "--port"
        - "8000"
        - "--gpu-memory-utilization"
        - "$(GPU_MEMORY)"
        - "--max-model-len"
        - "$(MAX_MODEL_LEN)"
        - "--quantization"
        - "$(QUANTIZATION)"
        - "--trust-remote-code"
        - "--enable-prefix-caching"
        env:
        - name: MODEL_NAME
          valueFrom:
            configMapKeyRef:
              name: vllm-config
              key: model
        - name: GPU_MEMORY
          valueFrom:
            configMapKeyRef:
              name: vllm-config
              key: gpu_memory
        - name: MAX_MODEL_LEN
          valueFrom:
            configMapKeyRef:
              name: vllm-config
              key: max_model_len
        - name: QUANTIZATION
          valueFrom:
            configMapKeyRef:
              name: vllm-config
              key: quantization
        resources:
          limits:
            nvidia.com/gpu: 1
            memory: "32Gi"
          requests:
            nvidia.com/gpu: 1
            memory: "16Gi"
        volumeMounts:
        - name: hf-cache
          mountPath: /root/.cache/huggingface
        - name: models
          mountPath: /models
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 180
          periodSeconds: 30
        readinessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 60
          periodSeconds: 10
      volumes:
      - name: hf-cache
        hostPath:
          path: /opt/hf-cache
      - name: models
        hostPath:
          path: /opt/models
      nodeSelector:
        accelerator: nvidia-tesla-v100  # Adjust for your GPU type
```

3. **Create service**
```yaml
# k8s/service.yaml
apiVersion: v1
kind: Service
metadata:
  name: vllm-service
  namespace: crush-vllm
spec:
  selector:
    app: vllm-server
  ports:
  - port: 8000
    targetPort: 8000
  type: LoadBalancer
```

4. **Create ingress (optional)**
```yaml
# k8s/ingress.yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: vllm-ingress
  namespace: crush-vllm
  annotations:
    nginx.ingress.kubernetes.io/rewrite-target: /
    nginx.ingress.kubernetes.io/proxy-read-timeout: "300"
    nginx.ingress.kubernetes.io/proxy-connect-timeout: "300"
spec:
  rules:
  - host: vllm.local
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: vllm-service
            port:
              number: 8000
```

5. **Deploy to cluster**
```bash
kubectl apply -f k8s/
kubectl get pods -n crush-vllm
kubectl get svc -n crush-vllm
```

#### Advanced Kubernetes Features

**Horizontal Pod Autoscaling**
```yaml
# k8s/hpa.yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: vllm-hpa
  namespace: crush-vllm
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: vllm-server
  minReplicas: 1
  maxReplicas: 5
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 80
```

**PersistentVolumeClaim for model storage**
```yaml
# k8s/pvc.yaml
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: models-pvc
  namespace: crush-vllm
spec:
  accessModes:
    - ReadWriteMany
  resources:
    requests:
      storage: 200Gi
  storageClassName: fast-ssd
```

### Bare Metal Setup

For production bare metal deployments with maximum performance.

#### System Preparation

1. **Operating System Setup**
```bash
# Ubuntu 22.04 LTS recommended
sudo apt update && sudo apt upgrade -y

# Install essential packages
sudo apt install -y \
    build-essential \
    cmake \
    git \
    curl \
    wget \
    python3.11 \
    python3.11-dev \
    python3.11-venv \
    nvidia-driver-535 \
    nvidia-cuda-toolkit

# Verify NVIDIA setup
nvidia-smi
nvcc --version
```

2. **Performance Optimizations**
```bash
# CPU governor for performance
echo performance | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor

# Increase network buffer sizes
echo 'net.core.rmem_default = 262144' | sudo tee -a /etc/sysctl.conf
echo 'net.core.rmem_max = 268435456' | sudo tee -a /etc/sysctl.conf
echo 'net.core.wmem_default = 262144' | sudo tee -a /etc/sysctl.conf
echo 'net.core.wmem_max = 268435456' | sudo tee -a /etc/sysctl.conf

# Apply changes
sudo sysctl -p

# Set GPU persistence mode
sudo nvidia-smi -pm 1

# Set GPU power limits (adjust for your card)
sudo nvidia-smi -pl 400  # For RTX 4090, adjust as needed
```

3. **Python Environment Setup**
```bash
# Create dedicated user for vLLM
sudo useradd -m -s /bin/bash vllm
sudo usermod -aG docker vllm

# Setup Python environment
sudo -u vllm python3.11 -m venv /home/vllm/venv
sudo -u vllm /home/vllm/venv/bin/pip install --upgrade pip

# Install vLLM with optimizations
sudo -u vllm /home/vllm/venv/bin/pip install \
    vllm \
    torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121 \
    transformers \
    accelerate \
    bitsandbytes \
    prometheus-client \
    nvidia-ml-py3
```

#### Production Service Setup

1. **Create systemd service**
```bash
sudo tee /etc/systemd/system/vllm.service << 'EOF'
[Unit]
Description=vLLM OpenAI API Server
After=network.target
Wants=network.target

[Service]
Type=simple
User=vllm
Group=vllm
WorkingDirectory=/home/vllm
Environment=PATH=/home/vllm/venv/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
Environment=CUDA_VISIBLE_DEVICES=0
Environment=NVIDIA_VISIBLE_DEVICES=all
Environment=HF_HOME=/home/vllm/.cache/huggingface
Environment=VLLM_USE_TRITON=1
Environment=VLLM_WORKER_MULTIPROC_METHOD=spawn

ExecStart=/home/vllm/venv/bin/python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen2.5-Coder-32B-Instruct-AWQ \
    --port 8000 \
    --host 0.0.0.0 \
    --gpu-memory-utilization 0.9 \
    --max-model-len 32768 \
    --tensor-parallel-size 1 \
    --quantization awq \
    --trust-remote-code \
    --enable-prefix-caching \
    --disable-log-requests \
    --max-num-seqs 256 \
    --max-num-batched-tokens 8192 \
    --enforce-eager false \
    --enable-chunked-prefill

Restart=always
RestartSec=10
LimitNOFILE=65536
LimitNPROC=65536

[Install]
WantedBy=multi-user.target
EOF
```

2. **Configure service**
```bash
# Reload systemd
sudo systemctl daemon-reload

# Enable and start service
sudo systemctl enable vllm.service
sudo systemctl start vllm.service

# Check status
sudo systemctl status vllm.service

# View logs
sudo journalctl -u vllm.service -f
```

3. **Setup log rotation**
```bash
sudo tee /etc/logrotate.d/vllm << 'EOF'
/var/log/vllm/*.log {
    daily
    missingok
    rotate 30
    compress
    delaycompress
    notifempty
    create 0640 vllm vllm
    postrotate
        systemctl reload vllm || true
    endscript
}
EOF
```

#### Load Balancer Setup (HAProxy)

For multiple vLLM instances:

```bash
# Install HAProxy
sudo apt install -y haproxy

# Configure HAProxy
sudo tee /etc/haproxy/haproxy.cfg << 'EOF'
global
    daemon
    chroot /var/lib/haproxy
    stats socket /run/haproxy/admin.sock mode 660 level admin
    stats timeout 30s
    user haproxy
    group haproxy

defaults
    mode http
    timeout connect 5000ms
    timeout client 50000ms
    timeout server 50000ms
    option httplog
    option dontlognull
    option log-health-checks

frontend vllm-frontend
    bind *:8080
    default_backend vllm-servers

backend vllm-servers
    balance roundrobin
    option httpchk GET /health
    http-check expect status 200
    
    server vllm1 localhost:8000 check inter 10s fall 3 rise 2
    server vllm2 localhost:8001 check inter 10s fall 3 rise 2
    server vllm3 localhost:8002 check inter 10s fall 3 rise 2

listen stats
    bind *:8404
    stats enable
    stats uri /stats
    stats refresh 30s
    stats admin if TRUE
EOF

# Start HAProxy
sudo systemctl enable haproxy
sudo systemctl start haproxy
```

## GPU Configuration

### Single GPU Setup

#### NVIDIA RTX Series (Consumer GPUs)

**RTX 3070/3080 (8-10GB VRAM)**
```bash
export VLLM_MODEL="microsoft/DialoGPT-medium"  # Smaller models
export GPU_MEMORY="0.85"
export MAX_MODEL_LEN="4096"
export QUANTIZATION="gptq"  # Aggressive quantization
export TENSOR_PARALLEL="1"
```

**RTX 3090/4080 (12-16GB VRAM)**
```bash
export VLLM_MODEL="codellama/CodeLlama-7b-Instruct-hf"
export GPU_MEMORY="0.9"
export MAX_MODEL_LEN="8192"
export QUANTIZATION="awq"
export TENSOR_PARALLEL="1"
```

**RTX 4090 (24GB VRAM)**
```bash
export VLLM_MODEL="Qwen/Qwen2.5-Coder-32B-Instruct-AWQ"
export GPU_MEMORY="0.9"
export MAX_MODEL_LEN="32768"
export QUANTIZATION="awq"
export TENSOR_PARALLEL="1"
```

#### NVIDIA Data Center GPUs

**Tesla V100 (16GB/32GB)**
```bash
export VLLM_MODEL="deepseek-ai/deepseek-coder-33b-instruct"
export GPU_MEMORY="0.9"
export MAX_MODEL_LEN="16384"
export QUANTIZATION="gptq"
export TENSOR_PARALLEL="1"
```

**A100 40GB/80GB**
```bash
export VLLM_MODEL="meta-llama/Llama-2-70b-chat-hf"
export GPU_MEMORY="0.95"
export MAX_MODEL_LEN="4096"
export QUANTIZATION="none"  # Full precision
export TENSOR_PARALLEL="2"  # Use 2 GPUs for 70B model
```

### Multi-GPU Setup

#### Tensor Parallelism Configuration

**2x RTX 4090 Setup**
```bash
export VLLM_MODEL="meta-llama/Llama-2-70b-chat-hf"
export GPU_MEMORY="0.9"
export MAX_MODEL_LEN="4096"
export TENSOR_PARALLEL="2"
export CUDA_VISIBLE_DEVICES="0,1"

# Start with tensor parallelism
python -m vllm.entrypoints.openai.api_server \
    --model $VLLM_MODEL \
    --tensor-parallel-size $TENSOR_PARALLEL \
    --gpu-memory-utilization $GPU_MEMORY \
    --max-model-len $MAX_MODEL_LEN \
    --trust-remote-code \
    --enable-prefix-caching
```

**4x A100 Setup**
```bash
export VLLM_MODEL="meta-llama/Llama-2-70b-chat-hf"
export GPU_MEMORY="0.95"
export MAX_MODEL_LEN="8192"
export TENSOR_PARALLEL="4"
export CUDA_VISIBLE_DEVICES="0,1,2,3"

# Advanced multi-GPU configuration
python -m vllm.entrypoints.openai.api_server \
    --model $VLLM_MODEL \
    --tensor-parallel-size $TENSOR_PARALLEL \
    --gpu-memory-utilization $GPU_MEMORY \
    --max-model-len $MAX_MODEL_LEN \
    --trust-remote-code \
    --enable-prefix-caching \
    --pipeline-parallel-size 1 \
    --distributed-executor-backend nccl
```

#### Pipeline Parallelism

For extremely large models:
```bash
# 8x A100 with both tensor and pipeline parallelism
python -m vllm.entrypoints.openai.api_server \
    --model meta-llama/Llama-2-70b-chat-hf \
    --tensor-parallel-size 4 \
    --pipeline-parallel-size 2 \
    --gpu-memory-utilization 0.95 \
    --max-model-len 8192 \
    --trust-remote-code \
    --distributed-executor-backend nccl
```

### Memory Management

#### GPU Memory Optimization

**Memory Mapping Strategy**
```python
# Custom memory configuration
import os

# Set memory fractions per GPU
os.environ['CUDA_MEM_FRACTION_0'] = '0.9'  # GPU 0: 90%
os.environ['CUDA_MEM_FRACTION_1'] = '0.8'  # GPU 1: 80%

# Enable memory pooling
os.environ['VLLM_USE_MEMORY_POOL'] = '1'
os.environ['VLLM_MEMORY_POOL_SIZE'] = '0.8'
```

**Dynamic Memory Allocation**
```bash
# Enable dynamic memory growth
export CUDA_MEMORY_GROWTH=true
export VLLM_MEMORY_POOL_INIT_SIZE=0.1
export VLLM_MEMORY_POOL_MAX_SIZE=0.9
```

## Model Configuration

### Supported Model Architectures

#### Code-Specialized Models

**Qwen2.5-Coder Series** (Recommended for Crush)
```json
{
  "providers": {
    "vllm": {
      "models": [
        {
          "id": "qwen2.5-coder-7b",
          "name": "Qwen 2.5 Coder 7B",
          "model_path": "Qwen/Qwen2.5-Coder-7B-Instruct",
          "context_window": 32768,
          "supports_tools": true,
          "quantization": "awq"
        },
        {
          "id": "qwen2.5-coder-32b", 
          "name": "Qwen 2.5 Coder 32B",
          "model_path": "Qwen/Qwen2.5-Coder-32B-Instruct-AWQ",
          "context_window": 32768,
          "supports_tools": true,
          "quantization": "awq"
        }
      ]
    }
  }
}
```

**DeepSeek Coder Series**
```json
{
  "models": [
    {
      "id": "deepseek-coder-7b",
      "name": "DeepSeek Coder 7B",
      "model_path": "deepseek-ai/deepseek-coder-7b-instruct-v1.5",
      "context_window": 16384,
      "supports_tools": true
    },
    {
      "id": "deepseek-coder-33b",
      "name": "DeepSeek Coder 33B", 
      "model_path": "deepseek-ai/deepseek-coder-33b-instruct",
      "context_window": 16384,
      "supports_tools": true,
      "quantization": "gptq"
    }
  ]
}
```

**Code Llama Series**
```json
{
  "models": [
    {
      "id": "codellama-7b",
      "name": "Code Llama 7B",
      "model_path": "codellama/CodeLlama-7b-Instruct-hf",
      "context_window": 16384,
      "supports_tools": false
    },
    {
      "id": "codellama-34b",
      "name": "Code Llama 34B",
      "model_path": "codellama/CodeLlama-34b-Instruct-hf", 
      "context_window": 16384,
      "supports_tools": false,
      "quantization": "gptq"
    }
  ]
}
```

#### General Purpose Models

**Llama 2 Series**
```json
{
  "models": [
    {
      "id": "llama2-7b-chat",
      "name": "Llama 2 7B Chat",
      "model_path": "meta-llama/Llama-2-7b-chat-hf",
      "context_window": 4096,
      "supports_tools": false
    },
    {
      "id": "llama2-70b-chat",
      "name": "Llama 2 70B Chat",
      "model_path": "meta-llama/Llama-2-70b-chat-hf",
      "context_window": 4096,
      "supports_tools": false,
      "tensor_parallel": 2
    }
  ]
}
```

**Mistral Series**
```json
{
  "models": [
    {
      "id": "mistral-7b-instruct",
      "name": "Mistral 7B Instruct",
      "model_path": "mistralai/Mistral-7B-Instruct-v0.2",
      "context_window": 32768,
      "supports_tools": false
    }
  ]
}
```

### Quantization Methods

#### AWQ (Activation-aware Weight Quantization)
- **Best for**: Code generation, tool use
- **Precision**: 4-bit weights
- **Memory savings**: ~4x reduction
- **Performance**: Minimal quality loss

```bash
# AWQ quantized models
python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen2.5-Coder-32B-Instruct-AWQ \
    --quantization awq \
    --gpu-memory-utilization 0.9
```

#### GPTQ (Generative Pre-trained Transformer Quantization)
- **Best for**: General purpose models
- **Precision**: 4-bit weights
- **Memory savings**: ~4x reduction
- **Performance**: Good balance

```bash
# GPTQ quantized models
python -m vllm.entrypoints.openai.api_server \
    --model TheBloke/deepseek-coder-33b-instruct-GPTQ \
    --quantization gptq \
    --gpu-memory-utilization 0.9
```

#### FP8 Quantization
- **Best for**: Data center GPUs (H100)
- **Precision**: 8-bit floating point
- **Memory savings**: ~2x reduction
- **Performance**: Near full precision

```bash
# FP8 quantization (H100 only)
python -m vllm.entrypoints.openai.api_server \
    --model meta-llama/Llama-2-70b-chat-hf \
    --quantization fp8 \
    --gpu-memory-utilization 0.95
```

### Custom Model Integration

#### Adding Custom Fine-tuned Models

1. **Prepare model directory**
```bash
# Model directory structure
/models/my-custom-model/
├── config.json
├── pytorch_model.bin (or model.safetensors)
├── tokenizer.json
├── tokenizer_config.json
└── special_tokens_map.json
```

2. **Update Crush configuration**
```json
{
  "providers": {
    "vllm": {
      "models": [
        {
          "id": "custom-coder",
          "name": "My Custom Coder",
          "model_path": "/models/my-custom-model",
          "context_window": 16384,
          "default_max_tokens": 2048,
          "supports_tools": true,
          "extra_config": {
            "trust_remote_code": true,
            "max_model_len": 16384,
            "gpu_memory_utilization": 0.9
          }
        }
      ]
    }
  }
}
```

3. **Start vLLM with custom model**
```bash
python -m vllm.entrypoints.openai.api_server \
    --model /models/my-custom-model \
    --trust-remote-code \
    --max-model-len 16384 \
    --gpu-memory-utilization 0.9
```

#### Model Conversion and Optimization

**Convert to vLLM format**
```python
# convert_model.py
from vllm import LLM

# Load and convert model
llm = LLM(
    model="path/to/your/model",
    trust_remote_code=True,
    quantization="awq",  # Optional
    tensor_parallel_size=1
)

# Save optimized model
llm.save_pretrained("path/to/optimized/model")
```

**Benchmark custom model**
```bash
# Benchmark script
python -c "
import time
from vllm import LLM, SamplingParams

llm = LLM(model='/models/my-custom-model')
prompts = ['def fibonacci(n):'] * 10

start = time.time()
outputs = llm.generate(prompts, SamplingParams(temperature=0.1, max_tokens=100))
end = time.time()

print(f'Generated {len(outputs)} completions in {end-start:.2f}s')
print(f'Throughput: {len(outputs)/(end-start):.2f} req/s')
"
```

## Performance Tuning

### vLLM Performance Parameters

#### Batching and Throughput

**Continuous Batching Configuration**
```bash
python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen2.5-Coder-32B-Instruct-AWQ \
    --max-num-seqs 256 \              # Maximum batch size
    --max-num-batched-tokens 8192 \   # Token budget per batch
    --max-paddings 256 \              # Padding tokens per batch
    --enable-prefix-caching \         # Cache common prefixes
    --gpu-memory-utilization 0.9
```

**Advanced Batching Parameters**
```bash
# Fine-tuned for high throughput
--max-num-seqs 512 \
--max-num-batched-tokens 16384 \
--max-paddings 512 \
--scheduler-delay-factor 0.0 \
--enable-chunked-prefill \
--max-num-on-the-fly-seq-groups 4
```

#### Memory and Compute Optimization

**CUDA Graph Optimization**
```bash
# Enable CUDA graphs for latency optimization
python -m vllm.entrypoints.openai.api_server \
    --model your-model \
    --enforce-eager false \           # Enable CUDA graphs
    --max-context-len-to-capture 8192 \
    --enable-prefix-caching
```

**Attention Kernel Selection**
```bash
# Use FlashAttention for better memory efficiency
export VLLM_USE_TRITON=1
export VLLM_FLASH_ATTN_CHUNK_SIZE=1024

python -m vllm.entrypoints.openai.api_server \
    --model your-model \
    --use-v2-block-manager \
    --block-size 32
```

#### Network and I/O Optimization

**Connection Pool Configuration**
```python
# vllm_config.py
import os

# Increase connection limits
os.environ['VLLM_MAX_CONNECTIONS'] = '1000'
os.environ['VLLM_CONNECTION_TIMEOUT'] = '300'
os.environ['VLLM_KEEP_ALIVE_TIMEOUT'] = '60'

# Enable request streaming
os.environ['VLLM_ENABLE_STREAMING'] = '1'
os.environ['VLLM_STREAM_CHUNK_SIZE'] = '1024'
```

### System-Level Optimizations

#### CPU and Memory

**CPU Affinity and Threading**
```bash
# Pin vLLM to specific CPU cores
taskset -c 0-15 python -m vllm.entrypoints.openai.api_server \
    --model your-model

# Set number of CPU threads
export OMP_NUM_THREADS=16
export MKL_NUM_THREADS=16
```

**Memory Management**
```bash
# Increase shared memory size
echo 'kernel.shmmax = 68719476736' | sudo tee -a /etc/sysctl.conf
echo 'kernel.shmall = 4294967296' | sudo tee -a /etc/sysctl.conf

# Set memory overcommit
echo 'vm.overcommit_memory = 1' | sudo tee -a /etc/sysctl.conf
sudo sysctl -p
```

#### GPU Performance Tuning

**GPU Clock and Power Management**
```bash
# Set maximum performance mode
sudo nvidia-smi -pm 1

# Set GPU clocks to maximum
sudo nvidia-smi -ac 1215,1410  # Memory,Graphics (adjust for your GPU)

# Set power limit to maximum
sudo nvidia-smi -pl 400  # Watts (adjust for your GPU)

# Enable persistence mode
sudo nvidia-smi -pm 1
```

**CUDA Environment Variables**
```bash
export CUDA_VISIBLE_DEVICES=0
export NVIDIA_VISIBLE_DEVICES=all
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_LAUNCH_BLOCKING=0
export CUDNN_BENCHMARK=1
```

#### Storage Optimization

**Model Storage Performance**
```bash
# Use SSD/NVMe for model storage
sudo mkdir -p /opt/models
sudo mount -t tmpfs -o size=100G tmpfs /opt/models  # RAM disk for ultimate speed

# Or use high-performance filesystem
sudo mkfs.ext4 -F /dev/nvme0n1p1
sudo mount -o noatime,nodiratime /dev/nvme0n1p1 /opt/models
```

**Cache Configuration**
```bash
# Optimize Hugging Face cache
export HF_HOME=/opt/hf-cache
export HF_DATASETS_CACHE=/opt/hf-cache/datasets
export TRANSFORMERS_CACHE=/opt/hf-cache/transformers

# Create optimized cache directory
sudo mkdir -p /opt/hf-cache
sudo chown -R $USER:$USER /opt/hf-cache
sudo chmod -R 755 /opt/hf-cache
```

### Benchmarking and Profiling

#### Performance Benchmarks

**Basic Throughput Test**
```python
# benchmark_throughput.py
import asyncio
import aiohttp
import time
import json
from concurrent.futures import ThreadPoolExecutor

async def send_request(session, prompt, request_id):
    data = {
        "model": "qwen2.5-coder-32b-instruct",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 100,
        "temperature": 0.1
    }
    
    start_time = time.time()
    async with session.post("http://localhost:8000/v1/chat/completions", 
                          json=data) as response:
        result = await response.json()
        end_time = time.time()
        
        return {
            "request_id": request_id,
            "latency": end_time - start_time,
            "tokens": len(result.get("choices", [{}])[0].get("message", {}).get("content", "").split()),
            "success": response.status == 200
        }

async def benchmark_concurrent_requests(num_requests=100, concurrency=10):
    prompts = [
        "Write a Python function to calculate fibonacci numbers",
        "Create a REST API endpoint using FastAPI", 
        "Implement a binary search algorithm",
        "Write a database migration script",
        "Create a React component for user authentication"
    ]
    
    async with aiohttp.ClientSession() as session:
        semaphore = asyncio.Semaphore(concurrency)
        
        async def limited_request(request_id):
            async with semaphore:
                prompt = prompts[request_id % len(prompts)]
                return await send_request(session, prompt, request_id)
        
        start_time = time.time()
        results = await asyncio.gather(*[
            limited_request(i) for i in range(num_requests)
        ])
        end_time = time.time()
        
        successful_requests = [r for r in results if r["success"]]
        total_tokens = sum(r["tokens"] for r in successful_requests)
        avg_latency = sum(r["latency"] for r in successful_requests) / len(successful_requests)
        
        print(f"Total time: {end_time - start_time:.2f}s")
        print(f"Successful requests: {len(successful_requests)}/{num_requests}")
        print(f"Throughput: {len(successful_requests)/(end_time - start_time):.2f} req/s")
        print(f"Tokens/second: {total_tokens/(end_time - start_time):.2f}")
        print(f"Average latency: {avg_latency:.2f}s")

if __name__ == "__main__":
    asyncio.run(benchmark_concurrent_requests())
```

**Latency Analysis**
```python
# benchmark_latency.py
import requests
import time
import statistics

def benchmark_latency(num_requests=50):
    url = "http://localhost:8000/v1/chat/completions"
    
    latencies = []
    
    for i in range(num_requests):
        data = {
            "model": "qwen2.5-coder-32b-instruct",
            "messages": [{"role": "user", "content": "def hello_world():"}],
            "max_tokens": 50,
            "temperature": 0.0
        }
        
        start_time = time.time()
        response = requests.post(url, json=data)
        end_time = time.time()
        
        if response.status_code == 200:
            latencies.append(end_time - start_time)
        
        time.sleep(0.1)  # Small delay between requests
    
    print(f"Requests: {len(latencies)}")
    print(f"Average latency: {statistics.mean(latencies):.3f}s")
    print(f"Median latency: {statistics.median(latencies):.3f}s")
    print(f"95th percentile: {sorted(latencies)[int(0.95 * len(latencies))]:.3f}s")
    print(f"99th percentile: {sorted(latencies)[int(0.99 * len(latencies))]:.3f}s")

if __name__ == "__main__":
    benchmark_latency()
```

#### GPU Monitoring Scripts

**Real-time GPU Usage**
```python
# gpu_monitor.py
import time
import psutil
import pynvml

def monitor_gpu():
    pynvml.nvmlInit()
    device_count = pynvml.nvmlDeviceGetCount()
    
    print(f"Monitoring {device_count} GPU(s)...")
    
    while True:
        for i in range(device_count):
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            
            # GPU info
            name = pynvml.nvmlDeviceGetName(handle).decode()
            
            # Memory info
            mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            mem_used_gb = mem_info.used / 1024**3
            mem_total_gb = mem_info.total / 1024**3
            mem_percent = (mem_info.used / mem_info.total) * 100
            
            # Utilization
            util = pynvml.nvmlDeviceGetUtilizationRates(handle)
            gpu_util = util.gpu
            memory_util = util.memory
            
            # Temperature
            temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
            
            # Power
            power = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0  # Convert to watts
            
            print(f"GPU {i} ({name}):")
            print(f"  Memory: {mem_used_gb:.1f}GB / {mem_total_gb:.1f}GB ({mem_percent:.1f}%)")
            print(f"  GPU Utilization: {gpu_util}%")
            print(f"  Memory Utilization: {memory_util}%") 
            print(f"  Temperature: {temp}°C")
            print(f"  Power: {power:.1f}W")
            print()
        
        # CPU and system memory
        cpu_percent = psutil.cpu_percent()
        memory = psutil.virtual_memory()
        
        print(f"CPU: {cpu_percent}%")
        print(f"System Memory: {memory.percent}% ({memory.used/1024**3:.1f}GB / {memory.total/1024**3:.1f}GB)")
        print("-" * 50)
        
        time.sleep(2)

if __name__ == "__main__":
    try:
        monitor_gpu()
    except KeyboardInterrupt:
        print("Monitoring stopped.")
```

## Monitoring & Observability

### Metrics Collection

#### Prometheus Integration

The provided configuration includes comprehensive metrics collection. Key metrics:

**vLLM Server Metrics**
- `vllm_request_total` - Total number of requests
- `vllm_request_duration_seconds` - Request latency histogram
- `vllm_request_prompt_tokens_total` - Input tokens processed
- `vllm_request_completion_tokens_total` - Output tokens generated
- `vllm_cache_hit_total` / `vllm_cache_miss_total` - Cache performance
- `vllm_gpu_cache_usage_percent` - GPU cache utilization
- `vllm_batch_size` - Current batch size

**GPU Metrics (via DCGM Exporter)**
- `DCGM_FI_DEV_GPU_UTIL` - GPU utilization percentage
- `DCGM_FI_DEV_MEM_COPY_UTIL` - Memory copy utilization
- `DCGM_FI_DEV_FB_USED` / `DCGM_FI_DEV_FB_TOTAL` - Memory usage
- `DCGM_FI_DEV_GPU_TEMP` - GPU temperature
- `DCGM_FI_DEV_POWER_USAGE` - Power consumption

#### Custom Metrics Collection

**Application-Level Metrics**
```python
# metrics_collector.py
from prometheus_client import Counter, Histogram, Gauge, start_http_server
import time
import threading

# Define metrics
REQUEST_COUNT = Counter('crush_requests_total', 'Total requests to Crush', ['model', 'endpoint'])
REQUEST_LATENCY = Histogram('crush_request_duration_seconds', 'Request latency', ['model'])
ACTIVE_SESSIONS = Gauge('crush_active_sessions', 'Number of active sessions')
MODEL_LOAD_TIME = Gauge('crush_model_load_seconds', 'Time to load model', ['model'])

class MetricsCollector:
    def __init__(self):
        self.start_time = time.time()
        
    def record_request(self, model, endpoint, latency):
        REQUEST_COUNT.labels(model=model, endpoint=endpoint).inc()
        REQUEST_LATENCY.labels(model=model).observe(latency)
    
    def update_active_sessions(self, count):
        ACTIVE_SESSIONS.set(count)
    
    def record_model_load_time(self, model, load_time):
        MODEL_LOAD_TIME.labels(model=model).set(load_time)

# Start metrics server
def start_metrics_server(port=8080):
    start_http_server(port)
    print(f"Metrics server started on port {port}")

if __name__ == "__main__":
    collector = MetricsCollector()
    start_metrics_server()
    
    # Keep the server running
    while True:
        time.sleep(60)
```

### Grafana Dashboards

#### vLLM Performance Dashboard

The provided `grafana-vllm-dashboard.json` includes panels for:

1. **Overview Panel**
   - Requests per second
   - Average latency
   - Error rate
   - Active connections

2. **Performance Panel**
   - Token throughput (input/output)
   - Batch size trends
   - Queue length
   - Cache hit rate

3. **Resource Utilization**
   - GPU utilization per device
   - GPU memory usage
   - System CPU and RAM
   - Network I/O

4. **Model-Specific Metrics**
   - Per-model request rates
   - Model-specific latencies
   - Context length distribution
   - Generation parameters

#### Custom Dashboard Creation

```json
{
  "dashboard": {
    "title": "Crush + vLLM Custom Metrics",
    "panels": [
      {
        "title": "Request Rate by Model",
        "type": "graph",
        "targets": [
          {
            "expr": "rate(vllm_request_success_total[5m])",
            "legendFormat": "{{model}}"
          }
        ]
      },
      {
        "title": "GPU Memory Usage",
        "type": "singlestat",
        "targets": [
          {
            "expr": "(DCGM_FI_DEV_FB_USED / DCGM_FI_DEV_FB_TOTAL) * 100",
            "legendFormat": "GPU {{gpu}}"
          }
        ]
      },
      {
        "title": "Token Throughput",
        "type": "graph", 
        "targets": [
          {
            "expr": "rate(vllm_request_prompt_tokens_total[1m])",
            "legendFormat": "Input Tokens/sec"
          },
          {
            "expr": "rate(vllm_request_completion_tokens_total[1m])",
            "legendFormat": "Output Tokens/sec"
          }
        ]
      }
    ]
  }
}
```

### Logging and Alerting

#### Structured Logging Configuration

**vLLM Logging Setup**
```python
# logging_config.py
import logging
import json
from datetime import datetime

class StructuredFormatter(logging.Formatter):
    def format(self, record):
        log_data = {
            'timestamp': datetime.utcnow().isoformat(),
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno
        }
        
        if hasattr(record, 'request_id'):
            log_data['request_id'] = record.request_id
        if hasattr(record, 'model'):
            log_data['model'] = record.model
        if hasattr(record, 'latency'):
            log_data['latency'] = record.latency
            
        return json.dumps(log_data)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    handlers=[
        logging.FileHandler('/var/log/vllm/vllm.json.log'),
        logging.StreamHandler()
    ]
)

# Apply structured formatter
for handler in logging.root.handlers:
    handler.setFormatter(StructuredFormatter())
```

**Log Analysis Scripts**
```bash
# Real-time error monitoring
tail -f /var/log/vllm/vllm.json.log | jq 'select(.level == "ERROR")'

# Performance analysis
grep '"level":"INFO"' /var/log/vllm/vllm.json.log | \
    jq 'select(has("latency")) | .latency' | \
    awk '{sum+=$1; count++} END {print "Average latency:", sum/count "s"}'

# Request rate analysis
grep '"level":"INFO"' /var/log/vllm/vllm.json.log | \
    jq 'select(has("request_id"))' | \
    wc -l
```

#### Alert Rules Configuration

**Prometheus Alert Rules**
```yaml
# alerts.yml
groups:
- name: vllm_alerts
  rules:
  - alert: vLLMHighLatency
    expr: vllm:request_latency_p99_5m > 10
    for: 2m
    labels:
      severity: warning
    annotations:
      summary: "vLLM high latency detected"
      description: "99th percentile latency is {{ $value }}s"
      
  - alert: vLLMHighErrorRate
    expr: rate(vllm_request_error_total[5m]) / rate(vllm_request_total[5m]) > 0.05
    for: 1m
    labels:
      severity: critical
    annotations:
      summary: "vLLM high error rate"
      description: "Error rate is {{ $value | humanizePercentage }}"
      
  - alert: GPUMemoryHigh
    expr: vllm:gpu_memory_usage_percent > 95
    for: 2m
    labels:
      severity: warning
    annotations:
      summary: "GPU memory usage high"
      description: "GPU memory usage is {{ $value }}%"
      
  - alert: vLLMDown
    expr: up{job="vllm"} == 0
    for: 30s
    labels:
      severity: critical
    annotations:
      summary: "vLLM server is down"
      description: "vLLM server has been down for more than 30 seconds"
```

**Alertmanager Configuration**
```yaml
# alertmanager.yml
global:
  smtp_smarthost: 'localhost:587'
  smtp_from: 'alertmanager@yourcompany.com'

route:
  group_by: ['alertname']
  group_wait: 10s
  group_interval: 10s
  repeat_interval: 1h
  receiver: 'web.hook'

receivers:
- name: 'web.hook'
  email_configs:
  - to: 'admin@yourcompany.com'
    subject: 'vLLM Alert: {{ .GroupLabels.alertname }}'
    body: |
      {{ range .Alerts }}
      Alert: {{ .Annotations.summary }}
      Description: {{ .Annotations.description }}
      {{ end }}
      
  slack_configs:
  - api_url: 'YOUR_SLACK_WEBHOOK_URL'
    channel: '#alerts'
    title: 'vLLM Alert: {{ .GroupLabels.alertname }}'
    text: '{{ range .Alerts }}{{ .Annotations.description }}{{ end }}'
```

## Troubleshooting

### Common Issues and Solutions

#### Model Loading Issues

**Problem: CUDA Out of Memory**
```
RuntimeError: CUDA out of memory. Tried to allocate X.XX GiB
```

**Solutions:**
```bash
# 1. Reduce GPU memory utilization
--gpu-memory-utilization 0.8  # Instead of 0.9

# 2. Reduce max model length
--max-model-len 16384  # Instead of 32768

# 3. Use more aggressive quantization
--quantization gptq  # Instead of awq

# 4. Enable tensor parallelism if multiple GPUs
--tensor-parallel-size 2

# 5. Clear GPU cache
python -c "import torch; torch.cuda.empty_cache()"
```

**Problem: Model Download Fails**
```
OSError: Can't load tokenizer for 'model-name'. Make sure that:
```

**Solutions:**
```bash
# 1. Set HuggingFace token for gated models
export HF_TOKEN="your_hf_token_here"

# 2. Pre-download model
python -c "
from transformers import AutoTokenizer, AutoModelForCausalLM
AutoTokenizer.from_pretrained('model-name')
print('Model downloaded successfully')
"

# 3. Check disk space
df -h ~/.cache/huggingface

# 4. Use local model path
--model /local/path/to/model
```

#### Performance Issues

**Problem: Low Throughput**
```
Expected: 100+ req/s, Getting: 10-20 req/s
```

**Diagnosis:**
```bash
# Check GPU utilization
nvidia-smi -l 1

# Check batch size metrics
curl http://localhost:8000/metrics | grep batch_size

# Check queue metrics
curl http://localhost:8000/metrics | grep queue
```

**Solutions:**
```bash
# 1. Increase batch size parameters
--max-num-seqs 512
--max-num-batched-tokens 16384

# 2. Enable performance optimizations
--enforce-eager false
--enable-chunked-prefill
--enable-prefix-caching

# 3. Tune scheduler
--scheduler-delay-factor 0.0
--max-parallel-loading-workers 4
```

**Problem: High Latency**
```
Expected: <1s, Getting: 5-10s
```

**Solutions:**
```bash
# 1. Reduce batch size for latency
--max-num-seqs 64
--max-num-batched-tokens 4096

# 2. Enable CUDA graphs
--enforce-eager false

# 3. Check for CPU bottlenecks
htop  # Look for high CPU usage
```

#### Connection Issues

**Problem: Connection Refused**
```
requests.exceptions.ConnectionError: ('Connection aborted.', RemoteDisconnected('Remote end closed connection without response'))
```

**Solutions:**
```bash
# 1. Check if vLLM is running
curl http://localhost:8000/health

# 2. Check port binding
netstat -tlnp | grep :8000

# 3. Check firewall
sudo ufw status
sudo iptables -L

# 4. Increase timeouts in client
timeout = 300  # 5 minutes
```

**Problem: API Compatibility Issues**
```
OpenAI API error: Invalid model name
```

**Solutions:**
```bash
# 1. Check available models
curl http://localhost:8000/v1/models

# 2. Update Crush configuration
{
  "providers": {
    "vllm": {
      "models": [
        {
          "id": "actual-model-id-from-api",
          "name": "Display Name"
        }
      ]
    }
  }
}
```

#### Docker Issues

**Problem: Docker NVIDIA Runtime Not Found**
```
docker: Error response from daemon: could not select device driver "nvidia" with capabilities: [[gpu]]
```

**Solutions:**
```bash
# 1. Install NVIDIA Container Toolkit
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
  sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
  sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit

# 2. Configure Docker
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker

# 3. Test GPU access
docker run --rm --runtime=nvidia --gpus all nvidia/cuda:11.8-base nvidia-smi
```

#### Kubernetes Issues

**Problem: GPU Scheduling Issues**
```
0/3 nodes are available: 3 Insufficient nvidia.com/gpu
```

**Solutions:**
```bash
# 1. Check GPU Operator installation
kubectl get pods -n gpu-operator

# 2. Verify node GPU advertising
kubectl describe nodes | grep nvidia.com/gpu

# 3. Check resource requests vs limits
kubectl describe pod vllm-pod -n crush-vllm

# 4. Install NVIDIA Device Plugin (if GPU Operator not used)
kubectl create -f https://raw.githubusercontent.com/NVIDIA/k8s-device-plugin/v0.14.1/nvidia-device-plugin.yml
```

### Debug Tools and Commands

#### vLLM Debug Commands

**Server Health Check**
```bash
# Basic health check
curl -f http://localhost:8000/health

# Detailed health check with JSON
curl -s http://localhost:8000/health | jq

# Model information
curl -s http://localhost:8000/v1/models | jq
```

**Performance Testing**
```bash
# Simple completion test
curl -X POST http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen2.5-coder-32b-instruct",
    "prompt": "def fibonacci(n):",
    "max_tokens": 100,
    "temperature": 0.1
  }' | jq

# Chat completion test  
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen2.5-coder-32b-instruct", 
    "messages": [{"role": "user", "content": "Write a hello world function in Python"}],
    "max_tokens": 100
  }' | jq
```

**Metrics Analysis**
```bash
# Get all metrics
curl -s http://localhost:8000/metrics

# Filter specific metrics
curl -s http://localhost:8000/metrics | grep vllm_request

# Analyze request patterns
curl -s http://localhost:8000/metrics | grep vllm_request_duration_seconds | sort
```

#### GPU Debug Commands

**NVIDIA Debugging**
```bash
# Check GPU status
nvidia-smi

# Continuous monitoring
watch -n 1 nvidia-smi

# Memory info
nvidia-smi --query-gpu=memory.used,memory.free,memory.total --format=csv

# Running processes
nvidia-smi pmon

# GPU topology
nvidia-smi topo -m

# Detailed GPU info
nvidia-ml-py3
python -c "
import pynvml
pynvml.nvmlInit()
print(f'Driver version: {pynvml.nvmlSystemGetDriverVersion()}')
print(f'CUDA version: {pynvml.nvmlSystemGetCudaDriverVersion()}')
handle = pynvml.nvmlDeviceGetHandleByIndex(0)
print(f'GPU name: {pynvml.nvmlDeviceGetName(handle)}')
"
```

**CUDA Environment Check**
```bash
# Check CUDA installation
nvcc --version
which nvcc

# Check CUDA libraries
ldconfig -p | grep cuda

# Test CUDA with PyTorch
python -c "
import torch
print(f'CUDA available: {torch.cuda.is_available()}')
print(f'CUDA version: {torch.version.cuda}')
print(f'Device count: {torch.cuda.device_count()}')
if torch.cuda.is_available():
    print(f'Current device: {torch.cuda.get_device_name()}')
"
```

#### Log Analysis Tools

**Real-time Log Monitoring**
```bash
# Follow vLLM logs
tail -f /var/log/vllm/vllm.log

# Filter for errors
tail -f /var/log/vllm/vllm.log | grep -i error

# Parse structured logs
tail -f /var/log/vllm/vllm.json.log | jq 'select(.level == "ERROR")'

# Monitor specific patterns
tail -f /var/log/vllm/vllm.log | grep -E "(OutOfMemory|CUDA|timeout)"
```

**Log Analysis Scripts**
```bash
# Error frequency analysis
cat /var/log/vllm/vllm.log | grep -i error | \
    awk '{print $1 " " $2}' | sort | uniq -c | sort -nr

# Performance trends
grep "request completed" /var/log/vllm/vllm.log | \
    awk '{print $NF}' | \
    sort -n | \
    awk '{
        count++; 
        sum += $1; 
        if (count == 1) min = $1; 
        max = $1
    } END {
        print "Requests:", count
        print "Average latency:", sum/count
        print "Min latency:", min
        print "Max latency:", max
    }'
```

### Support and Community Resources

#### Official Documentation
- [vLLM Documentation](https://docs.vllm.ai/)
- [Crush Documentation](https://github.com/charmbracelet/crush/blob/main/README.md)
- [HuggingFace Transformers](https://huggingface.co/docs/transformers/)

#### Community Forums
- [vLLM GitHub Discussions](https://github.com/vllm-project/vllm/discussions)
- [Charm Community Discord](https://charm.land/discord)
- [r/LocalLLaMA](https://reddit.com/r/LocalLLaMA)

#### Issue Reporting
- [vLLM Issues](https://github.com/vllm-project/vllm/issues)
- [Crush Issues](https://github.com/charmbracelet/crush/issues)

#### Professional Support
- [Charm Professional Support](mailto:vt100@charm.sh)
- NVIDIA Developer Support (for GPU-related issues)
- Cloud provider support (for cloud deployments)

## Example Use Cases

### Development Workflows

#### Code Generation and Review

**Setup: Qwen2.5-Coder for Development**
```json
{
  "$schema": "https://charm.land/crush.json",
  "providers": {
    "vllm-dev": {
      "name": "Development Assistant",
      "type": "openai",
      "base_url": "http://localhost:8000/v1",
      "models": [
        {
          "id": "qwen2.5-coder-32b-instruct",
          "name": "Qwen Coder 32B (Dev)",
          "context_window": 32768,
          "default_max_tokens": 2048,
          "supports_tools": true
        }
      ]
    }
  },
  "mcp": {
    "filesystem": {
      "type": "stdio",
      "command": "mcp-filesystem",
      "args": ["--allowed-paths", "/workspace", "/home/user/projects"]
    },
    "git": {
      "type": "stdio", 
      "command": "mcp-git",
      "args": ["--repo-path", "."]
    }
  },
  "permissions": {
    "allowed_tools": [
      "view",
      "edit", 
      "search",
      "mcp_filesystem_*",
      "mcp_git_*"
    ]
  }
}
```

**Example Development Session**
```bash
# Start Crush in your project directory
cd /path/to/your/project
crush

# Example prompts for development:
# "Review this Python function for performance issues"
# "Generate comprehensive unit tests for the user authentication module" 
# "Refactor this code to follow clean architecture principles"
# "Create API documentation for these endpoints"
# "Optimize this database query for better performance"
```

#### Multi-Language Development

**Multi-Model Setup for Different Languages**
```yaml
# docker-compose.dev.yml
version: '3.8'

services:
  python-coder:
    image: vllm/vllm-openai:latest
    container_name: python-coder
    runtime: nvidia
    ports:
      - "8000:8000"
    environment:
      - NVIDIA_VISIBLE_DEVICES=0
    command: >
      --model deepseek-ai/deepseek-coder-33b-instruct
      --port 8000
      --quantization gptq
      --trust-remote-code
      
  web-developer:
    image: vllm/vllm-openai:latest
    container_name: web-developer
    runtime: nvidia
    ports:
      - "8001:8000"
    environment:
      - NVIDIA_VISIBLE_DEVICES=1
    command: >
      --model Qwen/Qwen2.5-Coder-32B-Instruct-AWQ
      --port 8000
      --quantization awq
      --trust-remote-code

  general-assistant:
    image: vllm/vllm-openai:latest
    container_name: general-assistant
    runtime: nvidia
    ports:
      - "8002:8000"
    environment:
      - NVIDIA_VISIBLE_DEVICES=2
    command: >
      --model mistralai/Mistral-7B-Instruct-v0.2
      --port 8000
      --trust-remote-code
```

**Corresponding Crush Configuration**
```json
{
  "providers": {
    "python-expert": {
      "name": "Python Specialist",
      "type": "openai",
      "base_url": "http://localhost:8000/v1",
      "models": [{"id": "deepseek-coder-33b", "name": "Python Expert"}]
    },
    "web-developer": {
      "name": "Web Development",
      "type": "openai", 
      "base_url": "http://localhost:8001/v1",
      "models": [{"id": "qwen-coder-32b", "name": "Web Developer"}]
    },
    "general-assistant": {
      "name": "General Assistant",
      "type": "openai",
      "base_url": "http://localhost:8002/v1", 
      "models": [{"id": "mistral-7b", "name": "General Helper"}]
    }
  }
}
```

### Production Deployments

#### High-Availability Setup

**Multi-Instance with Load Balancing**
```yaml
# docker-compose.prod.yml
version: '3.8'

services:
  vllm-1:
    image: vllm/vllm-openai:latest
    container_name: vllm-instance-1
    runtime: nvidia
    expose:
      - "8000"
    environment:
      - NVIDIA_VISIBLE_DEVICES=0
      - VLLM_INSTANCE_ID=1
    command: >
      --model Qwen/Qwen2.5-Coder-32B-Instruct-AWQ
      --port 8000
      --gpu-memory-utilization 0.9
      --quantization awq
      --trust-remote-code
      --enable-prefix-caching
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
    restart: unless-stopped
    
  vllm-2:
    image: vllm/vllm-openai:latest
    container_name: vllm-instance-2
    runtime: nvidia
    expose:
      - "8000"
    environment:
      - NVIDIA_VISIBLE_DEVICES=1
      - VLLM_INSTANCE_ID=2
    command: >
      --model Qwen/Qwen2.5-Coder-32B-Instruct-AWQ
      --port 8000
      --gpu-memory-utilization 0.9
      --quantization awq
      --trust-remote-code
      --enable-prefix-caching
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
    restart: unless-stopped

  nginx-lb:
    image: nginx:alpine
    container_name: vllm-loadbalancer
    ports:
      - "8080:80"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf:ro
    depends_on:
      - vllm-1
      - vllm-2
    restart: unless-stopped
    
  redis:
    image: redis:alpine
    container_name: vllm-cache
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data
    restart: unless-stopped

volumes:
  redis_data:
```

**NGINX Load Balancer Configuration**
```nginx
# nginx.conf
upstream vllm_backend {
    least_conn;
    server vllm-1:8000 max_fails=3 fail_timeout=30s;
    server vllm-2:8000 max_fails=3 fail_timeout=30s;
}

server {
    listen 80;
    
    location /health {
        access_log off;
        add_header Content-Type text/plain;
        return 200 "healthy\n";
    }
    
    location / {
        proxy_pass http://vllm_backend;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        
        # Timeouts for LLM requests
        proxy_connect_timeout 60s;
        proxy_send_timeout 300s;
        proxy_read_timeout 300s;
        
        # Buffer settings
        proxy_buffering off;
        proxy_request_buffering off;
        
        # Health checks
        proxy_next_upstream error timeout http_500 http_502 http_503 http_504;
        proxy_next_upstream_tries 2;
    }
    
    # Metrics endpoint
    location /nginx_status {
        stub_status on;
        access_log off;
        allow 127.0.0.1;
        deny all;
    }
}
```

#### Enterprise Integration

**Authentication and Rate Limiting**
```python
# auth_proxy.py - Custom authentication proxy
from fastapi import FastAPI, HTTPException, Depends, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import httpx
import redis
import time
import os

app = FastAPI()
security = HTTPBearer()
redis_client = redis.Redis(host='localhost', port=6379, decode_responses=True)

VLLM_BACKEND = os.getenv('VLLM_BACKEND', 'http://localhost:8000')
RATE_LIMIT_PER_MINUTE = int(os.getenv('RATE_LIMIT_PER_MINUTE', '60'))

async def verify_api_key(credentials: HTTPAuthorizationCredentials = Depends(security)):
    # Implement your API key verification logic
    api_key = credentials.credentials
    
    # Example: Check against database or external service
    if not is_valid_api_key(api_key):
        raise HTTPException(status_code=401, detail="Invalid API key")
    
    return api_key

def is_valid_api_key(api_key: str) -> bool:
    # Implement your API key validation
    # This could check a database, external service, etc.
    valid_keys = os.getenv('VALID_API_KEYS', '').split(',')
    return api_key in valid_keys

def check_rate_limit(api_key: str) -> bool:
    """Check if API key is within rate limits"""
    current_minute = int(time.time() // 60)
    key = f"rate_limit:{api_key}:{current_minute}"
    
    current_count = redis_client.get(key)
    if current_count is None:
        redis_client.setex(key, 60, 1)
        return True
    
    if int(current_count) >= RATE_LIMIT_PER_MINUTE:
        return False
    
    redis_client.incr(key)
    return True

@app.post("/v1/chat/completions")
async def proxy_chat_completions(request: Request, api_key: str = Depends(verify_api_key)):
    # Rate limiting
    if not check_rate_limit(api_key):
        raise HTTPException(status_code=429, detail="Rate limit exceeded")
    
    # Get request body
    body = await request.json()
    
    # Add usage tracking
    redis_client.incr(f"usage:{api_key}:requests")
    
    # Proxy to vLLM
    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(
                f"{VLLM_BACKEND}/v1/chat/completions",
                json=body,
                timeout=300
            )
            
            # Track token usage if available
            if response.status_code == 200:
                result = response.json()
                if 'usage' in result:
                    usage = result['usage']
                    redis_client.incrby(f"usage:{api_key}:prompt_tokens", usage.get('prompt_tokens', 0))
                    redis_client.incrby(f"usage:{api_key}:completion_tokens", usage.get('completion_tokens', 0))
            
            return response.json()
            
        except httpx.TimeoutException:
            raise HTTPException(status_code=504, detail="Request timeout")
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health():
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
```

### Specialized Use Cases

#### Code Review Assistant

**Configuration for Code Review**
```json
{
  "$schema": "https://charm.land/crush.json",
  "providers": {
    "code-reviewer": {
      "name": "Code Review Assistant",
      "type": "openai",
      "base_url": "http://localhost:8000/v1",
      "models": [
        {
          "id": "qwen2.5-coder-32b-instruct",
          "name": "Senior Code Reviewer",
          "context_window": 32768,
          "default_max_tokens": 4096,
          "supports_tools": true
        }
      ]
    }
  },
  "mcp": {
    "git": {
      "type": "stdio",
      "command": "mcp-git-server",
      "args": ["--repo-path", "."]
    },
    "static-analysis": {
      "type": "stdio", 
      "command": "mcp-analysis-server",
      "args": ["--tools", "eslint,pylint,golangci-lint"]
    }
  }
}
```

**Automated Code Review Script**
```bash
#!/bin/bash
# automated_review.sh

set -e

# Configuration
REPO_PATH="${1:-.}"
BRANCH="${2:-main}"
TARGET_BRANCH="${3:-develop}"

cd "$REPO_PATH"

# Get changed files
CHANGED_FILES=$(git diff --name-only "$TARGET_BRANCH"..."$BRANCH")

if [ -z "$CHANGED_FILES" ]; then
    echo "No changes detected"
    exit 0
fi

echo "Files changed: $CHANGED_FILES"

# Start Crush with review configuration
export CRUSH_CONFIG="$REPO_PATH/.crush/review.json"

# Create review prompts for each file
for file in $CHANGED_FILES; do
    if [[ "$file" =~ \.(py|js|ts|go|java|cpp|c)$ ]]; then
        echo "Reviewing $file..."
        
        # Generate review prompt
        cat > /tmp/review_prompt.txt << EOF
Please review this code change in $file:

\`\`\`diff
$(git diff "$TARGET_BRANCH"..."$BRANCH" -- "$file")
\`\`\`

Focus on:
1. Code quality and best practices
2. Potential bugs or security issues  
3. Performance implications
4. Test coverage needs
5. Documentation requirements

Provide specific, actionable feedback.
EOF
        
        # Run review through Crush
        crush --non-interactive --prompt-file /tmp/review_prompt.txt >> "/tmp/review_$file.md"
    fi
done

# Consolidate reviews
cat /tmp/review_*.md > code_review_report.md
echo "Code review complete. See code_review_report.md"
```

#### Documentation Generation

**Setup for Documentation Generation**
```json
{
  "$schema": "https://charm.land/crush.json",
  "providers": {
    "doc-generator": {
      "name": "Documentation Generator",
      "type": "openai",
      "base_url": "http://localhost:8001/v1",
      "models": [
        {
          "id": "qwen2.5-coder-32b-instruct",
          "name": "Tech Writer",
          "context_window": 32768,
          "default_max_tokens": 8192
        }
      ]
    }
  },
  "mcp": {
    "filesystem": {
      "type": "stdio",
      "command": "mcp-filesystem-server",
      "args": ["--allowed-paths", "./src", "./docs"]
    },
    "ast-parser": {
      "type": "stdio",
      "command": "mcp-ast-server",
      "args": ["--languages", "python,javascript,typescript,go"]
    }
  }
}
```

**Documentation Generation Script**
```python
# generate_docs.py
import os
import subprocess
import json
from pathlib import Path

def extract_functions_and_classes(file_path):
    """Extract functions and classes from source file"""
    # This would integrate with AST parsing MCP server
    # For now, simplified version
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Simple extraction (in reality, use proper AST parsing)
    functions = []
    classes = []
    
    lines = content.split('\n')
    for i, line in enumerate(lines):
        if line.strip().startswith('def '):
            func_name = line.split('def ')[1].split('(')[0]
            functions.append({
                'name': func_name,
                'line': i + 1,
                'signature': line.strip()
            })
        elif line.strip().startswith('class '):
            class_name = line.split('class ')[1].split('(')[0].split(':')[0]
            classes.append({
                'name': class_name,
                'line': i + 1,
                'signature': line.strip()
            })
    
    return functions, classes

def generate_api_docs(source_dir, output_dir):
    """Generate API documentation for all source files"""
    
    source_path = Path(source_dir)
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    for py_file in source_path.rglob('*.py'):
        print(f"Processing {py_file}...")
        
        functions, classes = extract_functions_and_classes(py_file)
        
        if not functions and not classes:
            continue
            
        # Generate documentation prompt
        prompt = f"""Generate comprehensive API documentation for the Python module: {py_file.name}

Source file: {py_file}

Functions found:
{json.dumps(functions, indent=2)}

Classes found:
{json.dumps(classes, indent=2)}

File content:
```python
{py_file.read_text()}
```

Please generate:
1. Module overview and purpose
2. Detailed function documentation with parameters, return values, and examples
3. Class documentation with attributes and methods
4. Usage examples
5. Error handling information

Format the output as Markdown suitable for technical documentation.
"""

        # Save prompt to file for Crush processing
        prompt_file = f"/tmp/doc_prompt_{py_file.stem}.txt"
        with open(prompt_file, 'w') as f:
            f.write(prompt)
        
        # Generate documentation using Crush
        output_file = output_path / f"{py_file.stem}.md"
        cmd = f"crush --config .crush/doc_config.json --prompt-file {prompt_file}"
        
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        if result.returncode == 0:
            with open(output_file, 'w') as f:
                f.write(result.stdout)
            print(f"Generated documentation: {output_file}")
        else:
            print(f"Error generating docs for {py_file}: {result.stderr}")

if __name__ == "__main__":
    generate_api_docs("./src", "./docs/api")
```

#### Research and Analysis

**Configuration for Research Tasks**
```json
{
  "$schema": "https://charm.land/crush.json",
  "providers": {
    "researcher": {
      "name": "Research Assistant",
      "type": "openai",
      "base_url": "http://localhost:8002/v1",
      "models": [
        {
          "id": "qwen2.5-coder-32b-instruct",
          "name": "Research Analyst",
          "context_window": 32768,
          "default_max_tokens": 8192
        }
      ]
    }
  },
  "mcp": {
    "web-search": {
      "type": "http",
      "url": "http://localhost:3001/mcp",
      "headers": {
        "Authorization": "Bearer YOUR_SEARCH_API_TOKEN"
      }
    },
    "arxiv": {
      "type": "stdio",
      "command": "mcp-arxiv-server"
    },
    "github": {
      "type": "http", 
      "url": "https://api.github.com/mcp",
      "headers": {
        "Authorization": "token YOUR_GITHUB_TOKEN"
      }
    }
  }
}
```

**Research Automation Example**
```python
# research_assistant.py
import asyncio
import aiohttp
import json
from datetime import datetime, timedelta

class ResearchAssistant:
    def __init__(self, crush_config_path=".crush/research.json"):
        self.config_path = crush_config_path
        self.base_url = "http://localhost:8002/v1"
        
    async def research_topic(self, topic, days_back=30):
        """Research a topic using multiple sources"""
        
        prompt = f"""Research the topic: "{topic}"

Please perform a comprehensive research analysis including:

1. **Academic Research**:
   - Find recent papers from arXiv (last {days_back} days)
   - Summarize key findings and methodologies
   
2. **Industry Trends**:
   - Search for recent industry reports and blog posts
   - Identify emerging patterns and technologies
   
3. **Open Source Projects**:
   - Find relevant GitHub repositories
   - Analyze code trends and popular implementations
   
4. **Technical Analysis**:
   - Compare different approaches and solutions
   - Identify pros and cons of various methods
   
5. **Future Outlook**:
   - Predict future developments
   - Identify research gaps and opportunities

Format the response as a structured research report with citations.
"""

        async with aiohttp.ClientSession() as session:
            data = {
                "model": "qwen2.5-coder-32b-instruct",
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 8192,
                "temperature": 0.3
            }
            
            async with session.post(f"{self.base_url}/chat/completions", 
                                  json=data) as response:
                result = await response.json()
                return result['choices'][0]['message']['content']

    async def analyze_codebase(self, repo_url, focus_areas=None):
        """Analyze a codebase for specific aspects"""
        
        focus = focus_areas or [
            "Architecture patterns",
            "Code quality metrics", 
            "Security considerations",
            "Performance bottlenecks",
            "Testing coverage"
        ]
        
        prompt = f"""Analyze the codebase at: {repo_url}

Focus on the following areas:
{chr(10).join(f'- {area}' for area in focus)}

Please provide:
1. **Architecture Overview**: High-level structure and design patterns
2. **Code Quality Analysis**: Maintainability, readability, and best practices
3. **Security Assessment**: Potential vulnerabilities and security practices
4. **Performance Analysis**: Bottlenecks and optimization opportunities
5. **Testing Strategy**: Current testing approach and recommendations
6. **Documentation Quality**: API docs, README, and code comments
7. **Recommendations**: Specific improvement suggestions

Use the GitHub MCP server to access repository information.
"""

        async with aiohttp.ClientSession() as session:
            data = {
                "model": "qwen2.5-coder-32b-instruct",
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 8192,
                "temperature": 0.2
            }
            
            async with session.post(f"{self.base_url}/chat/completions",
                                  json=data) as response:
                result = await response.json()
                return result['choices'][0]['message']['content']

async def main():
    researcher = ResearchAssistant()
    
    # Research a topic
    topic = "Large Language Model Inference Optimization"
    research_report = await researcher.research_topic(topic)
    
    with open(f"research_{topic.replace(' ', '_').lower()}.md", 'w') as f:
        f.write(f"# Research Report: {topic}\n\n")
        f.write(f"Generated on: {datetime.now().isoformat()}\n\n")
        f.write(research_report)
    
    print(f"Research report saved for topic: {topic}")

if __name__ == "__main__":
    asyncio.run(main())
```

## Advanced Configuration

### Custom Model Integrations

#### Fine-tuned Model Setup

**Preparing a Fine-tuned Model for vLLM**
```python
# prepare_finetuned_model.py
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer
)
import torch
from peft import LoraConfig, get_peft_model, TaskType

def prepare_model_for_vllm(base_model_path, lora_weights_path, output_path):
    """Merge LoRA weights with base model for vLLM compatibility"""
    
    print(f"Loading base model from {base_model_path}")
    model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    
    tokenizer = AutoTokenizer.from_pretrained(base_model_path)
    
    # Load and merge LoRA weights
    if lora_weights_path:
        print(f"Merging LoRA weights from {lora_weights_path}")
        from peft import PeftModel
        
        model = PeftModel.from_pretrained(model, lora_weights_path)
        model = model.merge_and_unload()
    
    # Save merged model
    print(f"Saving merged model to {output_path}")
    model.save_pretrained(output_path, safe_serialization=True)
    tokenizer.save_pretrained(output_path)
    
    print("Model prepared successfully for vLLM")

# Example usage
if __name__ == "__main__":
    prepare_model_for_vllm(
        base_model_path="Qwen/Qwen2.5-Coder-32B-Instruct",
        lora_weights_path="./my_finetuned_lora",
        output_path="./models/my_custom_coder"
    )
```

**vLLM Configuration for Custom Model**
```bash
# Start vLLM with custom fine-tuned model
python -m vllm.entrypoints.openai.api_server \
    --model ./models/my_custom_coder \
    --trust-remote-code \
    --max-model-len 32768 \
    --gpu-memory-utilization 0.9 \
    --port 8000
```

#### Multi-Modal Model Integration

**Setup for Vision-Language Models**
```python
# vision_language_setup.py
import torch
from vllm import LLM, SamplingParams

# Configuration for vision-language model
model_config = {
    "model": "Qwen/Qwen2-VL-72B-Instruct-AWQ",
    "trust_remote_code": True,
    "max_model_len": 32768,
    "gpu_memory_utilization": 0.9,
    "quantization": "awq",
    "multimodal": True,
    "image_input_type": "pixel_values",
    "image_token_id": 151646,
    "image_input_shape": "1,3,448,448"
}

# Start vision-language model server
vllm_cmd = [
    "python", "-m", "vllm.entrypoints.openai.api_server",
    "--model", model_config["model"],
    "--trust-remote-code",
    "--max-model-len", str(model_config["max_model_len"]),
    "--gpu-memory-utilization", str(model_config["gpu_memory_utilization"]),
    "--quantization", model_config["quantization"],
    "--port", "8000"
]

print("Command to start vision-language model:")
print(" ".join(vllm_cmd))
```

**Crush Configuration for Vision Models**
```json
{
  "$schema": "https://charm.land/crush.json",
  "providers": {
    "vision-coder": {
      "name": "Vision-Language Coder",
      "type": "openai",
      "base_url": "http://localhost:8000/v1",
      "models": [
        {
          "id": "qwen2-vl-72b-instruct",
          "name": "Qwen2-VL 72B (Code + Vision)",
          "context_window": 32768,
          "supports_attachments": true,
          "supports_vision": true,
          "max_image_size": "1024x1024",
          "supported_formats": ["png", "jpg", "jpeg", "gif", "bmp"]
        }
      ]
    }
  }
}
```

### Advanced Networking

#### SSL/TLS Configuration

**NGINX SSL Proxy**
```nginx
# nginx_ssl.conf
server {
    listen 443 ssl http2;
    server_name your-domain.com;
    
    ssl_certificate /etc/ssl/certs/your-domain.crt;
    ssl_certificate_key /etc/ssl/private/your-domain.key;
    
    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_ciphers ECDHE-RSA-AES256-GCM-SHA512:DHE-RSA-AES256-GCM-SHA512:ECDHE-RSA-AES256-GCM-SHA384:DHE-RSA-AES256-GCM-SHA384;
    ssl_prefer_server_ciphers off;
    
    location / {
        proxy_pass http://localhost:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        
        # WebSocket support for streaming
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        
        proxy_connect_timeout 60s;
        proxy_send_timeout 300s;
        proxy_read_timeout 300s;
        
        proxy_buffering off;
        proxy_request_buffering off;
    }
}

# Redirect HTTP to HTTPS
server {
    listen 80;
    server_name your-domain.com;
    return 301 https://$server_name$request_uri;
}
```

#### Advanced Load Balancing

**Consul-Template Integration**
```bash
# consul_template_nginx.conf.tpl
upstream vllm_backend {
    {{range service "vllm-server"}}
    server {{.Address}}:{{.Port}} max_fails=3 fail_timeout=30s weight={{.ServiceMeta.weight | or "1"}};
    {{end}}
}

server {
    listen 80;
    
    location /health {
        access_log off;
        add_header Content-Type text/plain;
        return 200 "healthy\n";
    }
    
    location / {
        proxy_pass http://vllm_backend;
        proxy_next_upstream error timeout http_500 http_502 http_503 http_504;
        proxy_next_upstream_tries 3;
        
        # Health check URL
        proxy_next_upstream_timeout 60s;
        
        # Custom headers for load balancing
        proxy_set_header X-Load-Balancer "nginx";
        proxy_set_header X-Backend-Server $upstream_addr;
    }
}
```

### Security Hardening

#### API Security

**JWT Authentication Integration**
```python
# jwt_auth_middleware.py
from fastapi import FastAPI, HTTPException, Depends, Request
from fastapi.security import HTTPBearer
import jwt
import os
from datetime import datetime, timedelta

app = FastAPI()
security = HTTPBearer()

JWT_SECRET = os.getenv('JWT_SECRET', 'your-super-secret-key')
JWT_ALGORITHM = 'HS256'
JWT_EXPIRATION_HOURS = 24

def create_access_token(user_id: str, scopes: list = None):
    """Create JWT access token"""
    payload = {
        'sub': user_id,
        'scopes': scopes or [],
        'exp': datetime.utcnow() + timedelta(hours=JWT_EXPIRATION_HOURS),
        'iat': datetime.utcnow()
    }
    return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)

def verify_token(token: str) -> dict:
    """Verify JWT token"""
    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
        return payload
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token expired")
    except jwt.JWTError:
        raise HTTPException(status_code=401, detail="Invalid token")

async def get_current_user(request: Request):
    """Extract and verify user from JWT token"""
    authorization = request.headers.get('Authorization')
    if not authorization:
        raise HTTPException(status_code=401, detail="Authorization header missing")
    
    try:
        scheme, token = authorization.split()
        if scheme.lower() != 'bearer':
            raise HTTPException(status_code=401, detail="Invalid authorization scheme")
        
        payload = verify_token(token)
        return payload
        
    except ValueError:
        raise HTTPException(status_code=401, detail="Invalid authorization header")

@app.middleware("http")
async def auth_middleware(request: Request, call_next):
    """Global authentication middleware"""
    
    # Skip auth for health checks and public endpoints
    if request.url.path in ['/health', '/metrics']:
        return await call_next(request)
    
    # Verify authentication for API endpoints
    if request.url.path.startswith('/v1/'):
        try:
            user = await get_current_user(request)
            request.state.user = user
        except HTTPException:
            return HTTPException(status_code=401, detail="Authentication required")
    
    return await call_next(request)

@app.post("/v1/chat/completions")
async def protected_chat_completions(request: Request):
    """Protected chat completions endpoint"""
    user = request.state.user
    
    # Check user scopes/permissions
    if 'chat' not in user.get('scopes', []):
        raise HTTPException(status_code=403, detail="Insufficient permissions")
    
    # Forward to vLLM backend
    # ... implementation
```

#### Network Security

**Firewall Configuration (UFW)**
```bash
#!/bin/bash
# firewall_setup.sh

# Reset firewall
sudo ufw --force reset

# Default policies
sudo ufw default deny incoming
sudo ufw default allow outgoing

# SSH access
sudo ufw allow ssh

# vLLM API server (restrict to specific IPs)
sudo ufw allow from 10.0.0.0/8 to any port 8000
sudo ufw allow from 172.16.0.0/12 to any port 8000
sudo ufw allow from 192.168.0.0/16 to any port 8000

# HTTPS traffic
sudo ufw allow 443

# Monitoring (restrict to monitoring network)
sudo ufw allow from 10.0.100.0/24 to any port 9090  # Prometheus
sudo ufw allow from 10.0.100.0/24 to any port 3000  # Grafana

# Enable firewall
sudo ufw enable

# Show status
sudo ufw status verbose
```

**Container Security (Docker)**
```yaml
# docker-compose.secure.yml
version: '3.8'

services:
  vllm:
    image: vllm/vllm-openai:latest
    container_name: secure-vllm
    runtime: nvidia
    
    # Security configurations
    user: "1001:1001"  # Non-root user
    read_only: true
    cap_drop:
      - ALL
    cap_add:
      - CAP_SYS_NICE  # Required for GPU access
    security_opt:
      - no-new-privileges:true
      - seccomp:unconfined  # Required for CUDA
    
    # Resource limits
    deploy:
      resources:
        limits:
          memory: 32G
          pids: 1000
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    
    # Networking
    networks:
      - vllm-network
    
    # Volumes (read-only where possible)
    volumes:
      - models:/models:ro
      - cache:/root/.cache/huggingface:rw
      - /tmp:/tmp:rw,noexec,nosuid,size=2g
    
    environment:
      - NVIDIA_VISIBLE_DEVICES=all
      - CUDA_VISIBLE_DEVICES=0
    
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 60s

networks:
  vllm-network:
    driver: bridge
    ipam:
      config:
        - subnet: 172.20.0.0/16

volumes:
  models:
    driver: local
    driver_opts:
      type: none
      o: bind
      device: /opt/secure/models
  cache:
    driver: local
```

### Scalability Patterns

#### Horizontal Scaling with Kubernetes

**Advanced Kubernetes Deployment**
```yaml
# k8s/advanced-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: vllm-deployment
  namespace: crush-vllm
spec:
  replicas: 3
  strategy:
    type: RollingUpdate
    rollingUpdate:
      maxSurge: 1
      maxUnavailable: 0
  selector:
    matchLabels:
      app: vllm-server
  template:
    metadata:
      labels:
        app: vllm-server
        version: v1.0.0
      annotations:
        prometheus.io/scrape: "true"
        prometheus.io/port: "8000"
        prometheus.io/path: "/metrics"
    spec:
      # Pod Security Context
      securityContext:
        runAsNonRoot: true
        runAsUser: 1000
        runAsGroup: 1000
        fsGroup: 1000
        
      # Node Selection
      nodeSelector:
        accelerator: nvidia-tesla-v100
        instance-type: gpu-optimized
        
      # Tolerations for GPU nodes
      tolerations:
      - key: "nvidia.com/gpu"
        operator: "Exists"
        effect: "NoSchedule"
        
      # Anti-affinity for high availability
      affinity:
        podAntiAffinity:
          preferredDuringSchedulingIgnoredDuringExecution:
          - weight: 100
            podAffinityTerm:
              labelSelector:
                matchExpressions:
                - key: app
                  operator: In
                  values:
                  - vllm-server
              topologyKey: kubernetes.io/hostname
              
      containers:
      - name: vllm
        image: vllm/vllm-openai:v0.2.0
        ports:
        - containerPort: 8000
          name: http
          protocol: TCP
          
        # Resource Management
        resources:
          requests:
            nvidia.com/gpu: 1
            memory: "16Gi"
            cpu: "4"
          limits:
            nvidia.com/gpu: 1
            memory: "32Gi"
            cpu: "8"
            
        # Environment Variables
        env:
        - name: NVIDIA_VISIBLE_DEVICES
          value: "all"
        - name: CUDA_VISIBLE_DEVICES
          value: "0"
        - name: VLLM_MODEL
          valueFrom:
            configMapKeyRef:
              name: vllm-config
              key: model_name
        - name: GPU_MEMORY_UTILIZATION
          valueFrom:
            configMapKeyRef:
              name: vllm-config
              key: gpu_memory
              
        # Startup Command
        command:
        - python
        - -m
        - vllm.entrypoints.openai.api_server
        args:
        - --model
        - $(VLLM_MODEL)
        - --port
        - "8000"
        - --gpu-memory-utilization
        - $(GPU_MEMORY_UTILIZATION)
        - --max-model-len
        - "32768"
        - --quantization
        - awq
        - --trust-remote-code
        - --enable-prefix-caching
        - --max-num-seqs
        - "256"
        
        # Health Checks
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 180
          periodSeconds: 30
          timeoutSeconds: 10
          failureThreshold: 3
          
        readinessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 60
          periodSeconds: 10
          timeoutSeconds: 5
          failureThreshold: 3
          
        # Volume Mounts
        volumeMounts:
        - name: model-cache
          mountPath: /root/.cache/huggingface
        - name: shared-memory
          mountPath: /dev/shm
          
        # Security Context
        securityContext:
          allowPrivilegeEscalation: false
          readOnlyRootFilesystem: false
          capabilities:
            drop:
            - ALL
            add:
            - CAP_SYS_NICE
            
      # Volumes
      volumes:
      - name: model-cache
        persistentVolumeClaim:
          claimName: model-cache-pvc
      - name: shared-memory
        emptyDir:
          medium: Memory
          sizeLimit: 4Gi
          
      # DNS Configuration
      dnsPolicy: ClusterFirst
      restartPolicy: Always
      terminationGracePeriodSeconds: 60

---
# Horizontal Pod Autoscaler
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: vllm-hpa
  namespace: crush-vllm
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: vllm-deployment
  minReplicas: 2
  maxReplicas: 10
  
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
        
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 80
        
  # Custom metrics (requires metrics server)
  - type: Pods
    pods:
      metric:
        name: vllm_requests_per_second
      target:
        type: AverageValue
        averageValue: "10"
        
  behavior:
    scaleDown:
      stabilizationWindowSeconds: 300
      policies:
      - type: Percent
        value: 10
        periodSeconds: 60
    scaleUp:
      stabilizationWindowSeconds: 60
      policies:
      - type: Percent
        value: 50
        periodSeconds: 60

---
# Service
apiVersion: v1
kind: Service
metadata:
  name: vllm-service
  namespace: crush-vllm
  annotations:
    prometheus.io/scrape: "true"
    prometheus.io/port: "8000"
spec:
  selector:
    app: vllm-server
  ports:
  - name: http
    port: 8000
    targetPort: 8000
    protocol: TCP
  type: ClusterIP
  sessionAffinity: None

---
# Ingress
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: vllm-ingress
  namespace: crush-vllm
  annotations:
    nginx.ingress.kubernetes.io/rewrite-target: /
    nginx.ingress.kubernetes.io/proxy-read-timeout: "300"
    nginx.ingress.kubernetes.io/proxy-connect-timeout: "300"
    nginx.ingress.kubernetes.io/proxy-body-size: "100m"
    cert-manager.io/cluster-issuer: "letsencrypt-prod"
spec:
  tls:
  - hosts:
    - vllm.yourdomain.com
    secretName: vllm-tls
  rules:
  - host: vllm.yourdomain.com
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: vllm-service
            port:
              number: 8000
```

This comprehensive deployment guide covers everything from basic setup to advanced production configurations. The documentation is structured to be both beginner-friendly and comprehensive for advanced users, with practical examples, troubleshooting guidance, and real-world use cases.

The guide includes:
- Complete prerequisites and system requirements
- Multiple deployment methods (local, Docker, Kubernetes, bare metal)
- Detailed GPU configuration for different hardware
- Performance tuning and optimization
- Comprehensive monitoring and observability
- Extensive troubleshooting section
- Practical use cases and examples
- Advanced configuration patterns

This should serve as a complete reference for anyone deploying Crush with vLLM in any environment.