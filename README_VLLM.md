# vLLM Installation for RTX 5000 - Quick Start Guide

This directory contains optimized installation and monitoring tools for vLLM on Quadro RTX 5000 hardware.

## 🚀 Quick Start (3 Steps)

### Step 1: Check System Readiness
```bash
python3 check_vllm_readiness.py
```

### Step 2: Install vLLM (Optimized)
```bash
./install_vllm_optimized.sh
```

### Step 3: Activate and Test
```bash
source ~/.config/vllm/activate.sh
python ~/.config/vllm/benchmark.py
```

## 📁 Files Overview

| File | Purpose |
|------|---------|
| `check_vllm_readiness.py` | Pre-installation system validation |
| `install_vllm_optimized.sh` | Complete installation with RTX 5000 optimizations |
| `vllm_monitor.py` | Real-time performance monitoring |
| `VLLM_USAGE.md` | Comprehensive usage guide (created after installation) |

## 🎯 Hardware Specifications

- **GPU**: Quadro RTX 5000 (16GB VRAM, Compute 7.5)
- **CPU**: Intel i7-9850H (12 cores) 
- **RAM**: 32GB
- **OS**: Linux (Fedora 42)

## ⚡ Optimizations Applied

### Memory Management
- GPU utilization: 85% (13.6GB for models)
- Batch size: 8 sequences
- Block size: 16
- Max model length: 4096 tokens

### Performance Features
- Flash Attention enabled
- Prefix caching
- Chunked prefill
- Tensor parallelism optimized for single GPU

### Python 3.13 Compatibility
- Automatic fallback to Python 3.11/3.10 if available
- Specific package version pinning
- Build optimization flags
- Multiple installation strategies

## 🛠️ Monitoring Commands

```bash
# Real-time monitoring
python vllm_monitor.py monitor

# One-time status check
python vllm_monitor.py status

# Optimization tips
python vllm_monitor.py tips

# Performance report
python vllm_monitor.py report
```

## 📊 Expected Performance

### Model Size Recommendations
- **Small (< 1GB)**: microsoft/DialoGPT-medium, distilgpt2
- **Medium (1-4GB)**: gpt2-xl, microsoft/DialoGPT-large
- **Large (4-8GB)**: Llama-2-7b-chat (with quantization)
- **XL (8-13GB)**: Requires aggressive quantization (AWQ/GPTQ)

### Throughput Estimates
- Small models: 50-100 tokens/second
- Medium models: 20-50 tokens/second  
- Large models: 10-30 tokens/second (quantized)

## 🔧 Quick API Server

```bash
# Activate environment
source ~/.config/vllm/activate.sh

# Start server with optimized settings
python -m vllm.entrypoints.openai.api_server \
  --model microsoft/DialoGPT-medium \
  --gpu-memory-utilization 0.85 \
  --max-model-len 4096 \
  --port 8000
```

## 🐛 Troubleshooting

### Common Issues
1. **OOM Error**: Reduce `gpu_memory_utilization` to 0.7
2. **Import Error**: Check virtual environment activation
3. **CUDA Error**: Verify driver version with `nvidia-smi`
4. **Slow Performance**: Monitor with `python vllm_monitor.py monitor`

### Health Check
```bash
source ~/.config/vllm/activate.sh
python -c "
import torch, vllm
print(f'PyTorch CUDA: {torch.cuda.is_available()}')
print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}')
print(f'vLLM: {vllm.__version__}')
"
```

## 📚 Additional Resources

- **Configuration**: `~/.config/vllm/config.yaml`
- **Environment Setup**: `~/.config/vllm/activate.sh`
- **Performance Logs**: `~/.config/vllm/performance.log`
- **Virtual Environment**: `~/venvs/vllm-optimized/`

---

*Optimized for: Quadro RTX 5000, Python 3.13, 32GB RAM*  
*Created: $(date '+%Y-%m-%d')*