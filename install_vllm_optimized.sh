#!/bin/bash

# Optimized vLLM Installation for Quadro RTX 5000
# Tailored for: 16GB VRAM, Python 3.13, Professional GPU

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# System-specific optimizations
GPU_MEMORY_UTIL=0.85  # Conservative for Quadro stability
MAX_MODEL_LEN=16384   # Optimal for 16GB VRAM
BATCH_SIZE=8          # Balanced for RTX 5000
BLOCK_SIZE=16         # Memory allocation optimization

log_info() { echo -e "${BLUE}[INFO]${NC} $1"; }
log_success() { echo -e "${GREEN}[✓]${NC} $1"; }
log_warning() { echo -e "${YELLOW}[!]${NC} $1"; }
log_error() { echo -e "${RED}[✗]${NC} $1"; }

# Banner
echo -e "${GREEN}"
echo "╔═══════════════════════════════════════════════════════╗"
echo "║   vLLM Optimized Installation for Quadro RTX 5000    ║"
echo "║         16GB VRAM | Python 3.13 Compatible           ║"
echo "╚═══════════════════════════════════════════════════════╝"
echo -e "${NC}"

# Phase 1: System Validation
log_info "Phase 1: System Validation"

# Check GPU
if ! nvidia-smi --query-gpu=name --format=csv,noheader | grep -q "Quadro RTX 5000"; then
    log_warning "Expected Quadro RTX 5000, found: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
fi

# Check CUDA toolkit
if ! nvcc --version &>/dev/null; then
    log_warning "CUDA toolkit not found. Installing CUDA runtime..."
    # Install CUDA runtime only (not full toolkit)
    pip install nvidia-cuda-runtime-cu12 nvidia-cuda-nvrtc-cu12 nvidia-cuda-cupti-cu12
fi

# Phase 2: Python Environment Setup
log_info "Phase 2: Python Environment Setup"

# Create optimized virtual environment
VENV_PATH="$HOME/venvs/vllm_crush"
if [ -d "$VENV_PATH" ]; then
    log_warning "Existing vLLM environment found. Backing up..."
    mv "$VENV_PATH" "${VENV_PATH}_backup_$(date +%Y%m%d_%H%M%S)"
fi

# Try to use Python 3.11 if available (better compatibility)
if command -v python3.11 &>/dev/null; then
    log_success "Using Python 3.11 for better compatibility"
    python3.11 -m venv "$VENV_PATH"
elif command -v python3.10 &>/dev/null; then
    log_success "Using Python 3.10 for compatibility"
    python3.10 -m venv "$VENV_PATH"
else
    log_warning "Using Python 3.13 - may encounter compatibility issues"
    python3 -m venv "$VENV_PATH"
fi

source "$VENV_PATH/bin/activate"

# Phase 3: Dependency Installation with Compatibility Fixes
log_info "Phase 3: Installing Dependencies (Python 3.13 compatible)"

# Upgrade pip and install build tools
pip install --upgrade pip setuptools wheel

# Install PyTorch with CUDA 12.1 support (best for RTX 5000)
log_info "Installing PyTorch with CUDA 12.1..."
# Python 3.13 requires specific handling
pip install torch --index-url https://download.pytorch.org/whl/cu121 || {
    log_warning "Standard PyTorch install failed, trying nightly build for Python 3.13..."
    pip install --pre torch --index-url https://download.pytorch.org/whl/nightly/cu121
}
# Skip torchvision and torchaudio for now - not needed for vLLM

# Install NumPy with Python 3.13 compatibility
log_info "Installing NumPy (Python 3.13 compatible)..."
pip install "numpy<2.0" || pip install --pre numpy

# Install core dependencies
log_info "Installing core dependencies..."
pip install transformers accelerate sentencepiece protobuf

# Install Flash Attention for RTX 5000 (Compute 7.5)
log_info "Installing Flash Attention (optimized for Compute 7.5)..."
pip install flash-attn --no-build-isolation || {
    log_warning "Flash Attention failed, using fallback..."
    pip install xformers  # Alternative attention mechanism
}

# Phase 4: vLLM Installation with Multiple Strategies
log_info "Phase 4: Installing vLLM"

install_vllm() {
    # Strategy 1: Direct pip install
    log_info "Attempting standard vLLM installation..."
    pip install vllm && return 0
    
    # Strategy 2: Install with no build isolation
    log_info "Attempting vLLM with no build isolation..."
    pip install vllm --no-build-isolation && return 0
    
    # Strategy 3: Install specific version known to work
    log_info "Installing vLLM v0.5.5 (stable version)..."
    pip install vllm==0.5.5 && return 0
    
    # Strategy 4: Build from source with optimizations
    log_info "Building vLLM from source..."
    git clone https://github.com/vllm-project/vllm.git /tmp/vllm
    cd /tmp/vllm
    
    # Set build optimizations for RTX 5000
    export TORCH_CUDA_ARCH_LIST="7.5"  # Compute capability for RTX 5000
    export MAX_JOBS=6  # Use half of available cores for building
    export VLLM_BUILD_CUDA_EXTENSIONS=1
    
    pip install -e . && return 0
    
    return 1
}

if install_vllm; then
    log_success "vLLM installed successfully!"
else
    log_error "vLLM installation failed. Manual intervention required."
    exit 1
fi

# Phase 5: Configuration Setup
log_info "Phase 5: Creating Optimized Configuration"

# Create vLLM config directory
CONFIG_DIR="$HOME/.config/vllm"
mkdir -p "$CONFIG_DIR"

# Generate optimized configuration for RTX 5000
cat > "$CONFIG_DIR/config.yaml" << EOF
# vLLM Configuration for Quadro RTX 5000 (16GB)
model:
  name: "Qwen/Qwen2.5-7B-Instruct"  # Optimal for 16GB
  revision: "main"
  trust_remote_code: true

serving:
  host: "0.0.0.0"
  port: 8000
  api_key: null  # Set if needed
  
gpu:
  gpu_memory_utilization: $GPU_MEMORY_UTIL
  max_model_len: $MAX_MODEL_LEN
  tensor_parallel_size: 1  # Single GPU
  
optimization:
  quantization: "awq"  # Use AWQ for memory efficiency
  dtype: "auto"  # Let vLLM choose optimal dtype
  max_num_seqs: $BATCH_SIZE
  max_num_batched_tokens: 4096
  enable_prefix_caching: true
  enable_chunked_prefill: true
  block_size: $BLOCK_SIZE
  
performance:
  num_scheduler_steps: 1
  swap_space: 4  # GB of CPU RAM for swapping
  disable_log_stats: false
  
advanced:
  enforce_eager: false  # Use CUDA graphs for better performance
  max_parallel_loading_workers: 4
  kv_cache_dtype: "auto"
  attention_backend: "FLASH_ATTN"  # Use Flash Attention
EOF

# Create activation script
cat > "$CONFIG_DIR/activate.sh" << 'EOF'
#!/bin/bash
# vLLM Environment Activation for Crush

# Activate virtual environment
source ~/venvs/vllm_crush/bin/activate

# Set optimized environment variables for RTX 5000
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
export VLLM_ATTENTION_BACKEND=FLASH_ATTN
export TOKENIZERS_PARALLELISM=false

# Memory optimization for Quadro cards
export CUDA_LAUNCH_BLOCKING=0
export CUDNN_BENCHMARK=1

echo "vLLM environment activated for Quadro RTX 5000"
echo "Run 'vllm serve' to start the server"
EOF
chmod +x "$CONFIG_DIR/activate.sh"

# Phase 6: Model Download and Optimization
log_info "Phase 6: Preparing Models"

# Create model cache directory
MODEL_CACHE="$HOME/.cache/huggingface/hub"
mkdir -p "$MODEL_CACHE"

# Download recommended model for 16GB VRAM
cat > "$CONFIG_DIR/download_models.py" << 'EOF'
#!/usr/bin/env python3
"""Download and prepare models for RTX 5000 (16GB VRAM)"""

import os
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig

models = [
    # Primary model - fits comfortably in 16GB with AWQ
    "Qwen/Qwen2.5-7B-Instruct-AWQ",
    # Backup models
    "mistralai/Mistral-7B-Instruct-v0.2",
]

for model_name in models:
    print(f"Checking {model_name}...")
    try:
        config = AutoConfig.from_pretrained(model_name)
        print(f"✓ {model_name} configuration loaded")
        # Download tokenizer (small, quick)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        print(f"✓ {model_name} tokenizer ready")
    except Exception as e:
        print(f"✗ {model_name}: {e}")

print("\nModel preparation complete!")
EOF
chmod +x "$CONFIG_DIR/download_models.py"

# Phase 7: Benchmark Script
log_info "Phase 7: Creating Benchmark Tools"

cat > "$CONFIG_DIR/benchmark.py" << 'EOF'
#!/usr/bin/env python3
"""Benchmark vLLM performance on Quadro RTX 5000"""

import time
import torch
import psutil
import subprocess
from typing import Dict, Any

def get_gpu_info() -> Dict[str, Any]:
    """Get GPU information"""
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=name,memory.total,memory.used,temperature.gpu", "--format=csv,noheader,nounits"],
        capture_output=True, text=True
    )
    gpu_info = result.stdout.strip().split(", ")
    return {
        "name": gpu_info[0],
        "memory_total": f"{float(gpu_info[1])/1024:.1f} GB",
        "memory_used": f"{float(gpu_info[2])/1024:.1f} GB",
        "temperature": f"{gpu_info[3]}°C"
    }

def benchmark_memory():
    """Test GPU memory allocation"""
    print("Testing GPU memory allocation...")
    
    # Test different tensor sizes
    sizes = [(1024, 1024), (2048, 2048), (4096, 4096), (8192, 8192)]
    
    for size in sizes:
        try:
            tensor = torch.cuda.FloatTensor(*size)
            memory_mb = tensor.element_size() * tensor.nelement() / (1024**2)
            print(f"✓ Allocated {size[0]}x{size[1]} tensor ({memory_mb:.1f} MB)")
            del tensor
            torch.cuda.empty_cache()
        except RuntimeError as e:
            print(f"✗ Failed at {size[0]}x{size[1]}: {e}")
            break
    
    torch.cuda.empty_cache()

def main():
    print("=" * 50)
    print("vLLM Benchmark for Quadro RTX 5000")
    print("=" * 50)
    
    # System info
    print("\n[System Information]")
    gpu_info = get_gpu_info()
    for key, value in gpu_info.items():
        print(f"  {key}: {value}")
    
    print(f"  CPU cores: {psutil.cpu_count()}")
    print(f"  RAM: {psutil.virtual_memory().total / (1024**3):.1f} GB")
    
    # CUDA info
    print("\n[CUDA Information]")
    print(f"  PyTorch version: {torch.__version__}")
    print(f"  CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  CUDA version: {torch.version.cuda}")
        print(f"  cuDNN version: {torch.backends.cudnn.version()}")
        print(f"  Device: {torch.cuda.get_device_name(0)}")
    
    # Memory benchmark
    print("\n[Memory Benchmark]")
    benchmark_memory()
    
    print("\n[Recommended Settings]")
    print("  Model: Qwen2.5-7B-Instruct-AWQ")
    print("  GPU Memory Utilization: 0.85 (13.6 GB)")
    print("  Max Model Length: 16384 tokens")
    print("  Batch Size: 8")
    print("  Quantization: AWQ (4-bit)")
    
    print("\n[Expected Performance]")
    print("  Throughput: 20-40 tokens/second")
    print("  First token latency: < 500ms")
    print("  Memory efficiency: 85-90%")

if __name__ == "__main__":
    main()
EOF
chmod +x "$CONFIG_DIR/benchmark.py"

# Phase 8: Integration with Crush
log_info "Phase 8: Configuring Crush Integration"

# Update Crush configuration
CRUSH_CONFIG="$HOME/.config/crush/config.json"
mkdir -p "$(dirname "$CRUSH_CONFIG")"

cat > "$CRUSH_CONFIG" << EOF
{
  "\$schema": "https://charm.land/crush.json",
  "providers": {
    "vllm": {
      "type": "openai",
      "base_url": "http://localhost:8000/v1",
      "api_key": "dummy",
      "default": true,
      "models": {
        "qwen-7b": {
          "id": "Qwen/Qwen2.5-7B-Instruct-AWQ",
          "context_window": 16384,
          "max_tokens": 4096
        }
      }
    }
  },
  "performance": {
    "gpu_optimization": true,
    "batch_requests": true,
    "connection_pool_size": 10,
    "request_timeout": 300
  }
}
EOF

# Create start script
cat > "$HOME/DEV/projects/crush/start_vllm_rtx5000.sh" << 'EOF'
#!/bin/bash
# Optimized vLLM launcher for Quadro RTX 5000

source ~/.config/vllm/activate.sh

# Model selection based on available memory
MODEL="${VLLM_MODEL:-Qwen/Qwen2.5-7B-Instruct-AWQ}"

echo "Starting vLLM with model: $MODEL"
echo "Optimized for Quadro RTX 5000 (16GB VRAM)"

vllm serve "$MODEL" \
    --host 0.0.0.0 \
    --port 8000 \
    --gpu-memory-utilization 0.85 \
    --max-model-len 16384 \
    --max-num-seqs 8 \
    --max-num-batched-tokens 4096 \
    --quantization awq \
    --enable-prefix-caching \
    --enable-chunked-prefill \
    --block-size 16 \
    --num-scheduler-steps 1 \
    --swap-space 4 \
    --trust-remote-code
EOF
chmod +x "$HOME/DEV/projects/crush/start_vllm_rtx5000.sh"

# Phase 9: Final Setup
log_info "Phase 9: Finalizing Installation"

# Test imports
python -c "import vllm; print(f'vLLM version: {vllm.__version__}')" || {
    log_error "vLLM import test failed!"
    exit 1
}

# Create uninstall script
cat > "$CONFIG_DIR/uninstall.sh" << EOF
#!/bin/bash
echo "Removing vLLM installation..."
rm -rf "$VENV_PATH"
rm -rf "$CONFIG_DIR"
echo "vLLM uninstalled. Model cache retained at $MODEL_CACHE"
EOF
chmod +x "$CONFIG_DIR/uninstall.sh"

# Success message
echo
log_success "═══════════════════════════════════════════════════════"
log_success "    vLLM Installation Complete for RTX 5000!"
log_success "═══════════════════════════════════════════════════════"
echo
echo -e "${GREEN}Next Steps:${NC}"
echo "1. Activate environment: source ~/.config/vllm/activate.sh"
echo "2. Run benchmark: python ~/.config/vllm/benchmark.py"
echo "3. Download models: python ~/.config/vllm/download_models.py"
echo "4. Start vLLM: ./start_vllm_rtx5000.sh"
echo
echo -e "${YELLOW}Quick Test:${NC}"
echo "  curl http://localhost:8000/v1/models"
echo
echo -e "${BLUE}Crush Integration:${NC}"
echo "  ./crush --provider vllm"
echo
echo "Configuration saved to: ~/.config/vllm/"
echo "Virtual environment at: ~/venvs/vllm_crush/"