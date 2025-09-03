#!/bin/bash

# vLLM Installation with uv - Production Ready
# September 2025 - Optimized for Quadro RTX 5000

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'

echo -e "${CYAN}╔════════════════════════════════════════════╗${NC}"
echo -e "${CYAN}║   vLLM Installation with uv Package Manager  ║${NC}"
echo -e "${CYAN}║        Quadro RTX 5000 Optimized            ║${NC}"
echo -e "${CYAN}╚════════════════════════════════════════════╝${NC}"

# Configuration
VENV_DIR="$HOME/venvs/vllm_uv"
VLLM_VERSION="0.10.1.1"

# Step 1: Check prerequisites
echo -e "\n${BLUE}[1/7]${NC} Checking prerequisites..."

if ! command -v uv &> /dev/null; then
    echo -e "${RED}Error: uv not found${NC}"
    exit 1
fi

if ! nvidia-smi &> /dev/null; then
    echo -e "${RED}Error: NVIDIA GPU not detected${NC}"
    exit 1
fi

GPU_INFO=$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)
echo -e "${GREEN}✓ GPU: $GPU_INFO${NC}"
echo -e "${GREEN}✓ uv: $(uv --version)${NC}"

# Step 2: Clean and create environment
echo -e "\n${BLUE}[2/7]${NC} Creating Python 3.12 environment with uv..."

# Clean up if exists
[ -d "$VENV_DIR" ] && rm -rf "$VENV_DIR"

# Create new environment with Python 3.12
uv venv "$VENV_DIR" --python 3.12 --seed

# Step 3: Activate and install PyTorch
echo -e "\n${BLUE}[3/7]${NC} Installing PyTorch with CUDA support..."

source "$VENV_DIR/bin/activate"

# Install PyTorch first (required by vLLM)
uv pip install torch --index-url https://download.pytorch.org/whl/cu121

# Step 4: Install vLLM
echo -e "\n${BLUE}[4/7]${NC} Installing vLLM v${VLLM_VERSION}..."

uv pip install vllm==${VLLM_VERSION}

# Step 5: Install optional performance packages
echo -e "\n${BLUE}[5/7]${NC} Installing performance optimizations..."

# Try to install flash-attn (may fail on some systems)
uv pip install flash-attn --no-build-isolation 2>/dev/null || echo "Flash Attention optional, skipping..."

# Install additional tools
uv pip install transformers accelerate

# Step 6: Create environment setup script
echo -e "\n${BLUE}[6/7]${NC} Creating environment scripts..."

# RTX 5000 optimized environment
cat > "$VENV_DIR/rtx5000_env.sh" << 'EOF'
#!/bin/bash
# Optimized environment for Quadro RTX 5000

# CUDA settings
export CUDA_VISIBLE_DEVICES=0
export CUDA_LAUNCH_BLOCKING=0

# PyTorch optimizations
export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:512"
export TORCH_CUDA_ARCH_LIST="7.5"  # Quadro RTX 5000 compute capability

# vLLM optimizations
export VLLM_ATTENTION_BACKEND="FLASH_ATTN"
export TOKENIZERS_PARALLELISM="false"

# Memory optimizations
export VLLM_GPU_MEMORY_UTILIZATION="0.85"
export VLLM_SWAP_SPACE_GB="8"

echo "RTX 5000 optimizations loaded"
EOF
chmod +x "$VENV_DIR/rtx5000_env.sh"

# Create server launcher
cat > "$VENV_DIR/vllm-server" << 'EOF'
#!/bin/bash
# vLLM Server Launcher

source ~/venvs/vllm_uv/bin/activate
source ~/venvs/vllm_uv/rtx5000_env.sh

MODEL="${1:-microsoft/DialoGPT-medium}"
PORT="${2:-8000}"

echo "Starting vLLM server"
echo "Model: $MODEL"
echo "Port: $PORT"
echo "GPU: Quadro RTX 5000 (16GB)"

vllm serve "$MODEL" \
    --host 0.0.0.0 \
    --port "$PORT" \
    --gpu-memory-utilization 0.85 \
    --max-model-len 8192 \
    --max-num-seqs 128 \
    --enable-prefix-caching \
    --trust-remote-code \
    --disable-log-stats
EOF
chmod +x "$VENV_DIR/vllm-server"

# Create test script
cat > "$VENV_DIR/test_installation.py" << 'EOF'
#!/usr/bin/env python3
"""Test vLLM installation"""

import sys
import torch

def test_vllm():
    print("Testing vLLM Installation")
    print("=" * 50)
    
    # Check PyTorch
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # Check vLLM
    try:
        import vllm
        print(f"vLLM version: {vllm.__version__}")
        print("✅ vLLM import successful")
        
        # Test basic functionality
        from vllm import LLM, SamplingParams
        print("✅ vLLM components available")
        
    except Exception as e:
        print(f"❌ vLLM test failed: {e}")
        return False
    
    print("\n✅ All tests passed!")
    return True

if __name__ == "__main__":
    success = test_vllm()
    sys.exit(0 if success else 1)
EOF
chmod +x "$VENV_DIR/test_installation.py"

# Step 7: Create Crush integration
echo -e "\n${BLUE}[7/7]${NC} Setting up Crush integration..."

cat > "$VENV_DIR/crush-vllm" << 'EOF'
#!/bin/bash
# Crush + vLLM Integration Helper

COMMAND="$1"
shift

case "$COMMAND" in
    activate)
        echo "source ~/venvs/vllm_uv/bin/activate"
        echo "source ~/venvs/vllm_uv/rtx5000_env.sh"
        ;;
    server)
        ~/venvs/vllm_uv/vllm-server "$@"
        ;;
    test)
        source ~/venvs/vllm_uv/bin/activate
        python ~/venvs/vllm_uv/test_installation.py
        ;;
    *)
        echo "Usage: crush-vllm {activate|server|test}"
        echo "  activate - Print activation commands"
        echo "  server [model] [port] - Start vLLM server"
        echo "  test - Test installation"
        ;;
esac
EOF
chmod +x "$VENV_DIR/crush-vllm"

# Create Crush config
cat > ./crush_vllm.json << EOF
{
  "providers": {
    "vllm": {
      "type": "openai",
      "base_url": "http://localhost:8000/v1",
      "api_key": "dummy"
    }
  }
}
EOF

# Final test
echo -e "\n${BLUE}Testing installation...${NC}"
source "$VENV_DIR/bin/activate"
python "$VENV_DIR/test_installation.py"

# Success message
echo -e "\n${GREEN}════════════════════════════════════════════════${NC}"
echo -e "${GREEN}✅ vLLM Installation Complete!${NC}"
echo -e "${GREEN}════════════════════════════════════════════════${NC}"
echo
echo -e "${YELLOW}Quick Start:${NC}"
echo "1. Activate: source ~/venvs/vllm_uv/bin/activate"
echo "2. Load optimizations: source ~/venvs/vllm_uv/rtx5000_env.sh"  
echo "3. Start server: ~/venvs/vllm_uv/vllm-server"
echo
echo -e "${CYAN}Or use the helper:${NC}"
echo "~/venvs/vllm_uv/crush-vllm server"
echo
echo -e "${GREEN}Recommended models for RTX 5000:${NC}"
echo "• microsoft/DialoGPT-medium (345M)"
echo "• facebook/opt-2.7b (2.7B)"
echo "• mistralai/Mistral-7B-Instruct-v0.3 (7B)"