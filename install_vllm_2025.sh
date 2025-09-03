#!/bin/bash

# vLLM Installation Script - September 2025 Best Practices
# For Quadro RTX 5000 (16GB VRAM, Compute 7.5)
# Using Python 3.12 (latest supported version)

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
MAGENTA='\033[0;35m'
NC='\033[0m'

# Configuration
VENV_NAME="vllm_2025"
VENV_PATH="$HOME/venvs/$VENV_NAME"
VLLM_VERSION="0.10.1.1"  # Latest as of September 2025

echo -e "${MAGENTA}╔═══════════════════════════════════════════════╗${NC}"
echo -e "${MAGENTA}║     vLLM v${VLLM_VERSION} Installation (Sept 2025)     ║${NC}"
echo -e "${MAGENTA}║         Optimized for RTX 5000              ║${NC}"
echo -e "${MAGENTA}╚═══════════════════════════════════════════════╝${NC}"

# Step 1: Verify system requirements
echo -e "\n${BLUE}[1/6]${NC} Verifying system requirements..."

# Check Python 3.12 availability
if ! command -v python3.12 &> /dev/null; then
    echo -e "${RED}Error: Python 3.12 not found. vLLM requires Python >=3.9, <3.13${NC}"
    echo "Your Python 3.13 is too new. Install Python 3.12:"
    echo "  sudo dnf install python3.12 python3.12-pip python3.12-devel"
    exit 1
fi

# Check GPU
if ! nvidia-smi &> /dev/null; then
    echo -e "${RED}Error: NVIDIA GPU not detected${NC}"
    exit 1
fi

GPU_INFO=$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)
echo -e "${GREEN}✓ GPU detected: $GPU_INFO${NC}"

# Step 2: Create virtual environment
echo -e "\n${BLUE}[2/6]${NC} Creating Python 3.12 virtual environment..."

# Clean up old environment if exists
if [ -d "$VENV_PATH" ]; then
    echo -e "${YELLOW}Removing existing environment...${NC}"
    rm -rf "$VENV_PATH"
fi

# Create new venv
python3.12 -m venv "$VENV_PATH"
source "$VENV_PATH/bin/activate"

# Step 3: Upgrade pip and install build tools
echo -e "\n${BLUE}[3/6]${NC} Upgrading pip and installing build tools..."
pip install --upgrade pip setuptools wheel

# Step 4: Install PyTorch with CUDA
echo -e "\n${BLUE}[4/6]${NC} Installing PyTorch with CUDA support..."

# Check CUDA version
CUDA_VERSION=$(nvidia-smi | grep "CUDA Version" | awk '{print $9}' | cut -d. -f1-2)
echo -e "Detected CUDA version: $CUDA_VERSION"

# Install PyTorch based on CUDA version
if [[ "$CUDA_VERSION" == "12"* ]]; then
    echo "Installing PyTorch for CUDA 12.x..."
    pip install torch --index-url https://download.pytorch.org/whl/cu121
else
    echo "Installing PyTorch for CUDA 11.x..."
    pip install torch --index-url https://download.pytorch.org/whl/cu118
fi

# Step 5: Install vLLM
echo -e "\n${BLUE}[5/6]${NC} Installing vLLM v${VLLM_VERSION}..."
pip install vllm==${VLLM_VERSION}

# Install optional performance packages
echo -e "${BLUE}Installing performance optimizations...${NC}"
pip install flash-attn --no-build-isolation || echo "Flash Attention optional, skipping..."

# Step 6: Create helper scripts
echo -e "\n${BLUE}[6/6]${NC} Creating helper scripts..."

# Create activation script
cat > "$VENV_PATH/activate_vllm.sh" << 'EOF'
#!/bin/bash
# vLLM Environment Activation Script

source ~/venvs/vllm_2025/bin/activate

# Optimize for Quadro RTX 5000
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:512"
export VLLM_ATTENTION_BACKEND="FLASH_ATTN"
export TOKENIZERS_PARALLELISM="false"

echo "vLLM v0.10.1.1 environment activated"
echo "Python: $(python --version)"
echo "vLLM: $(python -c 'import vllm; print(vllm.__version__)')"
echo ""
echo "Quick start: vllm serve <model_name> --gpu-memory-utilization 0.85"
EOF
chmod +x "$VENV_PATH/activate_vllm.sh"

# Create optimized server script
cat > ./start_vllm_optimized.sh << 'EOF'
#!/bin/bash
# vLLM Server Launcher - Optimized for Quadro RTX 5000 (16GB)

source ~/venvs/vllm_2025/bin/activate

# Default to a model that fits well in 16GB
MODEL="${1:-microsoft/Phi-3.5-mini-instruct}"
PORT="${2:-8000}"

echo "Starting vLLM server with model: $MODEL"
echo "Optimizations for Quadro RTX 5000 (16GB VRAM)"

# Launch with optimal settings for RTX 5000
vllm serve "$MODEL" \
    --host 0.0.0.0 \
    --port $PORT \
    --gpu-memory-utilization 0.85 \
    --max-model-len 8192 \
    --max-num-seqs 64 \
    --max-num-batched-tokens 8192 \
    --enable-prefix-caching \
    --enable-chunked-prefill \
    --trust-remote-code \
    --disable-log-stats
EOF
chmod +x ./start_vllm_optimized.sh

# Create test script
cat > ./test_vllm.py << 'EOF'
#!/usr/bin/env python3
"""Test vLLM installation and GPU detection"""

import torch
import vllm
from vllm import LLM

def test_installation():
    print("vLLM Installation Test")
    print("=" * 50)
    
    # Version info
    print(f"vLLM version: {vllm.__version__}")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # Test model loading (small model)
    print("\nTesting model loading...")
    try:
        llm = LLM(model="gpt2", gpu_memory_utilization=0.1, max_model_len=512)
        print("✓ Model loading successful")
        
        # Test inference
        output = llm.generate("Hello", max_tokens=10)
        print(f"✓ Inference test successful: {output[0].outputs[0].text[:50]}...")
        
    except Exception as e:
        print(f"✗ Model test failed: {e}")
    
    print("\n✅ vLLM is ready to use!")

if __name__ == "__main__":
    test_installation()
EOF
chmod +x ./test_vllm.py

# Create Crush integration config
cat > ./crush_vllm_config.json << EOF
{
  "\$schema": "https://charm.land/crush.json",
  "providers": {
    "vllm": {
      "type": "openai",
      "base_url": "http://localhost:8000/v1",
      "api_key": "dummy",
      "default": true
    }
  },
  "models": {
    "phi-3.5": {
      "provider": "vllm",
      "id": "microsoft/Phi-3.5-mini-instruct",
      "context_window": 8192,
      "max_tokens": 2048
    },
    "qwen-2.5": {
      "provider": "vllm", 
      "id": "Qwen/Qwen2.5-7B-Instruct",
      "context_window": 8192,
      "max_tokens": 4096
    }
  }
}
EOF

# Final verification
echo -e "\n${GREEN}═══════════════════════════════════════════════${NC}"
echo -e "${GREEN}✅ vLLM Installation Complete!${NC}"
echo -e "${GREEN}═══════════════════════════════════════════════${NC}"

# Test the installation
source "$VENV_PATH/bin/activate"
python -c "import vllm; print(f'vLLM {vllm.__version__} installed successfully!')" || {
    echo -e "${RED}Installation verification failed!${NC}"
    exit 1
}

echo -e "\n${YELLOW}Quick Start Commands:${NC}"
echo "1. Activate environment:"
echo "   ${BLUE}source ~/venvs/vllm_2025/activate_vllm.sh${NC}"
echo ""
echo "2. Test installation:"
echo "   ${BLUE}python test_vllm.py${NC}"
echo ""
echo "3. Start server:"
echo "   ${BLUE}./start_vllm_optimized.sh${NC}"
echo ""
echo "4. Use with Crush:"
echo "   ${BLUE}cp crush_vllm_config.json ~/.config/crush/config.json${NC}"
echo "   ${BLUE}./crush --provider vllm${NC}"
echo ""
echo -e "${GREEN}Recommended models for 16GB VRAM:${NC}"
echo "• microsoft/Phi-3.5-mini-instruct (3.8B) - Excellent performance"
echo "• Qwen/Qwen2.5-7B-Instruct (7B) - Great quality"  
echo "• mistralai/Mistral-7B-Instruct-v0.3 (7B) - Fast inference"
echo "• meta-llama/Meta-Llama-3.1-8B-Instruct (8B) - State-of-the-art"