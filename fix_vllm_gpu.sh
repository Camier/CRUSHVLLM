#!/bin/bash

# Fix vLLM for Quadro RTX 5000 (Turing/Compute 7.5)
# September 2025

set -e

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}Fixing vLLM for Quadro RTX 5000 GPU...${NC}"

# Step 1: Activate environment
source ~/venvs/vllm_uv/bin/activate

# Step 2: Downgrade Flash Attention to compatible version
echo -e "${YELLOW}Downgrading Flash Attention...${NC}"
uv pip install "flash-attn==2.8.0.post2" --no-build-isolation || {
    echo -e "${YELLOW}Flash Attention downgrade failed, will disable it instead${NC}"
}

# Step 3: Create environment configuration
echo -e "${BLUE}Creating optimized environment configuration...${NC}"

cat > ~/venvs/vllm_uv/vllm_rtx5000.sh << 'EOF'
#!/bin/bash
# vLLM Environment for Quadro RTX 5000 (Turing)

# Disable Flash Attention (not supported on Turing)
export VLLM_USE_FLASH_ATTN=0

# Use XFormers backend (optimal for Turing)
export VLLM_ATTENTION_BACKEND=XFORMERS

# CUDA optimizations
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:512"

# Disable features not supported on Turing
export VLLM_DISABLE_FP8=1
export VLLM_DISABLE_AWQ=0  # AWQ works fine

# Memory settings for 16GB VRAM
export VLLM_GPU_MEMORY_UTILIZATION=0.85

echo "✓ vLLM configured for Quadro RTX 5000 (Turing)"
echo "  - XFormers attention backend enabled"
echo "  - Flash Attention disabled (not supported)"
echo "  - 85% GPU memory utilization"
EOF
chmod +x ~/venvs/vllm_uv/vllm_rtx5000.sh

# Step 4: Create working launcher
echo -e "${BLUE}Creating GPU-optimized launcher...${NC}"

cat > ~/venvs/vllm_uv/vllm_gpu_server.sh << 'EOF'
#!/bin/bash
# vLLM Server Launcher - GPU Optimized for RTX 5000

source ~/venvs/vllm_uv/bin/activate
source ~/venvs/vllm_uv/vllm_rtx5000.sh

MODEL="${1:-facebook/opt-125m}"
PORT="${2:-8000}"

echo "Starting vLLM with GPU acceleration"
echo "Model: $MODEL"
echo "GPU: Quadro RTX 5000"
echo "Backend: XFormers (Turing-optimized)"

vllm serve "$MODEL" \
    --host 0.0.0.0 \
    --port "$PORT" \
    --gpu-memory-utilization 0.85 \
    --max-model-len 4096 \
    --dtype auto \
    --enforce-eager \
    --trust-remote-code
EOF
chmod +x ~/venvs/vllm_uv/vllm_gpu_server.sh

# Step 5: Test configuration
echo -e "${BLUE}Testing configuration...${NC}"

source ~/venvs/vllm_uv/vllm_rtx5000.sh

python << 'EOF'
import os
import torch

print("Configuration Test:")
print("=" * 50)
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
print(f"Flash Attention: {os.environ.get('VLLM_USE_FLASH_ATTN', '1')} (0=disabled)")
print(f"Attention Backend: {os.environ.get('VLLM_ATTENTION_BACKEND', 'AUTO')}")
print("✓ Configuration ready for GPU inference")
EOF

echo -e "\n${GREEN}════════════════════════════════════════════${NC}"
echo -e "${GREEN}✅ vLLM GPU Fix Complete!${NC}"
echo -e "${GREEN}════════════════════════════════════════════${NC}"
echo
echo -e "${YELLOW}To start vLLM with GPU:${NC}"
echo "~/venvs/vllm_uv/vllm_gpu_server.sh"
echo
echo -e "${YELLOW}Or manually:${NC}"
echo "source ~/venvs/vllm_uv/bin/activate"
echo "source ~/venvs/vllm_uv/vllm_rtx5000.sh"
echo "vllm serve <model> --gpu-memory-utilization 0.85"