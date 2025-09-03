#!/bin/bash

# Simple vLLM Setup for Quadro RTX 5000
# Uses Python 3.12 to avoid compatibility issues

set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${GREEN}═══════════════════════════════════════════${NC}"
echo -e "${GREEN}  Simple vLLM Setup (Python 3.12)${NC}"
echo -e "${GREEN}═══════════════════════════════════════════${NC}"

# Step 1: Create Python 3.12 virtual environment
echo -e "\n${BLUE}[1/5]${NC} Creating Python 3.12 environment..."
python3.12 -m venv ~/venvs/vllm_py312

# Step 2: Activate and upgrade pip
echo -e "${BLUE}[2/5]${NC} Upgrading pip..."
source ~/venvs/vllm_py312/bin/activate
pip install --upgrade pip setuptools wheel

# Step 3: Install PyTorch with CUDA support
echo -e "${BLUE}[3/5]${NC} Installing PyTorch with CUDA 12.1..."
pip install torch --index-url https://download.pytorch.org/whl/cu121

# Step 4: Install vLLM
echo -e "${BLUE}[4/5]${NC} Installing vLLM..."
pip install vllm

# Step 5: Create activation and start scripts
echo -e "${BLUE}[5/5]${NC} Creating helper scripts..."

# Create activation script
cat > ~/venvs/vllm_py312/activate_vllm.sh << 'EOF'
#!/bin/bash
source ~/venvs/vllm_py312/bin/activate
export CUDA_VISIBLE_DEVICES=0
export VLLM_ATTENTION_BACKEND=FLASH_ATTN
echo "vLLM environment activated (Python 3.12)"
echo "Run: vllm serve <model_name> to start server"
EOF
chmod +x ~/venvs/vllm_py312/activate_vllm.sh

# Create quick start script
cat > ./start_vllm_server.sh << 'EOF'
#!/bin/bash
source ~/venvs/vllm_py312/bin/activate
export CUDA_VISIBLE_DEVICES=0

MODEL="${1:-microsoft/Phi-3-mini-4k-instruct}"
echo "Starting vLLM with model: $MODEL"

vllm serve "$MODEL" \
    --host 0.0.0.0 \
    --port 8000 \
    --gpu-memory-utilization 0.85 \
    --max-model-len 4096 \
    --trust-remote-code
EOF
chmod +x ./start_vllm_server.sh

# Success message
echo -e "\n${GREEN}═══════════════════════════════════════════${NC}"
echo -e "${GREEN}✅ vLLM Installation Complete!${NC}"
echo -e "${GREEN}═══════════════════════════════════════════${NC}"
echo
echo -e "${YELLOW}Quick Start:${NC}"
echo "1. Activate: source ~/venvs/vllm_py312/activate_vllm.sh"
echo "2. Start server: ./start_vllm_server.sh"
echo "3. Test: curl http://localhost:8000/v1/models"
echo
echo -e "${BLUE}Recommended models for 16GB VRAM:${NC}"
echo "• microsoft/Phi-3-mini-4k-instruct (3.8B)"
echo "• mistralai/Mistral-7B-Instruct-v0.2 (7B)"
echo "• meta-llama/Llama-2-7b-chat-hf (7B)"