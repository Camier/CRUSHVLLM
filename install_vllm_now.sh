#!/bin/bash

# Ultra-simple vLLM installation using Python 3.12

set -e

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${GREEN}Installing vLLM with Python 3.12...${NC}"

# Remove old environment if exists
rm -rf ~/venvs/vllm_simple 2>/dev/null || true

# Create new environment without pip (we'll add it manually)
python3.12 -m venv ~/venvs/vllm_simple --without-pip

# Activate environment
source ~/venvs/vllm_simple/bin/activate

# Install pip manually
curl https://bootstrap.pypa.io/get-pip.py | python

# Now install packages
pip install --upgrade pip
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install vllm

echo -e "\n${GREEN}✅ Installation complete!${NC}"
echo -e "${YELLOW}To start vLLM:${NC}"
echo "source ~/venvs/vllm_simple/bin/activate"
echo "vllm serve microsoft/Phi-3-mini-4k-instruct --gpu-memory-utilization 0.85"