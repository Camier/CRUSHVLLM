#!/bin/bash
# vLLM Quick Setup Script for RTX 5000
# This creates a Python 3.11 environment and installs vLLM with CUDA support

set -e

echo "🚀 Setting up vLLM with Python 3.11..."

# Create Python 3.11 environment with micromamba
echo "📦 Creating Python 3.11 environment..."
micromamba create -n vllm python=3.11 -y

# Activate the environment
echo "🔧 Activating environment..."
eval "$(micromamba shell hook --shell=bash)"
micromamba activate vllm

# Install PyTorch with CUDA 12.1 (compatible with CUDA 12.9)
echo "⚡ Installing PyTorch with CUDA support..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install vLLM
echo "🦙 Installing vLLM..."
pip install vllm

# Verify installation
echo "✅ Verifying installation..."
python -c "import torch; print(f'PyTorch version: {torch.__version__}')"
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
python -c "import torch; print(f'GPU count: {torch.cuda.device_count()}')"
python -c "import vllm; print(f'vLLM version: {vllm.__version__}')"

echo "🎉 vLLM setup complete!"
echo ""
echo "To use vLLM:"
echo "1. Activate environment: micromamba activate vllm"
echo "2. Run: python -m vllm.entrypoints.openai.api_server --model microsoft/DialoGPT-medium"
echo ""
echo "Or start with a simple test:"
echo "python -c \"from vllm import LLM; llm = LLM('microsoft/DialoGPT-medium'); print('✅ vLLM working!')\""