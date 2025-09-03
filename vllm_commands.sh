#!/bin/bash
# vLLM Management Commands for Quadro RTX 5000

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Helper function to run commands in vllm environment
vllm_run() {
    micromamba run -n vllm "$@"
}

case "$1" in
    "test")
        echo -e "${BLUE}🧪 Testing vLLM installation...${NC}"
        vllm_run python /home/miko/DEV/projects/crush/test_vllm.py
        ;;
    
    "server")
        MODEL=${2:-"microsoft/DialoGPT-medium"}
        PORT=${3:-8000}
        echo -e "${GREEN}🚀 Starting vLLM server with model: $MODEL${NC}"
        echo -e "${YELLOW}Server will be available at: http://localhost:$PORT${NC}"
        echo -e "${YELLOW}API endpoint: http://localhost:$PORT/v1${NC}"
        echo -e "${YELLOW}Press Ctrl+C to stop the server${NC}"
        echo ""
        vllm_run python /home/miko/DEV/projects/crush/vllm_server.py --model "$MODEL" --port "$PORT"
        ;;
    
    "models")
        echo -e "${BLUE}📚 Recommended models for RTX 5000 (16GB):${NC}"
        echo ""
        echo -e "${GREEN}Small models (< 4GB):${NC}"
        echo "  • microsoft/DialoGPT-small"
        echo "  • microsoft/DialoGPT-medium"
        echo "  • distilgpt2"
        echo "  • gpt2"
        echo ""
        echo -e "${YELLOW}Medium models (4-8GB):${NC}"
        echo "  • microsoft/DialoGPT-large"
        echo "  • EleutherAI/gpt-neo-1.3B"
        echo "  • facebook/opt-1.3b"
        echo ""
        echo -e "${RED}Large models (8-16GB - use carefully):${NC}"
        echo "  • EleutherAI/gpt-neo-2.7B"
        echo "  • facebook/opt-2.7b"
        echo "  • bigscience/bloom-3b"
        echo ""
        echo -e "${BLUE}Usage: ./vllm_commands.sh server MODEL_NAME${NC}"
        ;;
    
    "status")
        echo -e "${BLUE}📊 vLLM Environment Status:${NC}"
        echo ""
        vllm_run python -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')
import vllm
print(f'vLLM: {vllm.__version__}')
        "
        ;;
    
    "client")
        PORT=${2:-8000}
        echo -e "${BLUE}🔗 Testing API client connection...${NC}"
        vllm_run python -c "
import requests
import json

try:
    response = requests.get('http://localhost:$PORT/v1/models')
    if response.status_code == 200:
        models = response.json()
        print('✅ Server is running!')
        print('Available models:')
        for model in models.get('data', []):
            print(f'  • {model[\"id\"]}')
    else:
        print(f'❌ Server error: {response.status_code}')
except Exception as e:
    print(f'❌ Connection failed: {e}')
    print('Make sure the server is running with: ./vllm_commands.sh server')
        "
        ;;
    
    "benchmark")
        MODEL=${2:-"microsoft/DialoGPT-medium"}
        echo -e "${BLUE}⚡ Benchmarking vLLM with model: $MODEL${NC}"
        vllm_run python -c "
import time
from vllm import LLM, SamplingParams

print('Loading model...')
start_time = time.time()
llm = LLM(model='$MODEL', gpu_memory_utilization=0.8, max_model_len=512)
load_time = time.time() - start_time
print(f'Model loaded in {load_time:.2f} seconds')

prompts = ['Hello, how are you today?'] * 5
sampling_params = SamplingParams(temperature=0.8, max_tokens=50)

print('Generating responses...')
start_time = time.time()
outputs = llm.generate(prompts, sampling_params)
gen_time = time.time() - start_time

print(f'Generated {len(outputs)} responses in {gen_time:.2f} seconds')
print(f'Speed: {len(outputs)/gen_time:.2f} requests/second')

for i, output in enumerate(outputs[:2]):
    print(f'Sample {i+1}: {output.outputs[0].text[:100]}...')
        "
        ;;
    
    "activate")
        echo -e "${GREEN}🐍 To activate the vLLM environment, run:${NC}"
        echo -e "${YELLOW}micromamba activate vllm${NC}"
        ;;
    
    "install")
        echo -e "${RED}⚠️  Installation already completed!${NC}"
        echo -e "${BLUE}Use './vllm_commands.sh status' to check installation${NC}"
        ;;
    
    *)
        echo -e "${BLUE}🦙 vLLM Management Commands${NC}"
        echo ""
        echo -e "${GREEN}Usage: ./vllm_commands.sh [command] [options]${NC}"
        echo ""
        echo -e "${YELLOW}Commands:${NC}"
        echo "  test           - Test vLLM installation"
        echo "  server [model] - Start API server (default: DialoGPT-medium)"
        echo "  models         - Show recommended models for RTX 5000"
        echo "  status         - Show environment status"
        echo "  client [port]  - Test API client connection"
        echo "  benchmark [model] - Benchmark model performance"
        echo "  activate       - Show activation command"
        echo ""
        echo -e "${YELLOW}Examples:${NC}"
        echo "  ./vllm_commands.sh test"
        echo "  ./vllm_commands.sh server microsoft/DialoGPT-medium"
        echo "  ./vllm_commands.sh benchmark gpt2"
        echo ""
        echo -e "${BLUE}For OpenAI-compatible API usage:${NC}"
        echo "  curl http://localhost:8000/v1/models"
        ;;
esac