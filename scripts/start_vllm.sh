#!/bin/bash

# Start vLLM server with optimal settings for local inference
# This script configures vLLM for maximum performance on local GPU

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Default values
MODEL_NAME="${VLLM_MODEL:-Qwen/Qwen2.5-Coder-32B-Instruct-AWQ}"
PORT="${VLLM_PORT:-8000}"
GPU_MEMORY="${VLLM_GPU_MEMORY:-0.9}"
MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-32768}"
TENSOR_PARALLEL="${VLLM_TENSOR_PARALLEL:-1}"
QUANTIZATION="${VLLM_QUANTIZATION:-awq}"

echo -e "${GREEN}Starting vLLM Server${NC}"
echo "================================"

# Check for GPU
if ! command -v nvidia-smi &> /dev/null; then
    echo -e "${RED}Error: nvidia-smi not found. GPU required for vLLM${NC}"
    exit 1
fi

# Display GPU info
echo -e "${YELLOW}GPU Information:${NC}"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader

# Check if vLLM is installed
if ! python -c "import vllm" 2>/dev/null; then
    echo -e "${YELLOW}vLLM not installed. Installing...${NC}"
    pip install vllm
fi

# Detect optimal settings based on GPU
GPU_MEM_GB=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -n1 | awk '{print int($1/1024)}')
echo -e "${GREEN}Detected GPU memory: ${GPU_MEM_GB}GB${NC}"

# Adjust settings based on GPU memory
if [ $GPU_MEM_GB -lt 16 ]; then
    echo -e "${YELLOW}Limited GPU memory detected. Using reduced settings.${NC}"
    MAX_MODEL_LEN=8192
    GPU_MEMORY=0.85
    QUANTIZATION="gptq"  # More aggressive quantization
elif [ $GPU_MEM_GB -lt 24 ]; then
    echo -e "${YELLOW}Medium GPU memory detected. Using balanced settings.${NC}"
    MAX_MODEL_LEN=16384
    GPU_MEMORY=0.9
fi

# Check for multiple GPUs
GPU_COUNT=$(nvidia-smi --query-gpu=count --format=csv,noheader | wc -l)
if [ $GPU_COUNT -gt 1 ]; then
    echo -e "${GREEN}Multiple GPUs detected: $GPU_COUNT${NC}"
    read -p "Use tensor parallelism? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        TENSOR_PARALLEL=$GPU_COUNT
    fi
fi

# Create model cache directory
CACHE_DIR="${HOME}/.cache/vllm"
mkdir -p "$CACHE_DIR"

# Start vLLM server
echo -e "${GREEN}Starting vLLM with:${NC}"
echo "  Model: $MODEL_NAME"
echo "  Port: $PORT"
echo "  GPU Memory: $GPU_MEMORY"
echo "  Max Model Length: $MAX_MODEL_LEN"
echo "  Tensor Parallel: $TENSOR_PARALLEL"
echo "  Quantization: $QUANTIZATION"
echo ""

# Export environment variables for performance
export VLLM_USE_DUMMY_WEIGHTS=false
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export CUDA_VISIBLE_DEVICES=0  # Adjust if using specific GPUs

# Build vLLM command
VLLM_CMD="python -m vllm.entrypoints.openai.api_server"
VLLM_ARGS=(
    "--model" "$MODEL_NAME"
    "--port" "$PORT"
    "--gpu-memory-utilization" "$GPU_MEMORY"
    "--max-model-len" "$MAX_MODEL_LEN"
    "--tensor-parallel-size" "$TENSOR_PARALLEL"
    "--trust-remote-code"
    "--enable-prefix-caching"
    "--disable-log-requests"
    "--max-num-seqs" "256"
    "--max-num-batched-tokens" "8192"
)

# Add quantization if specified
if [ "$QUANTIZATION" != "none" ]; then
    VLLM_ARGS+=("--quantization" "$QUANTIZATION")
fi

# Add performance optimizations
VLLM_ARGS+=(
    "--enforce-eager" "false"  # Use CUDA graphs
    "--enable-chunked-prefill"
    "--max-parallel-loading-workers" "4"
)

# Function to handle shutdown
cleanup() {
    echo -e "\n${YELLOW}Shutting down vLLM server...${NC}"
    kill $VLLM_PID 2>/dev/null || true
    exit 0
}

trap cleanup SIGINT SIGTERM

# Start vLLM in background
echo -e "${GREEN}Launching vLLM server...${NC}"
$VLLM_CMD "${VLLM_ARGS[@]}" &
VLLM_PID=$!

# Wait for server to start
echo -e "${YELLOW}Waiting for server to start...${NC}"
for i in {1..30}; do
    if curl -s "http://localhost:$PORT/health" > /dev/null 2>&1; then
        echo -e "${GREEN}✓ vLLM server is running at http://localhost:$PORT${NC}"
        echo -e "${GREEN}✓ OpenAI-compatible API at http://localhost:$PORT/v1${NC}"
        break
    fi
    sleep 1
    echo -n "."
done

# Display usage instructions
echo ""
echo -e "${GREEN}Usage with Crush:${NC}"
echo "1. Update your crush.json to include the vLLM provider:"
echo '   {
     "providers": {
       "vllm": {
         "type": "openai",
         "base_url": "http://localhost:'$PORT'/v1"
       }
     }
   }'
echo ""
echo "2. Start Crush and select the vLLM provider"
echo ""
echo -e "${YELLOW}Server Logs:${NC}"
echo "================================"

# Follow server logs
wait $VLLM_PID