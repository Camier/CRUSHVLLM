# vLLM Quick Start Guide 🚀

**Status**: ✅ **WORKING** - vLLM successfully installed and tested!

## What Was Done

1. **Created Python 3.11 Environment**: Used micromamba to create a Python 3.11 environment (vLLM requires numpy<2.0 which doesn't support Python 3.13)
2. **Installed PyTorch with CUDA**: PyTorch 2.7.1 with CUDA 12.6 support
3. **Installed vLLM**: Version 0.10.1.1 with full GPU acceleration
4. **Verified GPU Support**: Quadro RTX 5000 (16GB) detected and working

## Verified Setup
- **GPU**: Quadro RTX 5000 (16.7 GB VRAM)
- **CUDA**: Available and working
- **vLLM**: Successfully tested with microsoft/DialoGPT-small model
- **Performance**: ~126 tokens/second output speed

## Quick Commands

```bash
# Test installation
./vllm_commands.sh test

# Start API server
./vllm_commands.sh server

# Check status
./vllm_commands.sh status

# See all commands
./vllm_commands.sh
```

## Usage Examples

### 1. Start vLLM Server (OpenAI Compatible)
```bash
./vllm_commands.sh server microsoft/DialoGPT-medium
```
- Server runs on: http://localhost:8000
- API endpoint: http://localhost:8000/v1

### 2. Test with curl
```bash
# List available models
curl http://localhost:8000/v1/models

# Generate text
curl -X POST http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "microsoft/DialoGPT-medium", 
    "prompt": "Hello, how are you?",
    "max_tokens": 50
  }'
```

### 3. Python Client Example
```python
import requests

# Chat completion
response = requests.post("http://localhost:8000/v1/chat/completions", 
    json={
        "model": "microsoft/DialoGPT-medium",
        "messages": [{"role": "user", "content": "Hello!"}],
        "max_tokens": 50
    })

print(response.json())
```

### 4. Use with OpenAI Python Client
```python
from openai import OpenAI

# Point to vLLM server
client = OpenAI(
    api_key="dummy-key",  # vLLM doesn't require real API key
    base_url="http://localhost:8000/v1"
)

response = client.chat.completions.create(
    model="microsoft/DialoGPT-medium",
    messages=[{"role": "user", "content": "Hello!"}],
    max_tokens=50
)

print(response.choices[0].message.content)
```

## Recommended Models for RTX 5000 (16GB)

### Small Models (< 4GB) - Fast, reliable
- `microsoft/DialoGPT-small` - Good for testing
- `microsoft/DialoGPT-medium` - Recommended starting point
- `distilgpt2` - Compact and efficient
- `gpt2` - Classic choice

### Medium Models (4-8GB) - Better quality
- `microsoft/DialoGPT-large` - Good balance
- `EleutherAI/gpt-neo-1.3B` - Higher quality
- `facebook/opt-1.3b` - Facebook's model

### Large Models (8-16GB) - Use with caution
- `EleutherAI/gpt-neo-2.7B` - High quality but memory intensive
- `facebook/opt-2.7b` - Larger Facebook model

## Performance Tips

1. **GPU Memory**: Default 80% utilization is optimal
2. **Context Length**: Shorter contexts = more concurrency
3. **Batch Size**: vLLM automatically optimizes batching
4. **Model Choice**: Start small, scale up as needed

## Troubleshooting

### If server won't start:
```bash
# Check environment
./vllm_commands.sh status

# Try smaller model
./vllm_commands.sh server microsoft/DialoGPT-small
```

### If out of memory:
- Use smaller model
- Reduce GPU memory utilization: `--gpu-memory 0.6`
- Reduce max context length

### If slow performance:
- Check GPU utilization with `nvidia-smi`
- Ensure CUDA is being used
- Try different model sizes

## Environment Management

```bash
# Activate environment manually
micromamba activate vllm

# Deactivate
micromamba deactivate

# List environments
micromamba env list

# Remove environment (if needed)
micromamba env remove -n vllm
```

## Files Created
- `/home/miko/DEV/projects/crush/vllm_setup.sh` - Installation script
- `/home/miko/DEV/projects/crush/vllm_commands.sh` - Management commands
- `/home/miko/DEV/projects/crush/test_vllm.py` - Test script
- `/home/miko/DEV/projects/crush/vllm_server.py` - Server wrapper
- `/home/miko/DEV/projects/crush/vllm_quickstart.md` - This guide

**Total Setup Time**: ~5 minutes ⚡
**Status**: Ready to use! 🎉