#!/bin/bash

# Setup vLLM models in Crush configuration
CONFIG_FILE="$HOME/.config/crush/config.json"

# Ensure config directory exists
mkdir -p "$(dirname "$CONFIG_FILE")"

# Create or update the configuration with vLLM models
cat > "$CONFIG_FILE" << 'EOF'
{
  "providers": {
    "vllm": {
      "type": "openai",
      "base_url": "http://localhost:8000/v1",
      "api_key": "dummy",
      "name": "vLLM Local",
      "models": [
        {
          "id": "facebook/opt-125m",
          "name": "OPT 125M (Fast Testing)",
          "contextWindow": 2048,
          "defaultMaxTokens": 512
        },
        {
          "id": "meta-llama/Llama-3.2-1B-Instruct",
          "name": "Llama 3.2 1B Instruct",
          "contextWindow": 8192,
          "defaultMaxTokens": 2048
        },
        {
          "id": "Qwen/Qwen2.5-Coder-3B-Instruct",
          "name": "Qwen2.5 Coder 3B",
          "contextWindow": 32768,
          "defaultMaxTokens": 4096
        },
        {
          "id": "Qwen/Qwen2.5-Coder-7B-Instruct",
          "name": "Qwen2.5 Coder 7B",
          "contextWindow": 32768,
          "defaultMaxTokens": 4096
        },
        {
          "id": "Qwen/Qwen2.5-14B-Instruct",
          "name": "Qwen2.5 14B Instruct",
          "contextWindow": 32768,
          "defaultMaxTokens": 4096
        },
        {
          "id": "deepseek-ai/deepseek-coder-6.7b-instruct",
          "name": "DeepSeek Coder 6.7B",
          "contextWindow": 16384,
          "defaultMaxTokens": 4096
        },
        {
          "id": "mistralai/Mistral-7B-Instruct-v0.2",
          "name": "Mistral 7B Instruct v0.2",
          "contextWindow": 32768,
          "defaultMaxTokens": 4096
        }
      ]
    }
  },
  "models": {
    "large": {
      "provider": "vllm",
      "model": "Qwen/Qwen2.5-Coder-7B-Instruct"
    },
    "small": {
      "provider": "vllm",
      "model": "facebook/opt-125m"
    }
  }
}
EOF

echo "✓ vLLM models configured in Crush"
echo ""
echo "Available models:"
echo "  • OPT 125M (Fast Testing)"
echo "  • Llama 3.2 1B Instruct"
echo "  • Qwen2.5 Coder 3B"
echo "  • Qwen2.5 Coder 7B"
echo "  • Qwen2.5 14B Instruct"
echo "  • DeepSeek Coder 6.7B"
echo "  • Mistral 7B Instruct v0.2"
echo ""
echo "Default large model: Qwen2.5 Coder 7B"
echo "Default small model: OPT 125M"
echo ""
echo "To switch models in Crush, press ⌘+M or use the model switcher"