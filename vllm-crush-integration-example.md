# vLLM + Crush Integration Example

This document provides a complete example of how to integrate vLLM with Crush CLI using the MCP (Model Context Protocol) framework.

## Setup and Configuration

### 1. Install MCP Servers

```bash
# Install the vLLM MCP server packages
npm install -g vllm-mcp-server@latest
npm install -g vllm-state-mcp@latest  
npm install -g vllm-monitor-mcp@latest
npm install -g vllm-discovery-mcp@latest
npm install -g vllm-optimizer-mcp@latest
```

### 2. Configure Crush for vLLM

Add the vLLM configuration to your `crush.json`:

```json
{
  "$schema": "https://charm.land/crush.json",
  "providers": {
    "vllm": {
      "name": "vLLM Local",
      "base_url": "http://localhost:8000/v1/",
      "type": "openai",
      "auto_discovery": true,
      "health_check": {
        "enabled": true,
        "interval": 30000,
        "timeout": 5000
      },
      "models": []
    }
  },
  "mcp": {
    "vllm-model-manager": {
      "type": "stdio",
      "command": "npx",
      "args": ["vllm-mcp-server", "--type", "model-manager"],
      "env": {
        "VLLM_API_ENDPOINT": "http://localhost:8000/v1",
        "VLLM_TIMEOUT": "30000",
        "LOG_LEVEL": "info"
      }
    },
    "vllm-state-persistence": {
      "type": "stdio", 
      "command": "npx",
      "args": ["vllm-state-mcp", "--type", "state-persistence"],
      "env": {
        "STATE_DB_PATH": "./.crush/vllm-state.db",
        "ENABLE_COMPRESSION": "true"
      }
    },
    "vllm-performance-monitor": {
      "type": "stdio",
      "command": "npx", 
      "args": ["vllm-monitor-mcp", "--type", "performance-monitor"],
      "env": {
        "MONITOR_INTERVAL_MS": "5000",
        "ENABLE_GPU_MONITORING": "true"
      }
    }
  },
  "permissions": {
    "allowed_tools": [
      "mcp_vllm-model-manager_list-models",
      "mcp_vllm-model-manager_load-model",
      "mcp_vllm-model-manager_unload-model",
      "mcp_vllm-model-manager_get-model-info",
      "mcp_vllm-state-persistence_save-state",
      "mcp_vllm-state-persistence_load-state", 
      "mcp_vllm-performance-monitor_get-metrics",
      "mcp_vllm-performance-monitor_get-gpu-stats"
    ]
  },
  "options": {
    "debug": true,
    "vllm_auto_discovery": true,
    "vllm_health_check_interval": 30000
  }
}
```

### 3. Start vLLM Server

```bash
# Start vLLM with a model (example with Llama 3.1 8B)
python -m vllm.entrypoints.openai.api_server \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --host 0.0.0.0 \
  --port 8000 \
  --tensor-parallel-size 1 \
  --gpu-memory-utilization 0.8 \
  --enable-prefix-caching \
  --served-model-name llama-3.1-8b
```

## Usage Examples

### Basic Model Management

```bash
# Start Crush with vLLM integration
crush

# List available models
crush --tool mcp_vllm-model-manager_list-models

# Load a specific model
crush --tool mcp_vllm-model-manager_load-model \
  --model-id meta-llama/Llama-3.1-70B-Instruct \
  --tensor-parallel-size 4 \
  --gpu-memory-utilization 0.8

# Get model information
crush --tool mcp_vllm-model-manager_get-model-info \
  --model-id meta-llama/Llama-3.1-70B-Instruct \
  --include-performance true

# Unload a model to free resources
crush --tool mcp_vllm-model-manager_unload-model \
  --model-id meta-llama/Llama-3.1-8B-Instruct
```

### State Management

```bash
# Save current session state
crush --tool mcp_vllm-state-persistence_save-state \
  --session-id "coding-session-2024" \
  --include-models true \
  --include-conversations true \
  --description "Before switching to larger model"

# Load previous state
crush --tool mcp_vllm-state-persistence_load-state \
  --session-id "coding-session-2024" \
  --restore-models true \
  --restore-conversations true
```

### Performance Monitoring

```bash
# Get current performance metrics
crush --tool mcp_vllm-performance-monitor_get-metrics \
  --duration-seconds 300 \
  --metrics throughput,latency,memory,gpu

# Check GPU statistics
crush --tool mcp_vllm-performance-monitor_get-gpu-stats \
  --include-temperature true \
  --include-power true
```

## Interactive Usage Scenarios

### Scenario 1: Model Switching During Conversation

```bash
# Start with a smaller model for quick responses
crush "I need help with Python code optimization"

# Check current performance
User: "The responses are fast but not detailed enough. Can we use a larger model?"

# Claude responds with model switching
Assistant: "I'll switch to a larger model for more detailed analysis. Let me check available models and system resources."

# Tools used automatically:
# 1. mcp_vllm-performance-monitor_get-metrics
# 2. mcp_vllm-model-manager_list-models  
# 3. mcp_vllm-state-persistence_save-state
# 4. mcp_vllm-model-manager_load-model (larger model)
# 5. mcp_vllm-model-manager_unload-model (smaller model)
```

### Scenario 2: Resource Optimization

```bash
# Start conversation
crush "I'm working on a machine learning project and need help with model architecture"

# After some conversation, performance degrades
User: "The responses are getting slower. Can you check what's happening?"