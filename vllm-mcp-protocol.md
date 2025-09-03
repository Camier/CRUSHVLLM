# vLLM MCP Protocol Specification

This document defines the Model Context Protocol (MCP) specification for vLLM integration with Crush CLI.

## Overview

The vLLM MCP integration provides three specialized MCP servers:

1. **vLLM Model Manager MCP**: Core model lifecycle management
2. **vLLM State Persistence MCP**: Session and configuration state management  
3. **vLLM Performance Monitor MCP**: Real-time performance monitoring and metrics

## MCP Server: vLLM Model Manager

### Purpose
Manages vLLM model lifecycle, configuration, and deployment operations.

### Tools

#### `list-models`
Lists all available models in the vLLM instance.

**Parameters**: None

**Returns**: 
```json
{
  "models": [
    {
      "id": "meta-llama/Llama-3.1-8B-Instruct",
      "name": "Llama 3.1 8B Instruct", 
      "status": "loaded|unloaded|loading|error",
      "memory_usage_mb": 16384,
      "context_window": 128000,
      "max_tokens": 4096,
      "tensor_parallel_size": 1,
      "pipeline_parallel_size": 1
    }
  ]
}
```

#### `load-model`
Loads a model into vLLM with specified configuration.

**Parameters**:
```json
{
  "model_id": "meta-llama/Llama-3.1-8B-Instruct",
  "tensor_parallel_size": 1,
  "pipeline_parallel_size": 1,
  "gpu_memory_utilization": 0.8,
  "max_model_len": 128000,
  "quantization": "none|awq|gptq",
  "dtype": "auto|half|float16|bfloat16",
  "enforce_eager": false,
  "enable_prefix_caching": true
}
```

**Returns**:
```json
{
  "success": true,
  "model_id": "meta-llama/Llama-3.1-8B-Instruct",
  "status": "loading",
  "estimated_load_time": 120,
  "memory_required_mb": 16384
}
```

#### `unload-model`
Unloads a model from vLLM to free resources.

**Parameters**:
```json
{
  "model_id": "meta-llama/Llama-3.1-8B-Instruct",
  "force": false
}
```

**Returns**:
```json
{
  "success": true,
  "model_id": "meta-llama/Llama-3.1-8B-Instruct", 
  "status": "unloaded",
  "memory_freed_mb": 16384
}
```

#### `get-model-info`
Retrieves detailed information about a specific model.

**Parameters**:
```json
{
  "model_id": "meta-llama/Llama-3.1-8B-Instruct"
}
```

**Returns**:
```json
{
  "model_id": "meta-llama/Llama-3.1-8B-Instruct",
  "name": "Llama 3.1 8B Instruct",
  "status": "loaded",
  "config": {
    "context_window": 128000,
    "max_tokens": 4096,
    "tensor_parallel_size": 1,
    "pipeline_parallel_size": 1,
    "quantization": "none",
    "dtype": "half"
  },
  "performance": {
    "memory_usage_mb": 16384,
    "throughput_tokens_per_second": 150,
    "latency_ms": 45,
    "gpu_utilization": 0.75
  },
  "capabilities": {
    "supports_streaming": true,
    "supports_logprobs": true,
    "supports_multiple_choice": true,
    "supports_function_calling": false
  }
}
```

#### `set-model-config`
Updates runtime configuration for a loaded model.

**Parameters**:
```json
{
  "model_id": "meta-llama/Llama-3.1-8B-Instruct",
  "config": {
    "max_tokens": 8192,
    "temperature": 0.7,
    "top_p": 0.9,
    "top_k": 50,
    "repetition_penalty": 1.1
  }
}
```

**Returns**:
```json
{
  "success": true,
  "model_id": "meta-llama/Llama-3.1-8B-Instruct",
  "updated_config": {
    "max_tokens": 8192,
    "temperature": 0.7,
    "top_p": 0.9,
    "top_k": 50,
    "repetition_penalty": 1.1
  }
}
```

### Resources

#### `model/{model_id}/config`
Provides access to model configuration as a readable resource.

#### `model/{model_id}/logs`
Provides access to model-specific logs and events.

## MCP Server: vLLM State Persistence

### Purpose
Manages persistent storage of vLLM configurations, session states, and conversation history.

### Tools

#### `save-state`
Saves current vLLM state including loaded models and configurations.

**Parameters**:
```json
{
  "session_id": "crush-session-123",
  "include_models": true,
  "include_conversations": true,
  "include_metrics": false,
  "compression": true
}
```

**Returns**:
```json
{
  "success": true,
  "session_id": "crush-session-123",
  "state_id": "state-456789",
  "size_bytes": 1024,
  "timestamp": "2024-03-15T10:30:00Z"
}
```

#### `load-state`
Restores vLLM state from persistent storage.

**Parameters**:
```json
{
  "session_id": "crush-session-123",
  "state_id": "state-456789",
  "restore_models": true,
  "restore_conversations": true
}
```

**Returns**:
```json
{
  "success": true,
  "session_id": "crush-session-123",
  "restored": {
    "models": 2,
    "conversations": 5,
    "configurations": 1
  },
  "timestamp": "2024-03-15T10:30:00Z"
}
```

#### `clear-state`
Clears stored state data with optional filtering.

**Parameters**:
```json
{
  "session_id": "crush-session-123",
  "older_than_hours": 24,
  "include_models": false,
  "include_conversations": true
}
```

**Returns**:
```json
{
  "success": true,
  "cleared_items": 15,
  "space_freed_bytes": 2048
}
```

### Resources

#### `state/sessions`
Lists all available saved sessions.

#### `state/session/{session_id}/summary`
Provides summary of saved session state.

## MCP Server: vLLM Performance Monitor

### Purpose
Monitors vLLM performance metrics, resource utilization, and system health.

### Tools

#### `get-metrics`
Retrieves comprehensive performance metrics.

**Parameters**:
```json
{
  "duration_seconds": 60,
  "include_history": true,
  "metrics": ["throughput", "latency", "memory", "gpu"]
}
```

**Returns**:
```json
{
  "timestamp": "2024-03-15T10:30:00Z",
  "duration_seconds": 60,
  "metrics": {
    "throughput": {
      "tokens_per_second": 150.5,
      "requests_per_second": 12.3,
      "avg_batch_size": 4.2
    },
    "latency": {
      "p50_ms": 45,
      "p95_ms": 120,
      "p99_ms": 200,
      "avg_ms": 52
    },
    "memory": {
      "gpu_used_mb": 14336,
      "gpu_total_mb": 24576,
      "gpu_utilization": 0.58,
      "system_ram_mb": 2048
    },
    "requests": {
      "total": 745,
      "successful": 742,
      "failed": 3,
      "error_rate": 0.004
    }
  }
}
```

#### `get-gpu-stats`
Retrieves detailed GPU utilization statistics.

**Parameters**:
```json
{
  "gpu_ids": [0, 1],
  "include_temperature": true,
  "include_power": true
}
```

**Returns**:
```json
{
  "gpus": [
    {
      "id": 0,
      "name": "NVIDIA A100-SXM4-40GB",
      "memory": {
        "used_mb": 32768,
        "total_mb": 40960,
        "utilization": 0.8
      },
      "compute": {
        "utilization": 0.75,
        "sm_utilization": 0.82,
        "tensor_utilization": 0.68
      },
      "temperature": {
        "current_c": 65,
        "max_c": 83,
        "warning_c": 80
      },
      "power": {
        "current_w": 280,
        "max_w": 400,
        "efficiency": 0.7
      }
    }
  ]
}
```

#### `get-memory-stats`
Retrieves detailed memory usage statistics.

**Parameters**: None

**Returns**:
```json
{
  "gpu_memory": {
    "allocated_mb": 16384,
    "cached_mb": 2048,
    "reserved_mb": 18432,
    "free_mb": 6144,
    "fragmentation_ratio": 0.12
  },
  "system_memory": {
    "process_mb": 4096,
    "available_mb": 28672,
    "swap_mb": 0
  },
  "model_memory": [
    {
      "model_id": "meta-llama/Llama-3.1-8B-Instruct",
      "weights_mb": 14336,
      "kv_cache_mb": 1024,
      "activations_mb": 512
    }
  ]
}
```

### Resources

#### `metrics/current`
Real-time performance metrics as a readable resource.

#### `metrics/history/{duration}`
Historical metrics over specified duration.

#### `alerts/active`
Currently active performance alerts.

## Integration Patterns

### Crush CLI Integration

The vLLM MCP servers integrate with Crush CLI through:

1. **Provider Configuration**: vLLM appears as a custom OpenAI-compatible provider
2. **MCP Tool Access**: All vLLM operations exposed as MCP tools with `mcp_` prefix
3. **State Synchronization**: Session state managed through MCP state persistence
4. **Performance Monitoring**: Real-time metrics accessible via MCP tools

### Example Usage in Crush

```bash
# Load a model via MCP
crush --tool mcp_vllm-model-manager_load-model \
  --model-id meta-llama/Llama-3.1-8B-Instruct \
  --tensor-parallel-size 2

# Check performance metrics
crush --tool mcp_vllm-performance-monitor_get-metrics \
  --duration-seconds 300

# Save current session state
crush --tool mcp_vllm-state-persistence_save-state \
  --session-id current \
  --include-models true
```

### Error Handling

All MCP tools implement consistent error handling:

```json
{
  "success": false,
  "error": {
    "code": "MODEL_LOAD_FAILED",
    "message": "Insufficient GPU memory to load model",
    "details": {
      "required_mb": 32768,
      "available_mb": 16384,
      "suggestion": "Reduce tensor_parallel_size or unload other models"
    }
  }
}
```

### Security Considerations

1. **Authentication**: Optional API key support for vLLM endpoints
2. **Access Control**: Tool permissions managed through Crush CLI configuration
3. **Resource Limits**: Configurable limits on model loading and resource usage
4. **Audit Logging**: All MCP operations logged for security auditing

## Performance Optimizations

### Connection Pooling
- Persistent connections to vLLM API endpoints
- Connection reuse across MCP tool calls
- Automatic connection health monitoring

### Caching
- Model metadata cached locally
- Performance metrics cached with configurable TTL
- State data compressed and cached

### Batch Operations
- Support for bulk model operations
- Batched metric collection
- Efficient state serialization

## Monitoring and Observability

### Health Checks
- Automatic vLLM endpoint health monitoring
- MCP server health status reporting
- Integration with Crush CLI health check system

### Metrics Export
- Prometheus-compatible metrics export
- Custom metrics dashboard integration
- Alert integration with external monitoring systems

### Logging
- Structured logging for all MCP operations
- Debug-level logging for troubleshooting
- Integration with Crush CLI logging system