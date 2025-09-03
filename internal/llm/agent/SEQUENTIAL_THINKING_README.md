# Sequential Thinking Agent for Crush

A sophisticated reasoning agent that implements structured, step-by-step problem-solving optimized for local inference with vLLM.

## Overview

The Sequential Thinking Agent processes complex problems through a series of well-defined thinking stages, similar to chain-of-thought reasoning but with explicit structure and control. It's designed to work seamlessly with Crush's agent system and is optimized for high-performance local inference using vLLM.

## Architecture

### Core Components

1. **Sequential Thinking Engine** (`sequential_thinking.go`)
   - Implements multi-stage reasoning pipeline
   - Supports parallel and sequential stage execution
   - Provides comprehensive metrics and monitoring
   - Integrates with Crush's message and provider systems

2. **Thinking Stages**
   - Problem Decomposition
   - Component Analysis
   - Hypothesis Generation
   - Hypothesis Evaluation
   - Validation Planning
   - Synthesis
   - Reflection

3. **Integration Layer**
   - Wraps as Crush Service interface
   - Compatible with existing agent infrastructure
   - Support for session management
   - Event-driven architecture

## Features

### Adaptive Reasoning Depth

The agent supports 5 levels of reasoning depth:

1. **Quick (Depth 1)**: Basic decomposition and synthesis
2. **Light (Depth 2)**: Adds hypothesis generation
3. **Standard (Depth 3)**: Includes hypothesis evaluation
4. **Deep (Depth 4)**: Adds validation planning
5. **Comprehensive (Depth 5)**: Full pipeline with reflection

### Performance Optimizations for vLLM

- **KV Cache Utilization**: Sequential execution for better cache hits
- **Batch Processing**: Parallel stages when beneficial
- **Token Management**: Adaptive token limits per stage
- **Temperature Control**: Lower temperatures for focused reasoning
- **Memory Efficiency**: Streaming responses to minimize memory usage

### Stage Dependencies

The agent automatically resolves stage dependencies and executes them in the optimal order:

```
problem_decomposition
    ↓
component_analysis
    ↓
hypothesis_generation
    ↓
hypothesis_evaluation
    ↓
validation_planning
    ↓
synthesis ← (all previous stages)
    ↓
reflection
```

## Usage

### Basic Setup

```go
import (
    "context"
    "github.com/charmbracelet/crush/internal/llm/agent"
    "github.com/charmbracelet/crush/internal/config"
)

// Create vLLM provider configuration
vllmConfig := &config.ProviderConfig{
    ID:      "vllm",
    Type:    "openai",
    BaseURL: "http://localhost:8000/v1",
}

// Configure agent
agentConfig := config.Agent{
    ID:    "sequential_thinker",
    Name:  "Sequential Thinking Agent",
    Model: config.SelectedModelTypeLarge,
    ExtraConfig: map[string]any{
        "reasoning_depth": 3,
        "parallel_stages": false,
        "enable_reflection": true,
    },
}

// Create agent
agent, err := agent.NewSequentialThinkingAgent(ctx, agentConfig, vllmConfig)
```

### Thinking Process

```go
// Execute sequential thinking
result, err := agent.Think(ctx, "Explain how to optimize database queries")

// Access results
fmt.Println("Solution:", result.FinalOutput)
fmt.Println("Reasoning:", result.Reasoning)
fmt.Println("Tokens Used:", result.TokensUsed)
fmt.Println("Time:", result.TotalTime)

// Access individual stages
for stage, stageResult := range result.Stages {
    fmt.Printf("Stage %s: %s\n", stage, stageResult.Content)
}
```

### Integration with Crush Sessions

```go
// Wrap as Crush service
service := agent.AsService()

// Run with session
events, err := service.Run(ctx, sessionID, userQuery)

// Process events
for event := range events {
    switch event.Type {
    case agent.AgentEventTypeResponse:
        // Handle response
    case agent.AgentEventTypeError:
        // Handle error
    }
}
```

## Configuration Options

### Agent Configuration

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `reasoning_depth` | int | 3 | Depth of reasoning (1-5) |
| `parallel_stages` | bool | true | Enable parallel stage execution |
| `max_stage_time` | int | 30 | Maximum seconds per stage |
| `enable_reflection` | bool | true | Enable reflection stage |
| `thinking_temperature` | float32 | 0.3 | Temperature for reasoning |
| `max_thinking_tokens` | int | 1024 | Max tokens per stage |
| `verbose_thinking` | bool | false | Enable verbose output |

### vLLM Provider Configuration

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `gpu_memory_utilization` | float32 | 0.9 | GPU memory to use |
| `max_model_len` | int | 8192 | Maximum model context |
| `tensor_parallel_size` | int | 1 | GPUs for tensor parallelism |
| `quantization` | string | "" | Quantization method |
| `enable_caching` | bool | true | Enable KV caching |

## Performance Optimization

### Model-Specific Settings

#### 7B Models (e.g., Mistral-7B, CodeLlama-7B)
```go
config := map[string]any{
    "reasoning_depth": 2,
    "max_thinking_tokens": 1024,
    "thinking_temperature": 0.4,
    "parallel_stages": false,
}
```

#### 13B Models (e.g., Llama-2-13B)
```go
config := map[string]any{
    "reasoning_depth": 3,
    "max_thinking_tokens": 1536,
    "thinking_temperature": 0.35,
    "parallel_stages": false,
}
```

#### 30B+ Models (e.g., Yi-34B, Mixtral-8x7B)
```go
config := map[string]any{
    "reasoning_depth": 4,
    "max_thinking_tokens": 2048,
    "thinking_temperature": 0.3,
    "parallel_stages": true,
}
```

### Optimization Tips

1. **Use Quantization**: Enable AWQ or GPTQ for larger models
2. **Adjust GPU Memory**: Set `gpu_memory_utilization` based on available VRAM
3. **Sequential Execution**: Disable `parallel_stages` for better cache utilization
4. **Token Management**: Balance `max_thinking_tokens` with model size
5. **Temperature Tuning**: Lower temperatures (0.2-0.4) for more focused reasoning

## Metrics and Monitoring

The agent provides comprehensive metrics:

```go
metrics := agent.GetMetrics()
// Returns:
// - total_requests: Total thinking requests
// - total_stages: Total stages executed
// - avg_stage_time_ms: Average time per stage
// - avg_thinking_time_ms: Average total thinking time
// - stage_success_rate: Percentage of successful stages
// - reflection_count: Number of reflections performed
// - tokens_used: Total tokens consumed
```

## Examples

### Problem Solving
```go
result, _ := agent.Think(ctx, `
    My application is experiencing memory leaks in production.
    The issue appears after 24-48 hours of runtime.
    Help me diagnose and fix this issue.
`)
```

### Code Analysis
```go
result, _ := agent.Think(ctx, `
    Analyze this function for performance improvements:
    [code snippet]
`)
```

### Architecture Design
```go
result, _ := agent.Think(ctx, `
    Design a scalable microservices architecture for 
    an e-commerce platform handling 100k concurrent users.
`)
```

## Testing

Run the test suite:

```bash
go test ./internal/llm/agent -run TestSequentialThinking
```

Run benchmarks:

```bash
go test ./internal/llm/agent -bench BenchmarkSequentialThinking
```

## Troubleshooting

### Common Issues

1. **Timeout Errors**: Increase `max_stage_time` for complex problems
2. **Memory Issues**: Reduce `max_thinking_tokens` or enable quantization
3. **Slow Performance**: Disable `parallel_stages` for better caching
4. **Incomplete Reasoning**: Increase `reasoning_depth` for more thorough analysis

### Debug Mode

Enable verbose output for debugging:

```go
agentConfig.ExtraConfig["verbose_thinking"] = true
```

## Architecture Benefits

### From an Architectural Review Perspective

1. **Separation of Concerns**: Each thinking stage has a single responsibility
2. **Dependency Management**: Explicit stage dependencies prevent circular references
3. **Extensibility**: Easy to add new thinking stages without modifying core logic
4. **Performance Isolation**: Each stage can be optimized independently
5. **Testability**: Individual stages can be unit tested
6. **Observability**: Comprehensive metrics at stage and aggregate levels
7. **Fault Tolerance**: Optional stages can fail without breaking the pipeline
8. **Resource Management**: Configurable timeouts and token limits per stage

## Future Enhancements

- [ ] Dynamic stage selection based on problem type
- [ ] Learned stage ordering optimization
- [ ] Distributed stage execution across multiple GPUs
- [ ] Stage result caching for similar problems
- [ ] Integration with external knowledge bases
- [ ] Adaptive reasoning depth based on problem complexity
- [ ] Multi-model ensemble reasoning
- [ ] Real-time stage performance profiling

## License

Part of the Crush project. See main LICENSE file.