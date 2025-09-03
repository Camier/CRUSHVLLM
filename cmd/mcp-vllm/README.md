# vLLM MCP Server

A Model Context Protocol (MCP) server that provides integration between Crush and vLLM for local LLM inference with advanced capabilities.

## Features

### Core Tools
- **list_models**: List all available models in vLLM
- **get_model_info**: Get detailed information about a specific model
- **generate_completion**: Generate text completions using vLLM
- **get_metrics**: Monitor GPU usage and inference performance
- **clear_cache**: Clear vLLM's cache
- **get_cache_stats**: Get cache utilization statistics

### Advanced Tools
- **sequential_thinking**: Execute step-by-step reasoning processes
- **batch_inference**: Optimized batch processing of multiple prompts

## Configuration

### Environment Variables
- `VLLM_URL`: vLLM server URL (default: `http://localhost:8000`)
- `MCP_PORT`: MCP server port (default: `8080`)

### Example Configuration
```bash
export VLLM_URL="http://localhost:8000"
export MCP_PORT="8080"
```

## Installation

1. Ensure you have Go 1.21+ installed
2. Build the MCP server:
```bash
cd crush/cmd/mcp-vllm
go mod tidy
go build -o mcp-vllm main.go
```

3. Run the server:
```bash
./mcp-vllm
```

## Integration with Crush

Add to your Crush MCP configuration:

```json
{
  "mcpServers": {
    "vLLM Integration MCP": {
      "command": "/path/to/crush/cmd/mcp-vllm/mcp-vllm",
      "args": [],
      "env": {
        "VLLM_URL": "http://localhost:8000",
        "MCP_PORT": "8080"
      }
    }
  }
}
```

## Usage Examples

### List Available Models
```json
{
  "tool": "list_models"
}
```

### Generate Completion
```json
{
  "tool": "generate_completion",
  "arguments": {
    "model": "meta-llama/Llama-2-7b-hf",
    "prompt": "Explain quantum computing in simple terms:",
    "max_tokens": 200,
    "temperature": 0.7
  }
}
```

### Sequential Thinking
```json
{
  "tool": "sequential_thinking",
  "arguments": {
    "model": "meta-llama/Llama-2-7b-hf",
    "problem": "How can we optimize database performance for a high-traffic web application?",
    "steps": 5
  }
}
```

### Batch Inference
```json
{
  "tool": "batch_inference",
  "arguments": {
    "model": "meta-llama/Llama-2-7b-hf",
    "prompts": [
      "Summarize the benefits of renewable energy:",
      "Explain the water cycle:",
      "What are the main causes of climate change?"
    ],
    "max_tokens": 150,
    "temperature": 0.6
  }
}
```

### Monitor Performance
```json
{
  "tool": "get_metrics"
}
```

## Architecture

### Components
- **VLLMServer**: HTTP client for vLLM API integration
- **MCP Tools**: Individual tool implementations
- **Batch Processing**: Optimized concurrent inference
- **Sequential Thinking**: Step-by-step reasoning engine

### Design Principles
- **Error Handling**: Comprehensive error checking and reporting
- **Context Management**: Proper context propagation for cancellation
- **Resource Management**: Connection pooling and timeout handling
- **Concurrency**: Safe parallel processing with controlled limits
- **Extensibility**: Interface-based design for easy testing and mocking

### Performance Features
- **Connection Pooling**: Reuses HTTP connections for efficiency
- **Batch Optimization**: Processes multiple requests concurrently
- **Cache Management**: Leverages vLLM's built-in caching
- **Resource Monitoring**: Real-time performance metrics

## Development

### Code Style
Follows Crush's Go coding standards:
- Uses `gofumpt` formatting
- Standard Go naming conventions
- Explicit error returns
- Context as first parameter
- Small, focused interfaces

### Testing
```bash
go test ./...
```

### Building for Production
```bash
CGO_ENABLED=0 GOOS=linux go build -a -installsuffix cgo -o mcp-vllm main.go
```

## Dependencies

- `github.com/mark3labs/mcp-go`: MCP protocol implementation
- Standard Go libraries for HTTP, JSON, and concurrency

## Security Considerations

- Server runs on localhost by default
- No authentication required for local development
- Environment variables for configuration (no hardcoded secrets)
- Input validation for all tool parameters
- Safe concurrent processing with proper resource limits

## Troubleshooting

### Common Issues

**Connection Failed**
- Verify vLLM server is running on specified URL
- Check firewall and network connectivity
- Ensure vLLM API is accessible

**Model Not Found**
- Verify model is loaded in vLLM
- Check model name spelling and format
- Use `list_models` to see available models

**Performance Issues**
- Monitor GPU memory usage with `get_metrics`
- Adjust batch sizes for optimal performance
- Clear cache if memory is full

**MCP Protocol Errors**
- Check MCP server logs for details
- Verify tool arguments match schema
- Ensure proper JSON formatting

## Contributing

1. Follow Crush's code style guidelines
2. Add tests for new functionality
3. Update documentation for new tools
4. Ensure backward compatibility

## License

Same as Crush project license.