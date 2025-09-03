# Crush vLLM Integration

High-performance local LLM inference with Charmbracelet Crush and vLLM.

## Features

- ⚡ **Fast Model Switching**: Seamless model switching with automatic server restart
- 🎨 **Beautiful TUI**: Charmbracelet Bubble Tea interface with progress indicators
- 🚀 **GPU Optimized**: Configured for NVIDIA GPUs with XFormers backend
- 🔧 **Developer Friendly**: One-command setup with hot reload development
- 🔌 **MCP Integration**: Model Context Protocol support for extended capabilities

## Quick Start

```bash
# Clone and setup
git clone https://github.com/Camier/CRUSHVLLM.git
cd CRUSHVLLM
make quick-start   # Complete setup + dev server

# Or manual setup
./scripts/quick-setup.sh
./crush
```

## Prerequisites

- Python 3.12+ (for vLLM)
- Go 1.22+
- NVIDIA GPU with CUDA support (optional, CPU mode available)
- Node.js 22+ (for MCP servers)

## Installation

### Automated Setup
```bash
make quick-start   # Everything in one command
```

### Manual Setup
```bash
# 1. Install vLLM
source ~/venvs/vllm_uv/bin/activate
uv pip install vllm

# 2. Build Crush
go build -o crush .

# 3. Configure models
./setup_vllm_models.sh
```

## Usage

### Starting Crush
```bash
./crush
```

### Model Switching
- Press `Ctrl+M` to open model selector
- Use number keys 1-9 for quick selection
- Server automatically restarts with new model

### Available Models
- facebook/opt-125m (Testing)
- meta-llama/Llama-3.2-1B-Instruct
- Qwen/Qwen2.5-Coder-3B-Instruct
- Qwen/Qwen2.5-Coder-7B-Instruct (Default)
- mistralai/Mistral-7B-Instruct-v0.2

## Development

### Commands
```bash
make dev          # Start development server with hot reload
make test         # Run tests
make lint         # Lint code
make build        # Production build
make monitor      # System monitoring dashboard
```

### Project Structure
```
crush/
├── internal/
│   ├── vllm/          # vLLM server management
│   │   ├── server.go  # Server lifecycle
│   │   └── messages.go # Status messages
│   ├── tui/           # Terminal UI
│   └── app/           # Application core
├── scripts/           # Development scripts
├── Makefile          # Build automation
└── crush.json        # Configuration
```

## Architecture

Crush integrates with vLLM through an OpenAI-compatible API:

1. **vLLM Server**: Runs as separate process on port 8000
2. **Server Manager**: Handles lifecycle and model switching
3. **OpenAI Provider**: Communicates with vLLM's REST API
4. **TUI Layer**: Provides visual feedback and controls

## Configuration

### GPU Settings
Configure in `~/.config/crush/config.json`:
```json
{
  "providers": {
    "vllm": {
      "type": "openai",
      "base_url": "http://localhost:8000/v1",
      "api_key": "dummy"
    }
  }
}
```

### Environment Variables
```bash
export VLLM_GPU_MEMORY_UTILIZATION=0.85
export VLLM_ATTENTION_BACKEND=XFORMERS
```

## Troubleshooting

### GPU Memory Issues
```bash
# Reduce memory usage
export VLLM_GPU_MEMORY_UTILIZATION=0.7

# Use smaller model
./crush  # Then select OPT-125M
```

### Server Won't Start
```bash
# Check if port is in use
lsof -i :8000

# Kill existing vLLM process
pkill -f vllm.entrypoints.openai
```

## Contributing

1. Fork the repository
2. Create your feature branch
3. Commit your changes
4. Push to the branch
5. Open a Pull Request

## License

MIT - See LICENSE file for details

## Acknowledgments

- [Charmbracelet](https://github.com/charmbracelet) for Crush and Bubble Tea
- [vLLM](https://github.com/vllm-project/vllm) for high-performance inference
- Claude AI agents for development assistance