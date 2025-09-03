#!/usr/bin/env bash
# One-command development environment setup for Crush vLLM integration
set -euo pipefail

# Colors
RED='\033[31m'
GREEN='\033[32m'
YELLOW='\033[33m'
BLUE='\033[34m'
RESET='\033[0m'

log() {
    echo -e "${GREEN}[SETUP]${RESET} $1"
}

warn() {
    echo -e "${YELLOW}[WARN]${RESET} $1"
}

error() {
    echo -e "${RED}[ERROR]${RESET} $1"
    exit 1
}

info() {
    echo -e "${BLUE}[INFO]${RESET} $1"
}

# Check if we're in the right directory
if [[ ! -f "go.mod" ]] || [[ ! -f "main.go" ]]; then
    error "Not in Crush project directory. Please run from project root."
fi

log "Setting up Crush vLLM development environment..."

# Step 1: Verify system requirements
log "Checking system requirements..."

# Check Go version
if ! command -v go &> /dev/null; then
    error "Go is not installed. Please install Go 1.24+ first."
fi

GO_VERSION=$(go version | grep -o 'go[0-9]\+\.[0-9]\+' | sed 's/go//')
if [[ "${GO_VERSION}" < "1.24" ]]; then
    warn "Go version ${GO_VERSION} detected. Recommend Go 1.24+ for best performance."
fi

# Check Docker (optional but recommended)
if command -v docker &> /dev/null; then
    info "Docker found - Docker workflows will be available"
    if command -v docker-compose &> /dev/null; then
        info "Docker Compose found - Full containerization available"
    else
        warn "Docker Compose not found - Some Docker workflows may not work"
    fi
else
    warn "Docker not found - Docker workflows will be disabled"
fi

# Check Python for vLLM (essential)
if command -v python3 &> /dev/null; then
    PYTHON_VERSION=$(python3 --version | grep -o '[0-9]\+\.[0-9]\+')
    if [[ "${PYTHON_VERSION}" > "3.8" ]]; then
        info "Python ${PYTHON_VERSION} found - vLLM compatible"
    else
        warn "Python ${PYTHON_VERSION} may not be compatible with vLLM (requires 3.8+)"
    fi
else
    warn "Python 3 not found - vLLM will require manual Python setup"
fi

# Check CUDA (for GPU acceleration)
if command -v nvidia-smi &> /dev/null; then
    GPU_INFO=$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader,nounits | head -1)
    info "GPU detected: ${GPU_INFO}"
    
    # Check CUDA toolkit
    if command -v nvcc &> /dev/null; then
        CUDA_VERSION=$(nvcc --version | grep -o 'release [0-9]\+\.[0-9]\+' | grep -o '[0-9]\+\.[0-9]\+')
        info "CUDA ${CUDA_VERSION} found"
    else
        warn "CUDA toolkit not found - may need manual installation for GPU acceleration"
    fi
else
    warn "No NVIDIA GPU detected - will use CPU-only mode"
fi

# Step 2: Install Go development tools
log "Installing Go development tools..."

go install github.com/golangci/golangci-lint/v2/cmd/golangci-lint@latest
go install mvdan.cc/gofumpt@latest
go install github.com/goreleaser/goreleaser/v2@latest

# Additional security tools
go install github.com/securecodewarrior/gosec/v2/cmd/gosec@latest
go install golang.org/x/vuln/cmd/govulncheck@latest

info "Go development tools installed"

# Step 3: Install Task runner (if not using Make)
if ! command -v task &> /dev/null; then
    log "Installing Task runner..."
    if command -v go &> /dev/null; then
        go install github.com/go-task/task/v3/cmd/task@latest
        info "Task runner installed"
    else
        warn "Could not install Task runner - using Makefile only"
    fi
fi

# Step 4: Setup project dependencies
log "Installing project dependencies..."
go mod download
go mod tidy
go mod verify

# Step 5: Setup vLLM environment
log "Setting up vLLM environment..."

# Create vLLM directory structure
mkdir -p models cache logs monitoring/data

# Setup Python virtual environment for vLLM
if command -v python3 &> /dev/null; then
    if [[ ! -d "venv" ]]; then
        log "Creating Python virtual environment..."
        python3 -m venv venv
    fi
    
    log "Activating virtual environment and installing vLLM..."
    source venv/bin/activate
    
    # Install vLLM with appropriate backend
    if command -v nvidia-smi &> /dev/null; then
        pip install vllm torch --extra-index-url https://download.pytorch.org/whl/cu118
        info "Installed vLLM with CUDA support"
    else
        pip install vllm torch
        warn "Installed CPU-only vLLM (slower performance)"
    fi
    
    # Install additional tools
    pip install huggingface-hub transformers tokenizers
    
    deactivate
else
    warn "Python not available - vLLM setup skipped"
fi

# Step 6: Setup monitoring
log "Setting up monitoring..."

# Create monitoring configuration
mkdir -p monitoring/{prometheus,grafana}

cat > monitoring/prometheus/prometheus.yml << 'EOF'
global:
  scrape_interval: 15s

scrape_configs:
  - job_name: 'crush-app'
    static_configs:
      - targets: ['localhost:6060']
  
  - job_name: 'vllm-server'
    static_configs:
      - targets: ['localhost:8000']
    metrics_path: /metrics
    
  - job_name: 'node-exporter'
    static_configs:
      - targets: ['localhost:9100']
EOF

# Step 7: Create development configuration
log "Creating development configuration..."

# Create crush configuration directory
mkdir -p ~/.config/crush

# Create basic development configuration if it doesn't exist
if [[ ! -f ~/.config/crush/crush.json ]]; then
    cat > ~/.config/crush/crush.json << 'EOF'
{
  "$schema": "https://charm.land/crush.json",
  "options": {
    "debug": true,
    "debug_lsp": true
  },
  "providers": {
    "vllm-local": {
      "name": "vLLM Local",
      "base_url": "http://localhost:8000/v1/",
      "type": "openai",
      "models": [
        {
          "id": "facebook/opt-125m",
          "name": "OPT 125M (Fast Testing)",
          "context_window": 2048,
          "default_max_tokens": 1024
        },
        {
          "id": "meta-llama/Llama-3.2-1B-Instruct",
          "name": "Llama 3.2 1B Instruct",
          "context_window": 32768,
          "default_max_tokens": 4096
        },
        {
          "id": "Qwen/Qwen2.5-Coder-3B-Instruct",
          "name": "Qwen2.5 Coder 3B",
          "context_window": 32768,
          "default_max_tokens": 8192
        },
        {
          "id": "Qwen/Qwen2.5-Coder-7B-Instruct",
          "name": "Qwen2.5 Coder 7B (Default Large)",
          "context_window": 32768,
          "default_max_tokens": 8192
        }
      ]
    }
  },
  "permissions": {
    "allowed_tools": [
      "view",
      "ls", 
      "grep",
      "edit"
    ]
  }
}
EOF
    info "Created development configuration at ~/.config/crush/crush.json"
fi

# Step 8: Setup Git hooks (optional)
if [[ -d ".git" ]]; then
    log "Setting up Git hooks..."
    
    cat > .git/hooks/pre-commit << 'EOF'
#!/usr/bin/env bash
set -euo pipefail

echo "Running pre-commit checks..."

# Format code
make fmt

# Run linters
make lint

# Run tests
make test

echo "Pre-commit checks passed!"
EOF
    
    chmod +x .git/hooks/pre-commit
    info "Git pre-commit hook installed"
fi

# Step 9: Create helpful aliases
log "Creating development aliases..."

cat > .envrc << 'EOF'
# Development environment variables
export CRUSH_PROFILE=true
export CRUSH_DEBUG=true
export VLLM_HOST=localhost
export VLLM_PORT=8000
export GPU_MEMORY_UTILIZATION=0.9

# Development aliases
alias crush-dev='make dev'
alias crush-test='make test-all'
alias crush-build='make build'
alias crush-clean='make clean'
alias vllm-start='make vllm-start'
alias vllm-stop='make vllm-stop'
alias vllm-status='make vllm-health'

# Load virtual environment if it exists
if [ -d "./venv" ]; then
    source ./venv/bin/activate
fi
EOF

# If direnv is available, allow it
if command -v direnv &> /dev/null; then
    direnv allow
    info "Environment variables will be loaded automatically with direnv"
else
    warn "direnv not found - source .envrc manually for environment variables"
fi

# Step 10: Build initial binary
log "Building initial Crush binary..."
make build

# Step 11: Run quick validation
log "Running quick validation..."

# Test Go build
if [[ -x "./crush" ]]; then
    info "Crush binary built successfully"
else
    error "Failed to build Crush binary"
fi

# Test basic functionality
if ./crush --help > /dev/null 2>&1; then
    info "Crush binary works correctly"
else
    warn "Crush binary may have issues - check manually"
fi

# Step 12: Display summary
log "Development environment setup complete!"

echo ""
echo -e "${BLUE}=== Setup Summary ===${RESET}"
echo -e "• Go development tools: ${GREEN}✓${RESET}"
echo -e "• Project dependencies: ${GREEN}✓${RESET}"
echo -e "• Crush binary: ${GREEN}✓${RESET}"

if [[ -d "venv" ]]; then
    echo -e "• vLLM environment: ${GREEN}✓${RESET}"
else
    echo -e "• vLLM environment: ${YELLOW}⚠${RESET} (manual setup needed)"
fi

if command -v nvidia-smi &> /dev/null; then
    echo -e "• GPU support: ${GREEN}✓${RESET}"
else
    echo -e "• GPU support: ${YELLOW}⚠${RESET} (CPU only)"
fi

echo ""
echo -e "${BLUE}=== Next Steps ===${RESET}"
echo "1. Start development server: ${GREEN}make dev${RESET}"
echo "2. Run tests: ${GREEN}make test${RESET}"
echo "3. Setup vLLM: ${GREEN}make vllm-setup${RESET}"
echo "4. Start vLLM server: ${GREEN}make vllm-start${RESET}"
echo "5. View all available commands: ${GREEN}make help${RESET}"

echo ""
echo -e "${GREEN}Happy coding! 🚀${RESET}"