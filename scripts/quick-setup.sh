#!/usr/bin/env bash
# Ultra-fast development setup - optimized for immediate productivity
set -euo pipefail

# Colors
RED='\033[31m'
GREEN='\033[32m'
YELLOW='\033[33m'
BLUE='\033[34m'
CYAN='\033[36m'
RESET='\033[0m'

log() { echo -e "${GREEN}[QUICK-SETUP]${RESET} $1"; }
warn() { echo -e "${YELLOW}[WARN]${RESET} $1"; }
error() { echo -e "${RED}[ERROR]${RESET} $1"; exit 1; }
info() { echo -e "${BLUE}[INFO]${RESET} $1"; }
progress() { echo -e "${CYAN}[PROGRESS]${RESET} $1"; }

# Configuration
PYTHON_VENV_PATH="venv"
VLLM_MODEL_FAST="facebook/opt-125m"  # Ultra-lightweight for development
VLLM_MODEL_DEFAULT="Qwen/Qwen2.5-Coder-3B-Instruct"  # Better for real work
VLLM_PORT=8000
MONITOR_PORT=9090

# Check if already in project directory
if [[ ! -f "go.mod" ]] || [[ ! -f "main.go" ]]; then
    error "Run this from the Crush project root directory"
fi

# Performance timing
START_TIME=$(date +%s)

log "🚀 Setting up Crush development environment (ultra-fast mode)"

# Step 1: Parallel system checks
progress "Checking system requirements..."
{
    # Go check
    if ! command -v go &>/dev/null; then
        error "Go not installed. Install Go 1.24+ first."
    fi
    GO_VERSION=$(go version | grep -o 'go[0-9]\+\.[0-9]\+' | sed 's/go//')
    info "Go ${GO_VERSION} detected"

    # Python check  
    if ! command -v python3 &>/dev/null; then
        error "Python3 not installed. Install Python 3.8+ for vLLM."
    fi
    PYTHON_VERSION=$(python3 --version | grep -o '[0-9]\+\.[0-9]\+')
    info "Python ${PYTHON_VERSION} detected"

    # UV check (modern Python package manager)
    if ! command -v uv &>/dev/null; then
        warn "UV not found. Installing for faster Python package management..."
        curl -LsSf https://astral.sh/uv/install.sh | sh
        export PATH="$HOME/.cargo/bin:$PATH"
    fi
    info "UV package manager available"
} &

# Step 2: Create directory structure in parallel
progress "Creating directory structure..."
{
    mkdir -p {logs,cache,models,monitoring/data,scripts/dev,configs/dev}
    touch logs/{crush.log,vllm.log,dev.log}
} &

# Step 3: Install Go tools in background
progress "Installing Go development tools..."
{
    go install github.com/golangci/golangci-lint/cmd/golangci-lint@latest &
    go install mvdan.cc/gofumpt@latest &
    go install github.com/securecodewarrior/gosec/v2/cmd/gosec@latest &
    go install golang.org/x/vuln/cmd/govulncheck@latest &
    wait  # Wait for all Go tool installations
} &

# Wait for parallel operations to complete
wait

# Step 4: Setup Python environment (optimized)
if [[ ! -d "${PYTHON_VENV_PATH}" ]]; then
    progress "Creating Python virtual environment with UV..."
    uv venv "${PYTHON_VENV_PATH}" --python python3
fi

progress "Installing vLLM and dependencies (fast mode)..."
source "${PYTHON_VENV_PATH}/bin/activate"
{
    # Use UV for faster installation
    uv pip install --upgrade pip
    
    # Install vLLM with CUDA if available, CPU-only otherwise
    if command -v nvidia-smi &>/dev/null; then
        uv pip install vllm torch --extra-index-url https://download.pytorch.org/whl/cu118
        info "✅ Installed vLLM with CUDA support"
    else
        uv pip install vllm torch
        warn "⚠️  Installed CPU-only vLLM (slower)"
    fi
    
    # Essential packages
    uv pip install huggingface-hub transformers tokenizers fastapi uvicorn
} &

# Step 5: Go project setup
progress "Setting up Go project..."
{
    go mod download
    go mod tidy
    go mod verify
    
    # Pre-build for faster first run
    go build -o crush .
    info "✅ Built Crush binary"
} &

# Wait for Python and Go setup
wait
deactivate 2>/dev/null || true

# Step 6: Create optimized development configuration
progress "Creating development configuration..."

# Create local crush config
mkdir -p ~/.config/crush
if [[ ! -f ~/.config/crush/crush.json ]]; then
    cat > ~/.config/crush/crush.json << 'EOF'
{
  "$schema": "https://charm.land/crush.json",
  "options": {
    "debug": true,
    "debug_lsp": false
  },
  "providers": {
    "vllm-dev": {
      "name": "vLLM Development",
      "base_url": "http://localhost:8000/v1/",
      "type": "openai",
      "models": [
        {
          "id": "facebook/opt-125m",
          "name": "OPT-125M (Lightning Fast)",
          "context_window": 2048,
          "default_max_tokens": 512
        },
        {
          "id": "Qwen/Qwen2.5-Coder-3B-Instruct",
          "name": "Qwen2.5-Coder-3B (Balanced)",
          "context_window": 32768,
          "default_max_tokens": 4096
        }
      ]
    }
  },
  "permissions": {
    "allowed_tools": ["view", "ls", "grep", "edit", "mcp_*"]
  }
}
EOF
    info "✅ Created development configuration"
fi

# Step 7: Create rapid development scripts
progress "Creating development automation scripts..."

# Hot-reload development server
cat > scripts/dev/hot-reload.sh << 'EOF'
#!/usr/bin/env bash
set -euo pipefail

log() { echo -e "\033[32m[HOT-RELOAD]\033[0m $1"; }
warn() { echo -e "\033[33m[WARN]\033[0m $1"; }

CRUSH_PID=""
VLLM_PID=""

cleanup() {
    log "Shutting down..."
    [[ -n "${CRUSH_PID}" ]] && kill "${CRUSH_PID}" 2>/dev/null || true
    [[ -n "${VLLM_PID}" ]] && kill "${VLLM_PID}" 2>/dev/null || true
    exit 0
}
trap cleanup SIGINT SIGTERM

# Start vLLM in background
if [[ "${1:-}" != "--no-vllm" ]]; then
    log "Starting vLLM server..."
    source venv/bin/activate
    python -m vllm.entrypoints.openai.api_server \
        --model facebook/opt-125m \
        --port 8000 \
        --host 0.0.0.0 \
        --max-model-len 1024 \
        --gpu-memory-utilization 0.3 \
        --trust-remote-code \
        --disable-log-requests > logs/vllm.log 2>&1 &
    VLLM_PID=$!
    deactivate
    
    # Wait for startup
    for i in {1..30}; do
        if curl -s http://localhost:8000/health > /dev/null 2>&1; then
            log "✅ vLLM server ready"
            break
        fi
        [[ $i -eq 30 ]] && warn "vLLM startup timeout"
        sleep 1
    done
fi

# Build and watch for changes
build_and_run() {
    log "Building Crush..."
    if CGO_ENABLED=1 go build -race -o crush .; then
        [[ -n "${CRUSH_PID}" ]] && kill "${CRUSH_PID}" 2>/dev/null || true
        CRUSH_PROFILE=true ./crush &
        CRUSH_PID=$!
        log "✅ Crush restarted (PID: ${CRUSH_PID})"
    else
        warn "Build failed"
    fi
}

build_and_run

# Watch for changes
if command -v inotifywait > /dev/null 2>&1; then
    log "👀 Watching for changes..."
    while inotifywait -r -e modify,create,delete \
        --exclude='(\.git|logs|cache|models|venv|crush$)' . 2>/dev/null; do
        sleep 0.5  # Debounce
        build_and_run
    done
else
    log "Install inotify-tools for automatic rebuilds"
    wait
fi
EOF
chmod +x scripts/dev/hot-reload.sh

# Quick test script  
cat > scripts/dev/quick-test.sh << 'EOF'
#!/usr/bin/env bash
set -euo pipefail

log() { echo -e "\033[32m[QUICK-TEST]\033[0m $1"; }

log "🧪 Running quick development tests..."

# Build test
log "Build test..."
go build -o crush-test . && rm crush-test
echo "✅ Build: PASS"

# Unit tests (parallel, fast)
log "Unit tests..."
go test -race -timeout=30s ./... -short
echo "✅ Unit tests: PASS" 

# Lint check
log "Lint check..."
if command -v golangci-lint > /dev/null; then
    golangci-lint run --timeout=30s --fast
    echo "✅ Lint: PASS"
fi

# Security scan (if available)
log "Security scan..."  
if command -v gosec > /dev/null; then
    gosec -quiet ./...
    echo "✅ Security: PASS"
fi

log "🎉 All quick tests passed!"
EOF
chmod +x scripts/dev/quick-test.sh

# Performance monitoring script
cat > scripts/dev/monitor-perf.sh << 'EOF'
#!/usr/bin/env bash
set -euo pipefail

log() { echo -e "\033[36m[MONITOR]\033[0m $1"; }

log "📊 Starting performance monitoring..."

# Start crush with profiling
export CRUSH_PROFILE=true
./crush &
CRUSH_PID=$!

# Wait a moment for startup
sleep 2

log "🔍 Performance endpoints available:"
echo "  CPU Profile:    http://localhost:6060/debug/pprof/profile?seconds=30"
echo "  Memory Profile: http://localhost:6060/debug/pprof/heap"
echo "  Goroutines:     http://localhost:6060/debug/pprof/goroutine"
echo "  Traces:         http://localhost:6060/debug/pprof/trace?seconds=10"

# Monitor resources
while kill -0 $CRUSH_PID 2>/dev/null; do
    CPU=$(ps -p $CRUSH_PID -o %cpu --no-headers | tr -d ' ')
    MEM=$(ps -p $CRUSH_PID -o %mem --no-headers | tr -d ' ')
    echo -ne "\r💻 CPU: ${CPU}% | 🧠 Memory: ${MEM}% | PID: ${CRUSH_PID}"
    sleep 1
done
echo ""

log "Monitoring stopped"
EOF
chmod +x scripts/dev/monitor-perf.sh

info "✅ Created development automation scripts"

# Step 8: Create environment configuration
progress "Creating environment configuration..."

cat > .env.development << 'EOF'
# Crush Development Environment
CRUSH_DEBUG=true
CRUSH_PROFILE=true
VLLM_HOST=localhost
VLLM_PORT=8000
GPU_MEMORY_UTILIZATION=0.3
HF_CACHE_DIR=./cache
TOKENIZERS_PARALLELISM=false

# Development aliases
alias crush-dev='./scripts/dev/hot-reload.sh'
alias crush-test='./scripts/dev/quick-test.sh'
alias crush-monitor='./scripts/dev/monitor-perf.sh'
alias vllm-logs='tail -f logs/vllm.log'
alias crush-logs='tail -f logs/crush.log'
EOF

# Create .envrc for direnv users
cat > .envrc << 'EOF'
#!/usr/bin/env bash
# Load development environment
source_env_if_exists .env.development

# Load Python virtual environment
source_env venv/bin/activate

# Development PATH additions
PATH_add ./scripts/dev
PATH_add ./venv/bin

# Performance optimizations
export GOGC=100
export GOMAXPROCS=$(nproc)

# Show helpful info
echo "🚀 Crush development environment loaded"
echo "💡 Quick commands: crush-dev, crush-test, crush-monitor"
EOF

if command -v direnv &>/dev/null; then
    direnv allow
    info "✅ Environment will auto-load with direnv"
else
    warn "Install direnv for automatic environment loading"
fi

# Step 9: Pre-download models for faster first run
progress "Pre-downloading development models..."
source "${PYTHON_VENV_PATH}/bin/activate"
{
    python -c "
from huggingface_hub import snapshot_download
import os
os.environ['HF_CACHE_DIR'] = './cache'

# Download lightweight model for development
try:
    snapshot_download('facebook/opt-125m', cache_dir='./cache')
    print('✅ Downloaded OPT-125M model')
except Exception as e:
    print(f'⚠️  Model download failed: {e}')
" 2>/dev/null || warn "Model pre-download failed (will download on first use)"
}
deactivate

# Step 10: Final validation
progress "Running validation..."
{
    # Test Go build
    if ./crush --help > /dev/null 2>&1; then
        info "✅ Crush binary works"
    else
        warn "⚠️  Crush binary may have issues"
    fi
    
    # Test Python environment
    source "${PYTHON_VENV_PATH}/bin/activate"
    if python -c "import vllm; print('✅ vLLM import successful')" 2>/dev/null; then
        info "✅ vLLM installation verified"
    else  
        warn "⚠️  vLLM installation may have issues"
    fi
    deactivate
} 

# Calculate setup time
END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))

log "🎉 Setup complete in ${DURATION} seconds!"

echo ""
echo -e "${BLUE}=== Quick Start Commands ===${RESET}"
echo -e "• ${GREEN}make dev${RESET}                    - Start hot-reload development server"
echo -e "• ${GREEN}./scripts/dev/quick-test.sh${RESET}  - Run fast development tests"  
echo -e "• ${GREEN}make test${RESET}                   - Run full test suite"
echo -e "• ${GREEN}make vllm-start${RESET}             - Start vLLM server only"
echo -e "• ${GREEN}make help${RESET}                   - Show all available commands"

echo ""
echo -e "${BLUE}=== Next Steps ===${RESET}"
echo "1. Run: ${GREEN}make dev${RESET} to start development server"
echo "2. Open: ${CYAN}http://localhost:6060/debug/pprof/${RESET} for profiling" 
echo "3. Test: ${CYAN}http://localhost:8000/health${RESET} for vLLM health"

echo ""
echo -e "${GREEN}🚀 Ready to crush it! ${RESET}"