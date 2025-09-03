#!/bin/bash

# Crush + vLLM Quick Start Script
# One-command setup for optimal local LLM inference

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
MAGENTA='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Configuration
CRUSH_DIR="$(dirname "$(readlink -f "$0")")"
VLLM_MODEL="${VLLM_MODEL:-Qwen/Qwen2.5-Coder-32B-Instruct-AWQ}"
VLLM_PORT="${VLLM_PORT:-8000}"
ENABLE_MONITORING="${ENABLE_MONITORING:-true}"
USE_DOCKER="${USE_DOCKER:-false}"

# Logging
log_info() { echo -e "${BLUE}[INFO]${NC} $1"; }
log_success() { echo -e "${GREEN}[✓]${NC} $1"; }
log_warning() { echo -e "${YELLOW}[!]${NC} $1"; }
log_error() { echo -e "${RED}[✗]${NC} $1"; }

# Banner
print_banner() {
    echo -e "${MAGENTA}"
    echo "╔═══════════════════════════════════════════╗"
    echo "║        Crush + vLLM Quick Start           ║"
    echo "║    High-Performance Local LLM Inference   ║"
    echo "╚═══════════════════════════════════════════╝"
    echo -e "${NC}"
}

# Check system requirements
check_requirements() {
    log_info "Checking system requirements..."
    
    # Check for GPU
    if ! command -v nvidia-smi &> /dev/null; then
        log_error "NVIDIA GPU not detected. vLLM requires a CUDA-capable GPU."
        log_info "You can still use Crush with other providers (OpenAI, Anthropic, etc.)"
        read -p "Continue without vLLM? (y/n) " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            exit 1
        fi
        SKIP_VLLM=true
    fi
    
    # Check GPU memory
    GPU_MEM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -n1)
    GPU_MEM_GB=$((GPU_MEM / 1024))
    log_success "GPU detected with ${GPU_MEM_GB}GB memory"
    
    if [ $GPU_MEM_GB -lt 8 ]; then
        log_warning "GPU has less than 8GB memory. Using reduced model settings."
        export VLLM_MAX_MODEL_LEN=4096
        export VLLM_GPU_MEMORY=0.85
    fi
    
    # Check Python
    if ! command -v python3 &> /dev/null; then
        log_error "Python 3 not found. Please install Python 3.8+"
        exit 1
    fi
    
    # Check Docker if needed
    if [ "$USE_DOCKER" == "true" ]; then
        if ! command -v docker &> /dev/null; then
            log_error "Docker not found. Please install Docker or run with USE_DOCKER=false"
            exit 1
        fi
        if ! command -v docker-compose &> /dev/null; then
            log_warning "docker-compose not found. Installing..."
            pip install docker-compose
        fi
    fi
    
    # Check Go for Crush
    if ! command -v go &> /dev/null; then
        log_warning "Go not found. Installing Go 1.23..."
        wget -q https://go.dev/dl/go1.23.0.linux-amd64.tar.gz
        sudo tar -C /usr/local -xzf go1.23.0.linux-amd64.tar.gz
        export PATH=$PATH:/usr/local/go/bin
        rm go1.23.0.linux-amd64.tar.gz
    fi
    
    log_success "All requirements met"
}

# Install dependencies
install_dependencies() {
    log_info "Installing dependencies..."
    
    # Check Python version
    PYTHON_VERSION=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
    log_info "Python version: $PYTHON_VERSION"
    
    if [[ "$PYTHON_VERSION" == "3.13" ]]; then
        log_warning "Python 3.13 detected. Some packages may have compatibility issues."
        log_info "Using compatibility mode..."
    fi
    
    # Python packages - install in correct order
    log_info "Installing Python packages..."
    pip install --quiet --upgrade pip
    
    # Install PyTorch first (required by vLLM)
    log_info "Installing PyTorch..."
    pip install --quiet torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
    
    # Install other dependencies
    log_info "Installing transformers and accelerate..."
    pip install --quiet transformers accelerate
    
    # Install vLLM last (depends on torch)
    log_info "Installing vLLM..."
    pip install --quiet vllm || {
        log_warning "vLLM installation failed. Trying alternative method..."
        # For Python 3.13, we might need to build from source
        pip install --quiet git+https://github.com/vllm-project/vllm.git
    }
    
    # Crush dependencies
    log_info "Building Crush..."
    cd "$CRUSH_DIR"
    go mod download
    go build -o crush cmd/crush/main.go || go build -o crush .
    
    log_success "Dependencies installed"
}

# Setup vLLM
setup_vllm() {
    log_info "Setting up vLLM server..."
    
    # Check if vLLM is already running
    if curl -s "http://localhost:$VLLM_PORT/health" > /dev/null 2>&1; then
        log_warning "vLLM already running on port $VLLM_PORT"
        return 0
    fi
    
    # Download model if needed
    log_info "Checking model availability..."
    python3 -c "from transformers import AutoConfig; AutoConfig.from_pretrained('$VLLM_MODEL')" 2>/dev/null || {
        log_info "Downloading model $VLLM_MODEL..."
        python3 -c "from transformers import AutoTokenizer; AutoTokenizer.from_pretrained('$VLLM_MODEL')"
    }
    
    # Start vLLM
    log_info "Starting vLLM server..."
    if [ "$USE_DOCKER" == "true" ]; then
        docker-compose up -d vllm
    else
        # Start in background
        nohup "$CRUSH_DIR/scripts/start_vllm.sh" > /tmp/vllm.log 2>&1 &
        VLLM_PID=$!
        echo $VLLM_PID > /tmp/vllm.pid
        
        # Wait for startup
        log_info "Waiting for vLLM to start..."
        for i in {1..30}; do
            if curl -s "http://localhost:$VLLM_PORT/health" > /dev/null 2>&1; then
                log_success "vLLM server started successfully"
                break
            fi
            sleep 2
            echo -n "."
        done
        echo ""
    fi
}

# Setup monitoring
setup_monitoring() {
    if [ "$ENABLE_MONITORING" != "true" ]; then
        return 0
    fi
    
    log_info "Setting up monitoring..."
    
    if [ "$USE_DOCKER" == "true" ]; then
        docker-compose up -d prometheus grafana
        log_success "Monitoring started - Grafana at http://localhost:3000"
    else
        # Manual setup for Prometheus/Grafana
        log_warning "Manual monitoring setup required. See monitoring/README.md"
    fi
}

# Configure Crush
configure_crush() {
    log_info "Configuring Crush..."
    
    # Create config directory
    mkdir -p ~/.config/crush
    
    # Generate configuration
    cat > ~/.config/crush/config.json <<EOF
{
  "providers": {
    "vllm": {
      "type": "openai",
      "base_url": "http://localhost:$VLLM_PORT/v1",
      "api_key": "dummy",
      "default": true
    }
  },
  "models": {
    "qwen-coder": {
      "provider": "vllm",
      "model": "$(basename $VLLM_MODEL)",
      "context_window": 32768,
      "max_tokens": 4096
    }
  }
}
EOF
    
    log_success "Crush configured with vLLM provider"
}

# Test integration
test_integration() {
    log_info "Testing integration..."
    
    # Test vLLM API
    log_info "Testing vLLM API..."
    RESPONSE=$(curl -s -X POST "http://localhost:$VLLM_PORT/v1/chat/completions" \
        -H "Content-Type: application/json" \
        -d '{
            "model": "'$(basename $VLLM_MODEL)'",
            "messages": [{"role": "user", "content": "Say hello"}],
            "max_tokens": 50
        }')
    
    if echo "$RESPONSE" | grep -q "content"; then
        log_success "vLLM API test passed"
    else
        log_error "vLLM API test failed"
        echo "$RESPONSE"
        return 1
    fi
    
    # Test Crush
    log_info "Testing Crush CLI..."
    echo "Hello, test" | "$CRUSH_DIR/crush" --provider vllm --non-interactive > /dev/null 2>&1 && \
        log_success "Crush integration test passed" || \
        log_warning "Crush test failed - manual configuration may be needed"
}

# Print status
print_status() {
    echo ""
    echo -e "${GREEN}═══════════════════════════════════════════${NC}"
    echo -e "${GREEN}       Setup Complete!${NC}"
    echo -e "${GREEN}═══════════════════════════════════════════${NC}"
    echo ""
    echo -e "${CYAN}Services:${NC}"
    echo -e "  • vLLM API: ${GREEN}http://localhost:$VLLM_PORT${NC}"
    echo -e "  • API Docs: ${GREEN}http://localhost:$VLLM_PORT/docs${NC}"
    
    if [ "$ENABLE_MONITORING" == "true" ]; then
        echo -e "  • Grafana:  ${GREEN}http://localhost:3000${NC} (admin/admin)"
        echo -e "  • Prometheus: ${GREEN}http://localhost:9090${NC}"
    fi
    
    echo ""
    echo -e "${CYAN}Quick Test:${NC}"
    echo -e "  ${YELLOW}echo 'Write hello world in Python' | ./crush${NC}"
    echo ""
    echo -e "${CYAN}Using with Crush:${NC}"
    echo -e "  ${YELLOW}./crush --provider vllm${NC}"
    echo ""
    echo -e "${CYAN}Stop Services:${NC}"
    if [ "$USE_DOCKER" == "true" ]; then
        echo -e "  ${YELLOW}docker-compose down${NC}"
    else
        echo -e "  ${YELLOW}kill \$(cat /tmp/vllm.pid)${NC}"
    fi
    echo ""
}

# Cleanup function
cleanup() {
    log_warning "Interrupted. Cleaning up..."
    if [ -f /tmp/vllm.pid ]; then
        kill $(cat /tmp/vllm.pid) 2>/dev/null || true
        rm /tmp/vllm.pid
    fi
    exit 1
}

# Main execution
main() {
    print_banner
    
    # Parse arguments
    while [[ $# -gt 0 ]]; do
        case $1 in
            --docker)
                USE_DOCKER=true
                shift
                ;;
            --no-monitoring)
                ENABLE_MONITORING=false
                shift
                ;;
            --model)
                VLLM_MODEL="$2"
                shift 2
                ;;
            --port)
                VLLM_PORT="$2"
                shift 2
                ;;
            --help)
                echo "Usage: $0 [OPTIONS]"
                echo ""
                echo "Options:"
                echo "  --docker         Use Docker for deployment"
                echo "  --no-monitoring  Disable Prometheus/Grafana"
                echo "  --model MODEL    vLLM model to use"
                echo "  --port PORT      vLLM server port (default: 8000)"
                echo "  --help           Show this help message"
                exit 0
                ;;
            *)
                log_error "Unknown option: $1"
                exit 1
                ;;
        esac
    done
    
    # Set trap for cleanup
    trap cleanup INT TERM
    
    # Run setup steps
    check_requirements
    install_dependencies
    setup_vllm
    setup_monitoring
    configure_crush
    test_integration
    print_status
    
    log_success "Quick start complete! Crush + vLLM is ready to use."
}

# Run main function
main "$@"