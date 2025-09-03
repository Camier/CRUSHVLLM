#!/usr/bin/env bash
# Development server with hot reload for Crush vLLM integration
set -euo pipefail

# Colors
RED='\033[31m'
GREEN='\033[32m'
YELLOW='\033[33m'
BLUE='\033[34m'
RESET='\033[0m'

log() {
    echo -e "${GREEN}[DEV]${RESET} $1"
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

# Configuration
CRUSH_BINARY="./crush"
VLLM_PORT=${VLLM_PORT:-8000}
PROFILE_PORT=${PROFILE_PORT:-6060}
MONITOR_INTERVAL=${MONITOR_INTERVAL:-2}
AUTO_RESTART=${AUTO_RESTART:-true}
ENABLE_PROFILING=${ENABLE_PROFILING:-true}

# Cleanup function
cleanup() {
    log "Shutting down development server..."
    
    # Kill background processes
    if [[ -n "${CRUSH_PID:-}" ]]; then
        kill ${CRUSH_PID} 2>/dev/null || true
        wait ${CRUSH_PID} 2>/dev/null || true
    fi
    
    if [[ -n "${WATCH_PID:-}" ]]; then
        kill ${WATCH_PID} 2>/dev/null || true
    fi
    
    if [[ -n "${VLLM_PID:-}" ]]; then
        kill ${VLLM_PID} 2>/dev/null || true
    fi
    
    # Clean up lock files
    rm -f .dev-server.pid .vllm-server.pid
    
    log "Development server stopped"
    exit 0
}

# Setup signal handlers
trap cleanup SIGINT SIGTERM EXIT

# Check if already running
if [[ -f .dev-server.pid ]]; then
    OLD_PID=$(cat .dev-server.pid)
    if kill -0 ${OLD_PID} 2>/dev/null; then
        error "Development server already running (PID: ${OLD_PID})"
    else
        rm -f .dev-server.pid
    fi
fi

# Create PID file
echo $$ > .dev-server.pid

log "Starting Crush development server..."

# Check if vLLM is needed and available
check_vllm() {
    if command -v python3 &> /dev/null && [[ -d "venv" ]]; then
        return 0
    fi
    return 1
}

# Start vLLM server if needed
start_vllm() {
    if [[ -f .vllm-server.pid ]]; then
        VLLM_PID=$(cat .vllm-server.pid)
        if kill -0 ${VLLM_PID} 2>/dev/null; then
            info "vLLM server already running (PID: ${VLLM_PID})"
            return 0
        else
            rm -f .vllm-server.pid
        fi
    fi
    
    if check_vllm; then
        log "Starting vLLM server on port ${VLLM_PORT}..."
        
        # Activate virtual environment and start vLLM
        source venv/bin/activate
        
        # Use a lightweight model for development
        nohup python -m vllm.entrypoints.openai.api_server \
            --model facebook/opt-125m \
            --port ${VLLM_PORT} \
            --host 0.0.0.0 \
            --gpu-memory-utilization 0.3 \
            --max-model-len 2048 \
            --trust-remote-code \
            --disable-log-requests \
            > logs/vllm-dev.log 2>&1 &
        
        VLLM_PID=$!
        echo ${VLLM_PID} > .vllm-server.pid
        
        # Wait for vLLM to be ready
        log "Waiting for vLLM server to be ready..."
        for i in {1..60}; do
            if curl -s http://localhost:${VLLM_PORT}/health > /dev/null 2>&1; then
                info "vLLM server is ready"
                break
            fi
            if [[ $i -eq 60 ]]; then
                warn "vLLM server taking longer than expected to start"
                break
            fi
            sleep 2
        done
        
        deactivate
    else
        warn "vLLM environment not available - skipping vLLM server"
    fi
}

# Build and start Crush
build_and_start() {
    log "Building Crush..."
    
    # Enable profiling if requested
    BUILD_FLAGS=""
    if [[ "${ENABLE_PROFILING}" == "true" ]]; then
        BUILD_FLAGS="-tags=profile"
        export CRUSH_PROFILE=true
    fi
    
    # Build with race detection in development
    CGO_ENABLED=1 go build -race ${BUILD_FLAGS} -o ${CRUSH_BINARY} .
    
    if [[ ! -x "${CRUSH_BINARY}" ]]; then
        error "Failed to build Crush"
    fi
    
    log "Starting Crush application..."
    
    # Start Crush in background
    CRUSH_ENV=""
    if [[ "${ENABLE_PROFILING}" == "true" ]]; then
        CRUSH_ENV="CRUSH_PROFILE=true CRUSH_DEBUG=true"
        info "Profiling enabled at http://localhost:${PROFILE_PORT}/debug/pprof/"
    fi
    
    env ${CRUSH_ENV} ${CRUSH_BINARY} &
    CRUSH_PID=$!
    
    # Give it a moment to start
    sleep 2
    
    if ! kill -0 ${CRUSH_PID} 2>/dev/null; then
        error "Crush failed to start"
    fi
    
    info "Crush started (PID: ${CRUSH_PID})"
}

# Watch for file changes
watch_files() {
    log "Watching for file changes..."
    
    # Use fswatch if available, otherwise fall back to basic polling
    if command -v fswatch &> /dev/null; then
        fswatch -o -r \
            --exclude='\.git' \
            --exclude='\.crush' \
            --exclude='logs' \
            --exclude='models' \
            --exclude='cache' \
            --exclude=${CRUSH_BINARY} \
            . | while read num; do
            log "Files changed, rebuilding..."
            restart_crush
        done &
    elif command -v inotifywait &> /dev/null; then
        while inotifywait -r -e modify,create,delete \
            --exclude='\.git' \
            --exclude='\.crush' \
            --exclude='logs' \
            --exclude='models' \
            --exclude='cache' \
            --exclude=${CRUSH_BINARY} \
            . 2>/dev/null; do
            log "Files changed, rebuilding..."
            restart_crush
        done &
    else
        warn "No file watcher available - manual restart required"
        return 1
    fi
    
    WATCH_PID=$!
}

# Restart Crush (not vLLM)
restart_crush() {
    if [[ -n "${CRUSH_PID:-}" ]]; then
        log "Stopping Crush..."
        kill ${CRUSH_PID} 2>/dev/null || true
        wait ${CRUSH_PID} 2>/dev/null || true
    fi
    
    build_and_start
}

# Status monitoring
monitor_status() {
    while true; do
        sleep ${MONITOR_INTERVAL}
        
        # Check Crush status
        if [[ -n "${CRUSH_PID:-}" ]] && ! kill -0 ${CRUSH_PID} 2>/dev/null; then
            warn "Crush process died, restarting..."
            if [[ "${AUTO_RESTART}" == "true" ]]; then
                build_and_start
            fi
        fi
        
        # Check vLLM status
        if [[ -n "${VLLM_PID:-}" ]] && ! kill -0 ${VLLM_PID} 2>/dev/null; then
            warn "vLLM process died"
            rm -f .vllm-server.pid
            if [[ "${AUTO_RESTART}" == "true" ]]; then
                start_vllm
            fi
        fi
    done &
}

# Display development information
show_info() {
    echo ""
    echo -e "${BLUE}=== Crush Development Server ===${RESET}"
    echo -e "Application: ${GREEN}http://localhost:8080${RESET} (if applicable)"
    
    if [[ "${ENABLE_PROFILING}" == "true" ]]; then
        echo -e "Profiling: ${GREEN}http://localhost:${PROFILE_PORT}/debug/pprof/${RESET}"
    fi
    
    if [[ -n "${VLLM_PID:-}" ]]; then
        echo -e "vLLM API: ${GREEN}http://localhost:${VLLM_PORT}/v1/${RESET}"
        echo -e "vLLM Health: ${GREEN}http://localhost:${VLLM_PORT}/health${RESET}"
    fi
    
    echo -e "Logs: ${YELLOW}tail -f logs/*.log${RESET}"
    echo ""
    
    echo -e "${BLUE}=== Controls ===${RESET}"
    echo -e "• ${GREEN}Ctrl+C${RESET} - Stop development server"
    echo -e "• ${GREEN}make test${RESET} - Run tests in another terminal"
    echo -e "• ${GREEN}make lint${RESET} - Run linters in another terminal"
    echo ""
    
    if [[ "${ENABLE_PROFILING}" == "true" ]]; then
        echo -e "${BLUE}=== Profiling Commands ===${RESET}"
        echo -e "• ${GREEN}make profile-cpu${RESET} - CPU profiling"
        echo -e "• ${GREEN}make profile-heap${RESET} - Memory profiling"
        echo ""
    fi
}

# Main execution
main() {
    # Create log directory
    mkdir -p logs
    
    # Start vLLM server
    start_vllm
    
    # Build and start Crush
    build_and_start
    
    # Start file watching
    if [[ "${AUTO_RESTART}" == "true" ]]; then
        if ! watch_files; then
            warn "File watching disabled - changes require manual restart"
        fi
    fi
    
    # Start status monitoring
    monitor_status
    
    # Show information
    show_info
    
    log "Development server ready! Press Ctrl+C to stop."
    
    # Keep script running
    wait
}

# Handle command line arguments
case "${1:-}" in
    --no-vllm)
        check_vllm() { return 1; }
        log "vLLM disabled"
        shift
        ;;
    --no-profiling)
        ENABLE_PROFILING=false
        log "Profiling disabled"
        shift
        ;;
    --no-watch)
        AUTO_RESTART=false
        log "Auto-restart disabled"
        shift
        ;;
    --help|-h)
        echo "Crush Development Server"
        echo ""
        echo "Usage: $0 [options]"
        echo ""
        echo "Options:"
        echo "  --no-vllm      Skip vLLM server startup"
        echo "  --no-profiling Disable profiling"
        echo "  --no-watch     Disable file watching and auto-restart"
        echo "  --help         Show this help"
        echo ""
        echo "Environment Variables:"
        echo "  VLLM_PORT=${VLLM_PORT}"
        echo "  PROFILE_PORT=${PROFILE_PORT}"
        echo "  MONITOR_INTERVAL=${MONITOR_INTERVAL}"
        echo "  AUTO_RESTART=${AUTO_RESTART}"
        echo "  ENABLE_PROFILING=${ENABLE_PROFILING}"
        exit 0
        ;;
esac

# Run main function
main "$@"