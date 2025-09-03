#!/usr/bin/env bash
# Advanced system monitoring for Crush vLLM development
set -euo pipefail

# Colors
RED='\033[31m'
GREEN='\033[32m'
YELLOW='\033[33m'
BLUE='\033[34m'
CYAN='\033[36m'
PURPLE='\033[35m'
RESET='\033[0m'

log() { echo -e "${GREEN}[MONITOR]${RESET} $1"; }
warn() { echo -e "${YELLOW}[WARN]${RESET} $1"; }
error() { echo -e "${RED}[ERROR]${RESET} $1"; exit 1; }
info() { echo -e "${BLUE}[INFO]${RESET} $1"; }

# Configuration
MONITOR_INTERVAL=${MONITOR_INTERVAL:-2}
LOG_FILE="logs/system-monitor.log"
CRUSH_BINARY="./crush"
VLLM_PORT=${VLLM_PORT:-8000}
PROFILE_PORT=${PROFILE_PORT:-6060}

# Create logs directory
mkdir -p logs

# System monitoring functions
get_system_info() {
    echo "=== System Information $(date) ===" >> "$LOG_FILE"
    
    # CPU Info
    local cpu_count=$(nproc)
    local load_avg=$(uptime | grep -o 'load average: [0-9.]*' | grep -o '[0-9.]*')
    echo "CPU Cores: $cpu_count" >> "$LOG_FILE"
    echo "Load Average: $load_avg" >> "$LOG_FILE"
    
    # Memory Info
    local mem_info=$(free -h | grep '^Mem:')
    echo "Memory: $mem_info" >> "$LOG_FILE"
    
    # GPU Info (if available)
    if command -v nvidia-smi &>/dev/null; then
        echo "=== GPU Information ===" >> "$LOG_FILE"
        nvidia-smi --query-gpu=name,memory.total,memory.used,memory.free,utilization.gpu,temperature.gpu --format=csv,noheader,nounits >> "$LOG_FILE"
    fi
    
    echo "" >> "$LOG_FILE"
}

get_process_info() {
    local process_name="$1"
    local pid="$2"
    
    if kill -0 "$pid" 2>/dev/null; then
        local cpu_percent=$(ps -p "$pid" -o %cpu --no-headers | tr -d ' ')
        local mem_percent=$(ps -p "$pid" -o %mem --no-headers | tr -d ' ')
        local mem_rss=$(ps -p "$pid" -o rss --no-headers | tr -d ' ')
        local mem_mb=$((mem_rss / 1024))
        
        echo "$process_name (PID: $pid): CPU: ${cpu_percent}%, Memory: ${mem_percent}% (${mem_mb}MB)"
        echo "$(date '+%Y-%m-%d %H:%M:%S') $process_name PID:$pid CPU:${cpu_percent}% MEM:${mem_mb}MB" >> "$LOG_FILE"
        
        return 0
    else
        echo "$process_name: Process not running"
        return 1
    fi
}

get_network_info() {
    echo "=== Network Status $(date) ===" >> "$LOG_FILE"
    
    # Check if vLLM port is listening
    if netstat -tuln | grep ":$VLLM_PORT " > /dev/null 2>&1; then
        echo "vLLM Server: Listening on port $VLLM_PORT" | tee -a "$LOG_FILE"
    else
        echo "vLLM Server: Not listening on port $VLLM_PORT" | tee -a "$LOG_FILE"
    fi
    
    # Check if profiling port is listening
    if netstat -tuln | grep ":$PROFILE_PORT " > /dev/null 2>&1; then
        echo "Crush Profiler: Listening on port $PROFILE_PORT" | tee -a "$LOG_FILE"
    else
        echo "Crush Profiler: Not listening on port $PROFILE_PORT" | tee -a "$LOG_FILE"
    fi
    
    # Check connection counts
    local vllm_connections=$(netstat -an | grep ":$VLLM_PORT " | grep ESTABLISHED | wc -l)
    echo "vLLM Active Connections: $vllm_connections" | tee -a "$LOG_FILE"
    
    echo "" >> "$LOG_FILE"
}

get_disk_info() {
    echo "=== Disk Usage $(date) ===" >> "$LOG_FILE"
    
    # Project directory usage
    local project_size=$(du -sh . 2>/dev/null | cut -f1)
    echo "Project Size: $project_size" | tee -a "$LOG_FILE"
    
    # Specific directories
    for dir in logs cache models venv; do
        if [[ -d "$dir" ]]; then
            local dir_size=$(du -sh "$dir" 2>/dev/null | cut -f1)
            echo "$dir/: $dir_size" | tee -a "$LOG_FILE"
        fi
    done
    
    # Available space
    local free_space=$(df -h . | tail -1 | awk '{print $4}')
    echo "Available Space: $free_space" | tee -a "$LOG_FILE"
    
    echo "" >> "$LOG_FILE"
}

monitor_vllm_health() {
    if curl -s "http://localhost:$VLLM_PORT/health" > /dev/null 2>&1; then
        echo -e "vLLM Health: ${GREEN}✓ Healthy${RESET}"
        echo "$(date '+%Y-%m-%d %H:%M:%S') vLLM Health: OK" >> "$LOG_FILE"
        return 0
    else
        echo -e "vLLM Health: ${RED}✗ Unhealthy${RESET}"
        echo "$(date '+%Y-%m-%d %H:%M:%S') vLLM Health: FAIL" >> "$LOG_FILE"
        return 1
    fi
}

monitor_crush_profile() {
    if curl -s "http://localhost:$PROFILE_PORT/debug/pprof/" > /dev/null 2>&1; then
        echo -e "Crush Profiler: ${GREEN}✓ Available${RESET}"
        
        # Get goroutine count
        local goroutines=$(curl -s "http://localhost:$PROFILE_PORT/debug/pprof/goroutine?debug=1" | grep -c "^goroutine " || echo "0")
        echo "  Goroutines: $goroutines"
        echo "$(date '+%Y-%m-%d %H:%M:%S') Crush Goroutines: $goroutines" >> "$LOG_FILE"
        
        return 0
    else
        echo -e "Crush Profiler: ${RED}✗ Not available${RESET}"
        return 1
    fi
}

show_performance_summary() {
    echo ""
    echo -e "${PURPLE}=== Performance Summary ===${RESET}"
    
    # System load
    local load_avg=$(uptime | grep -o 'load average: [0-9.]*' | grep -o '[0-9.]*')
    local cpu_cores=$(nproc)
    echo -e "System Load: $load_avg / $cpu_cores cores"
    
    # Memory usage
    local mem_used=$(free | grep '^Mem:' | awk '{printf "%.1f", $3/$2 * 100.0}')
    echo -e "Memory Usage: ${mem_used}%"
    
    # GPU usage (if available)
    if command -v nvidia-smi &>/dev/null; then
        local gpu_util=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits | head -1)
        local gpu_mem_used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
        local gpu_mem_total=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)
        local gpu_mem_percent=$(echo "scale=1; $gpu_mem_used * 100 / $gpu_mem_total" | bc)
        echo -e "GPU Utilization: ${gpu_util}%"
        echo -e "GPU Memory: ${gpu_mem_percent}% (${gpu_mem_used}MB/${gpu_mem_total}MB)"
    fi
}

interactive_monitor() {
    echo -e "${BLUE}🔍 Interactive System Monitor${RESET}"
    echo "Press 'q' to quit, 'r' to refresh, 'p' for process details, 'h' for help"
    echo ""
    
    while true; do
        # Clear screen and show header
        clear
        echo -e "${CYAN}=== Crush vLLM Development Monitor ===${RESET}"
        echo "$(date)"
        echo ""
        
        # Find PIDs
        local crush_pid=""
        local vllm_pid=""
        
        if [[ -f ".dev-server.pid" ]]; then
            local dev_pid=$(cat .dev-server.pid 2>/dev/null || echo "")
            if [[ -n "$dev_pid" ]] && kill -0 "$dev_pid" 2>/dev/null; then
                echo -e "Development Server: ${GREEN}Running (PID: $dev_pid)${RESET}"
            fi
        fi
        
        if [[ -f "test-vllm.pid" ]]; then
            vllm_pid=$(cat test-vllm.pid 2>/dev/null || echo "")
        elif [[ -f ".vllm-server.pid" ]]; then
            vllm_pid=$(cat .vllm-server.pid 2>/dev/null || echo "")
        fi
        
        crush_pid=$(pgrep -f "$CRUSH_BINARY" | head -1 || echo "")
        if [[ -z "$vllm_pid" ]]; then
            vllm_pid=$(pgrep -f "vllm.entrypoints.openai.api_server" | head -1 || echo "")
        fi
        
        # Process monitoring
        echo -e "${YELLOW}=== Process Status ===${RESET}"
        if [[ -n "$crush_pid" ]]; then
            get_process_info "Crush" "$crush_pid"
        else
            echo -e "Crush: ${RED}Not running${RESET}"
        fi
        
        if [[ -n "$vllm_pid" ]]; then
            get_process_info "vLLM" "$vllm_pid"
        else
            echo -e "vLLM: ${RED}Not running${RESET}"
        fi
        
        echo ""
        
        # Health checks
        echo -e "${YELLOW}=== Health Status ===${RESET}"
        monitor_vllm_health
        monitor_crush_profile
        
        echo ""
        
        # Performance summary
        show_performance_summary
        
        echo ""
        echo -e "${CYAN}Press 'q' to quit, 'r' to refresh, 'p' for process details${RESET}"
        
        # Read user input with timeout
        if read -t "$MONITOR_INTERVAL" -n 1 key; then
            case "$key" in
                q|Q)
                    echo ""
                    log "Monitoring stopped"
                    break
                    ;;
                r|R)
                    continue
                    ;;
                p|P)
                    echo ""
                    echo -e "${BLUE}=== Detailed Process Information ===${RESET}"
                    
                    if [[ -n "$crush_pid" ]]; then
                        echo "Crush Process Details:"
                        ps -p "$crush_pid" -o pid,ppid,cmd,etime,pcpu,pmem,rss,vsz
                        echo ""
                    fi
                    
                    if [[ -n "$vllm_pid" ]]; then
                        echo "vLLM Process Details:"
                        ps -p "$vllm_pid" -o pid,ppid,cmd,etime,pcpu,pmem,rss,vsz
                        echo ""
                    fi
                    
                    echo "Press any key to continue..."
                    read -n 1
                    ;;
                h|H)
                    echo ""
                    echo -e "${BLUE}=== Monitor Help ===${RESET}"
                    echo "q/Q - Quit monitoring"
                    echo "r/R - Refresh immediately"
                    echo "p/P - Show detailed process information"
                    echo "h/H - Show this help"
                    echo ""
                    echo "Monitoring refreshes automatically every ${MONITOR_INTERVAL} seconds"
                    echo "Press any key to continue..."
                    read -n 1
                    ;;
            esac
        fi
    done
}

continuous_monitor() {
    log "Starting continuous monitoring (interval: ${MONITOR_INTERVAL}s)"
    log "Logging to: $LOG_FILE"
    
    # Initial system info
    get_system_info
    
    local iteration=0
    while true; do
        iteration=$((iteration + 1))
        
        echo -e "\n${CYAN}=== Monitor Iteration $iteration ($(date)) ===${RESET}"
        
        # Find PIDs
        local crush_pid=$(pgrep -f "$CRUSH_BINARY" | head -1 || echo "")
        local vllm_pid=""
        
        if [[ -f ".vllm-server.pid" ]]; then
            vllm_pid=$(cat .vllm-server.pid 2>/dev/null || echo "")
        fi
        if [[ -z "$vllm_pid" ]]; then
            vllm_pid=$(pgrep -f "vllm.entrypoints.openai.api_server" | head -1 || echo "")
        fi
        
        # Monitor processes
        if [[ -n "$crush_pid" ]]; then
            get_process_info "Crush" "$crush_pid"
        else
            echo -e "Crush: ${RED}Not running${RESET}"
        fi
        
        if [[ -n "$vllm_pid" ]]; then
            get_process_info "vLLM" "$vllm_pid"
        else
            echo -e "vLLM: ${RED}Not running${RESET}"
        fi
        
        # Health checks
        monitor_vllm_health
        monitor_crush_profile
        
        # System info every 10 iterations
        if (( iteration % 10 == 0 )); then
            get_system_info
            get_network_info
            get_disk_info
        fi
        
        sleep "$MONITOR_INTERVAL"
    done
}

# Performance profiling helpers
start_profiling() {
    local duration="${1:-30}"
    local output_dir="profiling/$(date +%Y%m%d_%H%M%S)"
    
    mkdir -p "$output_dir"
    
    log "Starting $duration second profiling session..."
    log "Output directory: $output_dir"
    
    if curl -s "http://localhost:$PROFILE_PORT/debug/pprof/" > /dev/null; then
        # CPU Profile
        log "Capturing CPU profile..."
        curl -s "http://localhost:$PROFILE_PORT/debug/pprof/profile?seconds=$duration" > "$output_dir/cpu.prof" &
        
        # Memory Profile
        log "Capturing memory profile..."
        curl -s "http://localhost:$PROFILE_PORT/debug/pprof/heap" > "$output_dir/heap.prof" &
        
        # Goroutine Profile
        curl -s "http://localhost:$PROFILE_PORT/debug/pprof/goroutine" > "$output_dir/goroutine.prof" &
        
        wait
        
        log "Profiling complete. Use these commands to analyze:"
        echo "  go tool pprof $output_dir/cpu.prof"
        echo "  go tool pprof $output_dir/heap.prof"
        echo "  go tool pprof $output_dir/goroutine.prof"
    else
        error "Profiling endpoint not available. Start Crush with CRUSH_PROFILE=true"
    fi
}

# Main execution
main() {
    case "${1:-interactive}" in
        interactive|i)
            interactive_monitor
            ;;
        continuous|c)
            continuous_monitor
            ;;
        profile)
            start_profiling "${2:-30}"
            ;;
        info)
            get_system_info
            get_network_info
            get_disk_info
            ;;
        --help|-h)
            echo "Crush vLLM System Monitor"
            echo ""
            echo "Usage: $0 [mode] [options]"
            echo ""
            echo "Modes:"
            echo "  interactive (default)  - Interactive monitoring dashboard"
            echo "  continuous            - Continuous monitoring with logging"
            echo "  profile [seconds]     - Capture profiling data"
            echo "  info                  - Show system information once"
            echo ""
            echo "Environment Variables:"
            echo "  MONITOR_INTERVAL=$MONITOR_INTERVAL    - Refresh interval (seconds)"
            echo "  VLLM_PORT=$VLLM_PORT             - vLLM server port"
            echo "  PROFILE_PORT=$PROFILE_PORT          - Profiling port"
            exit 0
            ;;
        *)
            error "Unknown mode: $1. Use --help for usage information."
            ;;
    esac
}

# Run main function
main "$@"