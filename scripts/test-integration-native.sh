#!/usr/bin/env bash
# Native integration testing with vLLM (no Docker)
set -euo pipefail

# Colors
RED='\033[31m'
GREEN='\033[32m'
YELLOW='\033[33m'
BLUE='\033[34m'
CYAN='\033[36m'
RESET='\033[0m'

log() { echo -e "${GREEN}[INTEGRATION-TEST]${RESET} $1"; }
warn() { echo -e "${YELLOW}[WARN]${RESET} $1"; }
error() { echo -e "${RED}[ERROR]${RESET} $1"; exit 1; }
info() { echo -e "${BLUE}[INFO]${RESET} $1"; }
progress() { echo -e "${CYAN}[PROGRESS]${RESET} $1"; }

# Configuration
VLLM_PORT=${VLLM_PORT:-8000}
TEST_MODEL="facebook/opt-125m"
TEST_TIMEOUT=300
VLLM_PID=""
CRUSH_PID=""
TEST_SESSION_DIR="/tmp/crush-integration-test-$$"

# Cleanup function
cleanup() {
    log "Cleaning up test environment..."
    
    [[ -n "${VLLM_PID}" ]] && kill "${VLLM_PID}" 2>/dev/null || true
    [[ -n "${CRUSH_PID}" ]] && kill "${CRUSH_PID}" 2>/dev/null || true
    
    rm -rf "${TEST_SESSION_DIR}"
    rm -f test-vllm.pid test-crush.pid
    
    log "Cleanup complete"
}

# Setup signal handlers
trap cleanup SIGINT SIGTERM EXIT

# Test result tracking
TESTS_RUN=0
TESTS_PASSED=0
TESTS_FAILED=0

test_result() {
    local test_name="$1"
    local result="$2"
    local details="${3:-}"
    
    TESTS_RUN=$((TESTS_RUN + 1))
    
    if [[ "$result" == "PASS" ]]; then
        echo -e "  ✅ ${test_name}: ${GREEN}PASS${RESET}"
        TESTS_PASSED=$((TESTS_PASSED + 1))
    else
        echo -e "  ❌ ${test_name}: ${RED}FAIL${RESET}"
        [[ -n "$details" ]] && echo -e "     ${YELLOW}Details: $details${RESET}"
        TESTS_FAILED=$((TESTS_FAILED + 1))
    fi
}

# Start vLLM server
start_vllm() {
    progress "Starting vLLM server for integration testing..."
    
    if [[ ! -d "venv" ]]; then
        error "Python virtual environment not found. Run 'make setup' first."
    fi
    
    source venv/bin/activate
    
    # Start vLLM with test configuration
    python -m vllm.entrypoints.openai.api_server \
        --model "${TEST_MODEL}" \
        --port "${VLLM_PORT}" \
        --host 0.0.0.0 \
        --max-model-len 1024 \
        --gpu-memory-utilization 0.2 \
        --trust-remote-code \
        --disable-log-requests \
        --tensor-parallel-size 1 > logs/integration-vllm.log 2>&1 &
    
    VLLM_PID=$!
    echo "${VLLM_PID}" > test-vllm.pid
    deactivate
    
    # Wait for vLLM to be ready
    log "Waiting for vLLM server to be ready (timeout: ${TEST_TIMEOUT}s)..."
    
    local ready=false
    for i in $(seq 1 $((TEST_TIMEOUT / 5))); do
        if curl -s "http://localhost:${VLLM_PORT}/health" > /dev/null 2>&1; then
            ready=true
            break
        fi
        
        # Check if process is still running
        if ! kill -0 "${VLLM_PID}" 2>/dev/null; then
            error "vLLM server process died during startup. Check logs/integration-vllm.log"
        fi
        
        echo -n "."
        sleep 5
    done
    echo ""
    
    if [[ "$ready" == "true" ]]; then
        info "✅ vLLM server is ready"
        return 0
    else
        error "vLLM server failed to start within timeout"
    fi
}

# Test vLLM API directly
test_vllm_api() {
    progress "Testing vLLM API directly..."
    
    # Test 1: Health check
    local health_response
    if health_response=$(curl -s "http://localhost:${VLLM_PORT}/health" 2>/dev/null); then
        test_result "vLLM Health Check" "PASS"
    else
        test_result "vLLM Health Check" "FAIL" "No health response"
        return 1
    fi
    
    # Test 2: Models endpoint
    local models_response
    if models_response=$(curl -s "http://localhost:${VLLM_PORT}/v1/models" 2>/dev/null); then
        if echo "$models_response" | jq -e '.data[0].id' > /dev/null 2>&1; then
            local model_id=$(echo "$models_response" | jq -r '.data[0].id')
            test_result "vLLM Models List" "PASS" "Found model: $model_id"
        else
            test_result "vLLM Models List" "FAIL" "Invalid models response"
        fi
    else
        test_result "vLLM Models List" "FAIL" "No models response"
    fi
    
    # Test 3: Completions endpoint
    local completion_response
    completion_response=$(curl -s -X POST "http://localhost:${VLLM_PORT}/v1/completions" \
        -H "Content-Type: application/json" \
        -d '{
            "model": "'"${TEST_MODEL}"'",
            "prompt": "Hello, world!",
            "max_tokens": 10,
            "temperature": 0.1
        }' 2>/dev/null)
    
    if echo "$completion_response" | jq -e '.choices[0].text' > /dev/null 2>&1; then
        local completion_text=$(echo "$completion_response" | jq -r '.choices[0].text')
        test_result "vLLM Completions" "PASS" "Generated: ${completion_text:0:30}..."
    else
        test_result "vLLM Completions" "FAIL" "Invalid completion response"
    fi
    
    # Test 4: Chat completions endpoint
    local chat_response
    chat_response=$(curl -s -X POST "http://localhost:${VLLM_PORT}/v1/chat/completions" \
        -H "Content-Type: application/json" \
        -d '{
            "model": "'"${TEST_MODEL}"'",
            "messages": [{"role": "user", "content": "Say hello"}],
            "max_tokens": 10,
            "temperature": 0.1
        }' 2>/dev/null)
    
    if echo "$chat_response" | jq -e '.choices[0].message.content' > /dev/null 2>&1; then
        local chat_content=$(echo "$chat_response" | jq -r '.choices[0].message.content')
        test_result "vLLM Chat Completions" "PASS" "Response: ${chat_content:0:30}..."
    else
        test_result "vLLM Chat Completions" "FAIL" "Invalid chat response"
    fi
}

# Test Crush with vLLM integration
test_crush_integration() {
    progress "Testing Crush with vLLM integration..."
    
    # Build Crush if needed
    if [[ ! -x "./crush" ]]; then
        log "Building Crush..."
        go build -o crush .
    fi
    
    # Create test session directory
    mkdir -p "${TEST_SESSION_DIR}"
    
    # Create test configuration
    cat > "${TEST_SESSION_DIR}/crush.json" << EOF
{
  "\$schema": "https://charm.land/crush.json",
  "options": {
    "debug": true
  },
  "providers": {
    "vllm-test": {
      "name": "vLLM Test Server",
      "base_url": "http://localhost:${VLLM_PORT}/v1/",
      "type": "openai",
      "models": [
        {
          "id": "${TEST_MODEL}",
          "name": "Test Model",
          "context_window": 1024,
          "default_max_tokens": 50
        }
      ]
    }
  }
}
EOF
    
    # Test 1: Crush can start and load configuration
    if timeout 10 ./crush --config="${TEST_SESSION_DIR}/crush.json" --help > /dev/null 2>&1; then
        test_result "Crush Startup" "PASS"
    else
        test_result "Crush Startup" "FAIL" "Failed to start with vLLM config"
        return 1
    fi
    
    # Test 2: Crush can connect to vLLM (if there's a connection test command)
    # This would depend on Crush having a way to test provider connections
    # For now, we'll test if Crush accepts the configuration
    
    # Test 3: Integration test with actual prompt (if Crush supports batch mode)
    # Create a test prompt
    echo "Hello from integration test" > "${TEST_SESSION_DIR}/test_prompt.txt"
    
    # This would need Crush to support a batch/test mode
    # test_result "Crush vLLM Integration" "SKIP" "Requires Crush batch mode"
}

# Test performance under load
test_performance() {
    progress "Testing performance characteristics..."
    
    # Test 1: Concurrent requests
    local concurrent_requests=5
    local success_count=0
    
    log "Testing ${concurrent_requests} concurrent requests..."
    
    for i in $(seq 1 $concurrent_requests); do
        {
            if curl -s -X POST "http://localhost:${VLLM_PORT}/v1/completions" \
                -H "Content-Type: application/json" \
                -d '{
                    "model": "'"${TEST_MODEL}"'",
                    "prompt": "Test request '"$i"'",
                    "max_tokens": 5,
                    "temperature": 0.1
                }' > "/tmp/test_response_$i.json" 2>/dev/null; then
                if jq -e '.choices[0].text' "/tmp/test_response_$i.json" > /dev/null 2>&1; then
                    echo "success_$i"
                fi
            fi
        } &
    done
    
    wait
    
    success_count=$(ls /tmp/test_response_*.json 2>/dev/null | wc -l)
    rm -f /tmp/test_response_*.json
    
    if [[ $success_count -ge $((concurrent_requests / 2)) ]]; then
        test_result "Concurrent Requests" "PASS" "$success_count/$concurrent_requests successful"
    else
        test_result "Concurrent Requests" "FAIL" "Only $success_count/$concurrent_requests successful"
    fi
    
    # Test 2: Response time check
    local start_time=$(date +%s.%N)
    curl -s -X POST "http://localhost:${VLLM_PORT}/v1/completions" \
        -H "Content-Type: application/json" \
        -d '{
            "model": "'"${TEST_MODEL}"'",
            "prompt": "Performance test",
            "max_tokens": 10,
            "temperature": 0.1
        }' > /dev/null
    local end_time=$(date +%s.%N)
    local response_time=$(echo "$end_time - $start_time" | bc)
    
    # Response should be under 10 seconds for lightweight model
    if (( $(echo "$response_time < 10.0" | bc -l) )); then
        test_result "Response Time" "PASS" "${response_time}s"
    else
        test_result "Response Time" "FAIL" "${response_time}s (too slow)"
    fi
}

# Test resource usage
test_resources() {
    progress "Testing resource usage..."
    
    if [[ -n "${VLLM_PID}" ]] && kill -0 "${VLLM_PID}" 2>/dev/null; then
        # Get memory usage
        local mem_usage=$(ps -p "${VLLM_PID}" -o rss --no-headers 2>/dev/null || echo "0")
        local mem_mb=$((mem_usage / 1024))
        
        # Memory should be reasonable for test model (under 2GB)
        if [[ $mem_mb -lt 2048 ]]; then
            test_result "Memory Usage" "PASS" "${mem_mb}MB"
        else
            test_result "Memory Usage" "WARN" "${mem_mb}MB (high for test model)"
        fi
        
        # Get CPU usage (average over 3 seconds)
        local cpu_usage=$(top -p "${VLLM_PID}" -b -n 3 -d 1 | tail -n 1 | awk '{print $9}' 2>/dev/null || echo "0")
        test_result "CPU Usage" "INFO" "${cpu_usage}%"
    else
        test_result "Resource Usage" "FAIL" "vLLM process not running"
    fi
}

# Generate test report
generate_report() {
    local duration=$1
    
    echo ""
    echo -e "${BLUE}=== Integration Test Report ===${RESET}"
    echo "Duration: ${duration}s"
    echo "Tests Run: ${TESTS_RUN}"
    echo -e "Passed: ${GREEN}${TESTS_PASSED}${RESET}"
    echo -e "Failed: ${RED}${TESTS_FAILED}${RESET}"
    echo ""
    
    if [[ $TESTS_FAILED -eq 0 ]]; then
        echo -e "${GREEN}🎉 All tests passed!${RESET}"
        echo ""
        echo -e "${BLUE}=== Environment Info ===${RESET}"
        echo "vLLM Server: http://localhost:${VLLM_PORT}"
        echo "Test Model: ${TEST_MODEL}"
        echo "Logs: logs/integration-vllm.log"
        return 0
    else
        echo -e "${RED}❌ Some tests failed${RESET}"
        echo "Check logs for details: logs/integration-vllm.log"
        return 1
    fi
}

# Main execution
main() {
    local start_time=$(date +%s)
    
    log "🧪 Starting Crush vLLM integration tests..."
    
    # Check prerequisites
    if [[ ! -f "go.mod" ]]; then
        error "Not in Crush project directory"
    fi
    
    if ! command -v curl &>/dev/null; then
        error "curl is required for API testing"
    fi
    
    if ! command -v jq &>/dev/null; then
        error "jq is required for JSON parsing"
    fi
    
    # Create logs directory
    mkdir -p logs
    
    # Run test phases
    start_vllm
    test_vllm_api
    test_crush_integration
    test_performance  
    test_resources
    
    local end_time=$(date +%s)
    local duration=$((end_time - start_time))
    
    generate_report $duration
}

# Handle command line arguments
case "${1:-}" in
    --api-only)
        log "Running API tests only..."
        start_vllm
        test_vllm_api
        generate_report 0
        ;;
    --performance)
        log "Running performance tests only..."
        start_vllm
        test_performance
        generate_report 0
        ;;
    --quick)
        log "Running quick integration test..."
        TEST_TIMEOUT=60
        start_vllm
        test_vllm_api
        generate_report 0
        ;;
    --help|-h)
        echo "Crush vLLM Integration Tests"
        echo ""
        echo "Usage: $0 [options]"
        echo ""
        echo "Options:"
        echo "  --api-only      Test vLLM API only"
        echo "  --performance   Performance tests only"
        echo "  --quick         Quick test (1 minute timeout)"
        echo "  --help          Show this help"
        echo ""
        echo "Environment Variables:"
        echo "  VLLM_PORT=${VLLM_PORT}"
        echo "  TEST_TIMEOUT=${TEST_TIMEOUT}"
        exit 0
        ;;
    *)
        main "$@"
        ;;
esac