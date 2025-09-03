# Crush vLLM Development Makefile  
# Ultra-optimized native workflow (no Docker) for development, testing, and deployment

.PHONY: help setup dev test build clean vllm monitor profile deploy ci docs quick-*
.DEFAULT_GOAL := help

# Configuration
PROJECT_NAME := crush
BINARY_NAME := crush
GO_VERSION := 1.24
VERSION := $(shell git describe --tags --dirty --always 2>/dev/null || echo "dev")
BUILD_TIME := $(shell date -u +"%Y-%m-%dT%H:%M:%SZ")
COMMIT := $(shell git rev-parse --short HEAD 2>/dev/null || echo "unknown")

# Build flags (optimized for development)
LDFLAGS := -ldflags "-X main.version=$(VERSION) -X main.buildTime=$(BUILD_TIME) -X main.commit=$(COMMIT)"
CGO_ENABLED := 0
GOEXPERIMENT := greenteagc

# Development configuration
VENV_PATH := venv
VLLM_PORT := 8000
PROFILE_PORT := 6060

# Colors for output
RED := \033[31m
GREEN := \033[32m
YELLOW := \033[33m
BLUE := \033[34m
CYAN := \033[36m
RESET := \033[0m

help: ## Show this help message with quick commands
	@echo "$(BLUE)🚀 Crush vLLM Development Workflow (Native)$(RESET)"
	@echo "=============================================="
	@echo ""
	@echo "$(CYAN)⚡ Quick Start (choose one):$(RESET)"
	@echo "  $(GREEN)make quick-start$(RESET)    - Complete setup + start development server"
	@echo "  $(GREEN)make quick-dev$(RESET)      - Fast dev server (if already setup)"
	@echo "  $(GREEN)make quick-test$(RESET)     - Run essential tests quickly"
	@echo ""
	@awk 'BEGIN {FS = ":.*##"; printf "$(GREEN)All available targets:$(RESET)\n"} /^[a-zA-Z_-]+:.*?##/ { printf "  $(YELLOW)%-20s$(RESET) %s\n", $$1, $$2 } /^##@/ { printf "\n$(BLUE)%s$(RESET)\n", substr($$0, 5) }' $(MAKEFILE_LIST)

##@ ⚡ Ultra-Fast Commands
quick-start: ## Complete setup + start development server (one command)
	@echo "$(CYAN)🚀 Ultra-fast complete setup...$(RESET)"
	@./scripts/quick-setup.sh && make quick-dev

quick-dev: ## Start development server (hot reload, profiling enabled)
	@echo "$(CYAN)⚡ Starting development server...$(RESET)"
	@./scripts/dev/hot-reload.sh

quick-test: ## Run essential tests (< 30 seconds)
	@echo "$(CYAN)🧪 Running quick tests...$(RESET)"
	@./scripts/dev/quick-test.sh

quick-check: ## Rapid health check for development environment
	@echo "$(CYAN)🔍 Quick environment check...$(RESET)"
	@echo -n "Go: "; go version 2>/dev/null || echo "$(RED)Missing$(RESET)"
	@echo -n "Python: "; python3 --version 2>/dev/null || echo "$(RED)Missing$(RESET)"
	@echo -n "vLLM: "; [[ -d $(VENV_PATH) ]] && echo "$(GREEN)Ready$(RESET)" || echo "$(YELLOW)Not setup$(RESET)"
	@echo -n "Crush binary: "; [[ -x "./$(BINARY_NAME)" ]] && echo "$(GREEN)Built$(RESET)" || echo "$(YELLOW)Not built$(RESET)"

##@ 🔨 Development & Build
setup: ## Complete development environment setup
	@echo "$(GREEN)Setting up development environment...$(RESET)"
	@./scripts/quick-setup.sh

dev: ## Start development environment with hot reload
	@echo "$(GREEN)Starting development environment...$(RESET)"
	@./scripts/dev-server.sh

dev-no-vllm: ## Start development server without vLLM
	@echo "$(GREEN)Starting development server (no vLLM)...$(RESET)"
	@./scripts/dev-server.sh --no-vllm

##@ 🔨 Build & Test  
build: ## Build optimized binary
	@echo "$(GREEN)Building $(PROJECT_NAME)...$(RESET)"
	@CGO_ENABLED=$(CGO_ENABLED) GOEXPERIMENT=$(GOEXPERIMENT) go build $(LDFLAGS) -o $(BINARY_NAME) .
	@echo "$(GREEN)✅ Built $(BINARY_NAME)$(RESET)"

build-dev: ## Build for development with debugging
	@echo "$(GREEN)Building development binary...$(RESET)"
	@CGO_ENABLED=1 go build -race $(LDFLAGS) -o $(BINARY_NAME)-dev .

build-fast: ## Fast build (no optimization)
	@echo "$(GREEN)Fast build...$(RESET)"
	@go build -o $(BINARY_NAME) .

test: ## Run unit tests with race detection
	@echo "$(GREEN)Running unit tests...$(RESET)"
	@go test -v ./... -race -timeout=60s -short

test-all: ## Run all tests (unit + integration)
	@echo "$(GREEN)Running all tests...$(RESET)"
	@make test
	@make test-integration

test-integration: ## Run integration tests with vLLM (native)
	@echo "$(GREEN)Running native integration tests...$(RESET)"
	@./scripts/test-integration-native.sh

test-integration-quick: ## Run quick integration tests
	@echo "$(GREEN)Running quick integration tests...$(RESET)"
	@./scripts/test-integration-native.sh --quick

test-coverage: ## Generate test coverage report
	@echo "$(GREEN)Generating coverage report...$(RESET)"
	@mkdir -p coverage
	@go test -v ./... -coverprofile=coverage/coverage.out -covermode=atomic
	@go tool cover -html=coverage/coverage.out -o coverage/coverage.html
	@go tool cover -func=coverage/coverage.out | grep total | awk '{print "Total coverage: " $$3}'
	@echo "$(GREEN)Coverage report: coverage/coverage.html$(RESET)"

benchmark: ## Run performance benchmarks
	@echo "$(GREEN)Running benchmarks...$(RESET)"
	@mkdir -p benchmarks
	@go test -bench=. -benchmem -benchtime=3s ./... | tee benchmarks/latest.txt

lint: ## Run linters (fast)
	@echo "$(GREEN)Running linters...$(RESET)"
	@golangci-lint run --timeout=3m --fast

lint-fix: ## Run linters with auto-fix
	@echo "$(GREEN)Auto-fixing lint issues...$(RESET)"
	@golangci-lint run --timeout=5m --fix

fmt: ## Format code
	@echo "$(GREEN)Formatting code...$(RESET)"
	@gofumpt -w .
	@echo "$(GREEN)✅ Code formatted$(RESET)"

clean: ## Clean build artifacts and caches
	@echo "$(GREEN)Cleaning build artifacts...$(RESET)"
	@rm -f $(BINARY_NAME) $(BINARY_NAME)-dev $(BINARY_NAME)-race
	@rm -rf coverage benchmarks profiling
	@go clean -cache -testcache -modcache
	@echo "$(GREEN)✅ Clean complete$(RESET)"

##@ 🤖 vLLM Operations (Native)
vllm-setup: ## Setup vLLM Python environment  
	@echo "$(GREEN)Setting up vLLM environment...$(RESET)"
	@if [[ ! -d "$(VENV_PATH)" ]]; then python3 -m venv $(VENV_PATH); fi
	@source $(VENV_PATH)/bin/activate && pip install --upgrade pip uv
	@source $(VENV_PATH)/bin/activate && uv pip install vllm torch huggingface-hub transformers

vllm-start: ## Start vLLM server (lightweight model)
	@echo "$(GREEN)Starting vLLM server on port $(VLLM_PORT)...$(RESET)"
	@if [[ ! -f ".vllm-server.pid" ]]; then \
		source $(VENV_PATH)/bin/activate && \
		nohup python -m vllm.entrypoints.openai.api_server \
			--model facebook/opt-125m \
			--port $(VLLM_PORT) \
			--host 0.0.0.0 \
			--gpu-memory-utilization 0.3 \
			--max-model-len 2048 \
			--trust-remote-code > logs/vllm.log 2>&1 & \
		echo $$! > .vllm-server.pid && \
		echo "$(GREEN)vLLM server started (PID: $$(cat .vllm-server.pid))$(RESET)"; \
	else \
		echo "$(YELLOW)vLLM server already running$(RESET)"; \
	fi

vllm-stop: ## Stop vLLM server
	@echo "$(GREEN)Stopping vLLM server...$(RESET)"
	@if [[ -f ".vllm-server.pid" ]]; then \
		kill $$(cat .vllm-server.pid) 2>/dev/null || true && \
		rm -f .vllm-server.pid && \
		echo "$(GREEN)vLLM server stopped$(RESET)"; \
	else \
		echo "$(YELLOW)vLLM server not running$(RESET)"; \
	fi

vllm-restart: ## Restart vLLM server
	@make vllm-stop
	@sleep 2
	@make vllm-start

vllm-health: ## Check vLLM server health
	@echo "$(GREEN)Checking vLLM server health...$(RESET)"
	@if curl -s http://localhost:$(VLLM_PORT)/health > /dev/null; then \
		echo "$(GREEN)✅ vLLM server is healthy$(RESET)"; \
	else \
		echo "$(RED)❌ vLLM server is not responding$(RESET)"; \
		exit 1; \
	fi

vllm-logs: ## Show vLLM server logs (follow)
	@echo "$(GREEN)Showing vLLM server logs...$(RESET)"
	@tail -f logs/vllm.log 2>/dev/null || echo "$(YELLOW)No vLLM logs found$(RESET)"

vllm-test: ## Test vLLM server with sample request
	@echo "$(GREEN)Testing vLLM server...$(RESET)"
	@curl -X POST http://localhost:$(VLLM_PORT)/v1/completions \
		-H "Content-Type: application/json" \
		-d '{"model": "facebook/opt-125m", "prompt": "Hello, world!", "max_tokens": 10}' | jq .

##@ 📊 Monitoring & Profiling (Native)
monitor: ## Interactive system monitor
	@echo "$(GREEN)Starting interactive system monitor...$(RESET)"
	@./scripts/monitor-system.sh interactive

monitor-continuous: ## Continuous monitoring with logging
	@echo "$(GREEN)Starting continuous monitoring...$(RESET)"
	@./scripts/monitor-system.sh continuous

profile: ## Capture profiling data (30 seconds)
	@echo "$(GREEN)Capturing profiling data...$(RESET)"
	@./scripts/monitor-system.sh profile 30

profile-cpu: ## Interactive CPU profiling
	@echo "$(GREEN)Starting CPU profiling (web UI)...$(RESET)"
	@if curl -s http://localhost:$(PROFILE_PORT)/debug/pprof/ > /dev/null; then \
		go tool pprof -http :6061 http://localhost:$(PROFILE_PORT)/debug/pprof/profile?seconds=30; \
	else \
		echo "$(RED)Profiling not available. Start Crush with CRUSH_PROFILE=true$(RESET)"; \
	fi

profile-heap: ## Interactive memory profiling
	@echo "$(GREEN)Starting memory profiling (web UI)...$(RESET)"
	@if curl -s http://localhost:$(PROFILE_PORT)/debug/pprof/ > /dev/null; then \
		go tool pprof -http :6061 http://localhost:$(PROFILE_PORT)/debug/pprof/heap; \
	else \
		echo "$(RED)Profiling not available. Start Crush with CRUSH_PROFILE=true$(RESET)"; \
	fi

perf-info: ## Show current performance information
	@echo "$(GREEN)Current performance status...$(RESET)"
	@./scripts/monitor-system.sh info

##@ 🚀 CI/CD & Deployment
ci: ## Run full CI pipeline locally
	@echo "$(GREEN)Running full CI pipeline...$(RESET)"
	@make fmt lint test benchmark security-scan

ci-quick: ## Run essential CI checks (fast)
	@echo "$(GREEN)Running essential CI checks...$(RESET)"
	@make fmt lint test

security-scan: ## Run security scans
	@echo "$(GREEN)Running security scans...$(RESET)"
	@if command -v gosec >/dev/null 2>&1; then gosec ./...; else echo "$(YELLOW)gosec not installed$(RESET)"; fi
	@if command -v govulncheck >/dev/null 2>&1; then govulncheck ./...; else echo "$(YELLOW)govulncheck not installed$(RESET)"; fi

install: ## Install Crush binary to GOPATH
	@echo "$(GREEN)Installing $(PROJECT_NAME)...$(RESET)"
	@go install $(LDFLAGS) .
	@echo "$(GREEN)✅ Installed to $$(go env GOPATH)/bin/$(BINARY_NAME)$(RESET)"

install-tools: ## Install development tools
	@echo "$(GREEN)Installing development tools...$(RESET)"
	@go install github.com/golangci/golangci-lint/cmd/golangci-lint@latest
	@go install mvdan.cc/gofumpt@latest
	@go install github.com/goreleaser/goreleaser/v2@latest
	@go install github.com/securecodewarrior/gosec/v2/cmd/gosec@latest
	@go install golang.org/x/vuln/cmd/govulncheck@latest
	@echo "$(GREEN)✅ Development tools installed$(RESET)"

##@ 📚 Documentation & Utilities
docs: ## Generate comprehensive project documentation
	@echo "$(GREEN)Generating comprehensive documentation...$(RESET)"
	@./scripts/generate-dev-docs.sh

schema: ## Generate JSON schema
	@echo "$(GREEN)Generating JSON schema...$(RESET)"
	@if [[ -x "./$(BINARY_NAME)" ]]; then \
		./$(BINARY_NAME) schema > schema.json && \
		echo "$(GREEN)✅ Schema generated: schema.json$(RESET)"; \
	else \
		echo "$(YELLOW)Binary not found. Run 'make build' first.$(RESET)"; \
	fi

update-deps: ## Update Go dependencies
	@echo "$(GREEN)Updating dependencies...$(RESET)"
	@go mod tidy
	@go mod verify
	@go list -u -m all | grep -E '\[(.*)\]$$' || echo "$(GREEN)All dependencies up to date$(RESET)"

##@ ℹ️  Information & Status
status: ## Show comprehensive development status
	@echo "$(CYAN)🔍 Development Environment Status$(RESET)"
	@echo "=================================="
	@make quick-check
	@echo ""
	@echo "$(BLUE)Services:$(RESET)"
	@if [[ -f ".vllm-server.pid" ]] && kill -0 $$(cat .vllm-server.pid) 2>/dev/null; then \
		echo "  vLLM Server: $(GREEN)✓ Running$(RESET)"; \
	else \
		echo "  vLLM Server: $(RED)✗ Stopped$(RESET)"; \
	fi
	@if pgrep -f "./$(BINARY_NAME)" > /dev/null; then \
		echo "  Crush App: $(GREEN)✓ Running$(RESET)"; \
	else \
		echo "  Crush App: $(RED)✗ Stopped$(RESET)"; \
	fi
	@echo ""
	@echo "$(BLUE)Directories:$(RESET)"
	@for dir in logs cache models coverage benchmarks; do \
		if [[ -d "$$dir" ]]; then \
			size=$$(du -sh "$$dir" 2>/dev/null | cut -f1); \
			echo "  $$dir/: $$size"; \
		fi; \
	done

version: ## Show version information
	@echo "$(BLUE)Crush vLLM Integration$(RESET)"
	@echo "Version: $(VERSION)"
	@echo "Commit:  $(COMMIT)"
	@echo "Built:   $(BUILD_TIME)"
	@echo "Go:      $$(go version | cut -d' ' -f3)"

deps: ## Show dependency information
	@echo "$(GREEN)Go Dependencies:$(RESET)"
	@go list -m -u all | head -20
	@echo ""
	@echo "$(GREEN)Python Dependencies (if venv exists):$(RESET)"
	@if [[ -d "$(VENV_PATH)" ]]; then \
		source $(VENV_PATH)/bin/activate && pip list | grep -E "(vllm|torch|transformers)" || echo "No ML packages found"; \
	else \
		echo "Virtual environment not found"; \
	fi

##@ 🧹 Maintenance & Cleanup
reset: ## Reset entire development environment
	@echo "$(YELLOW)⚠️  Resetting development environment...$(RESET)"
	@make vllm-stop
	@make clean
	@rm -rf $(VENV_PATH) .vllm-server.pid .dev-server.pid
	@rm -rf logs/* cache/* models/*
	@echo "$(GREEN)✅ Environment reset complete$(RESET)"
	@echo "Run 'make setup' to reinitialize"

deep-clean: ## Deep clean (including caches and downloads)
	@echo "$(YELLOW)⚠️  Deep cleaning...$(RESET)"
	@make reset
	@go clean -cache -testcache -modcache
	@rm -rf ~/.cache/huggingface 2>/dev/null || true
	@echo "$(GREEN)✅ Deep clean complete$(RESET)"

##@ 🎯 Workflow Shortcuts  
# Ultra-fast development workflows
new-dev: setup quick-dev ## Complete new developer setup + start dev server
morning: quick-check vllm-health quick-test ## Morning development health check
pre-push: fmt lint test ## Pre-push validation
full-test: test test-integration benchmark ## Comprehensive testing