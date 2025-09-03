# Crush vLLM Makefile
.PHONY: help build run dev test clean setup quick-start

# Default target
help:
	@echo "Crush vLLM Build System"
	@echo ""
	@echo "Quick commands:"
	@echo "  make quick-start  - Complete setup and start dev server"
	@echo "  make build       - Build Crush binary"
	@echo "  make run         - Run Crush"
	@echo "  make dev         - Development mode with vLLM"
	@echo "  make test        - Run tests"
	@echo "  make clean       - Clean build artifacts"
	@echo "  make setup       - Initial environment setup"
	@echo ""
	@echo "vLLM commands:"
	@echo "  make vllm-start  - Start vLLM server"
	@echo "  make vllm-stop   - Stop vLLM server"
	@echo "  make vllm-status - Check vLLM server status"

# Build the Crush binary
build:
	@echo "Building Crush..."
	@go build -o crush .
	@echo "✓ Build complete"

# Run Crush
run: build
	@./crush

# Development mode - start vLLM and Crush
dev: vllm-start
	@echo "Starting Crush in development mode..."
	@./crush

# Run tests
test:
	@echo "Running tests..."
	@go test ./... -v

# Clean build artifacts
clean:
	@echo "Cleaning build artifacts..."
	@rm -f crush
	@go clean -cache
	@echo "✓ Clean complete"

# Initial setup
setup:
	@echo "Setting up development environment..."
	@if [ -f scripts/quick-setup.sh ]; then \
		chmod +x scripts/quick-setup.sh && ./scripts/quick-setup.sh; \
	else \
		echo "Setting up vLLM environment..."; \
		if [ ! -d ~/venvs/vllm_uv ]; then \
			uv venv ~/venvs/vllm_uv --python 3.12; \
		fi; \
		. ~/venvs/vllm_uv/bin/activate && uv pip install vllm; \
	fi
	@echo "✓ Setup complete"

# Quick start - setup and run
quick-start: setup build dev

# vLLM server management
vllm-start:
	@echo "Starting vLLM server..."
	@if pgrep -f "vllm.entrypoints.openai" > /dev/null; then \
		echo "vLLM server already running"; \
	else \
		. ~/venvs/vllm_uv/bin/activate && \
		VLLM_USE_FLASH_ATTN=0 VLLM_ATTENTION_BACKEND=XFORMERS \
		vllm serve facebook/opt-125m \
			--host 0.0.0.0 \
			--port 8000 \
			--gpu-memory-utilization 0.85 \
			--dtype auto \
			--enforce-eager \
			--trust-remote-code > /tmp/vllm.log 2>&1 & \
		echo "vLLM server started (PID: $$!)"; \
		sleep 3; \
	fi

vllm-stop:
	@echo "Stopping vLLM server..."
	@pkill -f vllm.entrypoints.openai || echo "No vLLM server running"
	@echo "✓ vLLM server stopped"

vllm-status:
	@if pgrep -f "vllm.entrypoints.openai" > /dev/null; then \
		echo "✓ vLLM server is running"; \
		echo "Logs: tail -f /tmp/vllm.log"; \
	else \
		echo "✗ vLLM server is not running"; \
	fi