# vLLM MCP Server Makefile

.PHONY: build run clean test deps dev install

# Go parameters
GOCMD=go
GOBUILD=$(GOCMD) build
GOCLEAN=$(GOCMD) clean
GOTEST=$(GOCMD) test
GOGET=$(GOCMD) get
GOMOD=$(GOCMD) mod

# Binary name
BINARY_NAME=mcp-vllm
BINARY_PATH=./$(BINARY_NAME)

# Default target
all: deps build

# Install dependencies
deps:
	$(GOMOD) tidy
	$(GOMOD) download

# Build the binary
build: deps
	$(GOBUILD) -v -o $(BINARY_PATH) .

# Build for production (static binary)
build-prod: deps
	CGO_ENABLED=0 GOOS=linux $(GOBUILD) -a -installsuffix cgo -ldflags '-extldflags "-static"' -o $(BINARY_PATH) .

# Build for multiple platforms
build-all: deps
	# Linux
	GOOS=linux GOARCH=amd64 $(GOBUILD) -o $(BINARY_NAME)-linux-amd64 .
	# macOS
	GOOS=darwin GOARCH=amd64 $(GOBUILD) -o $(BINARY_NAME)-darwin-amd64 .
	# Windows
	GOOS=windows GOARCH=amd64 $(GOBUILD) -o $(BINARY_NAME)-windows-amd64.exe .

# Run the server
run: build
	./$(BINARY_NAME)

# Development mode with auto-restart
dev:
	$(GOCMD) run . &
	echo $$! > .dev.pid

# Stop development server
dev-stop:
	@if [ -f .dev.pid ]; then \
		kill `cat .dev.pid` 2>/dev/null || true; \
		rm .dev.pid; \
	fi

# Run tests
test:
	$(GOTEST) -v ./...

# Run tests with coverage
test-coverage:
	$(GOTEST) -v -cover ./...

# Format code
fmt:
	gofumpt -w .
	$(GOCMD) mod tidy

# Lint code
lint:
	golangci-lint run

# Clean build artifacts
clean:
	$(GOCLEAN)
	rm -f $(BINARY_NAME)
	rm -f $(BINARY_NAME)-*
	rm -f .dev.pid

# Install the binary to GOPATH/bin
install: build
	cp $(BINARY_PATH) $(GOPATH)/bin/

# Quick development cycle
quick: clean fmt build run

# Docker build (if Dockerfile exists)
docker-build:
	docker build -t vllm-mcp-server .

# Help
help:
	@echo "Available targets:"
	@echo "  deps         - Download dependencies"
	@echo "  build        - Build the binary"
	@echo "  build-prod   - Build static production binary"
	@echo "  build-all    - Build for multiple platforms"
	@echo "  run          - Build and run the server"
	@echo "  dev          - Run in development mode"
	@echo "  dev-stop     - Stop development server"
	@echo "  test         - Run tests"
	@echo "  test-coverage- Run tests with coverage"
	@echo "  fmt          - Format code"
	@echo "  lint         - Lint code"
	@echo "  clean        - Clean build artifacts"
	@echo "  install      - Install binary to GOPATH/bin"
	@echo "  quick        - Quick development cycle"
	@echo "  help         - Show this help"

# Environment variables for configuration
.EXPORT_ALL_VARIABLES:
VLLM_URL ?= http://localhost:8000
MCP_PORT ?= 8080
GO111MODULE = on