#!/usr/bin/env bash
# Automated documentation generation for Crush vLLM development
set -euo pipefail

# Colors
RED='\033[31m'
GREEN='\033[32m'
YELLOW='\033[33m'
BLUE='\033[34m'
CYAN='\033[36m'
RESET='\033[0m'

log() { echo -e "${GREEN}[DOC-GEN]${RESET} $1"; }
warn() { echo -e "${YELLOW}[WARN]${RESET} $1"; }
error() { echo -e "${RED}[ERROR]${RESET} $1"; exit 1; }
info() { echo -e "${BLUE}[INFO]${RESET} $1"; }

# Configuration
DOCS_DIR="docs"
OUTPUT_FILE="${DOCS_DIR}/DEVELOPER_GUIDE.md"

# Create docs directory
mkdir -p "${DOCS_DIR}"

log "Generating comprehensive developer documentation..."

# Generate main developer guide
cat > "${OUTPUT_FILE}" << 'EOF'
# Crush vLLM Development Guide

> **Ultra-optimized native development workflow for Crush with vLLM integration**

## 🚀 Quick Start (< 2 minutes)

### One-Command Setup
```bash
# Complete setup + start development server
make quick-start
```

### If Already Setup
```bash  
# Fast development server with hot reload
make quick-dev

# Quick health check
make quick-check

# Essential tests (< 30 seconds)
make quick-test
```

## 📋 Prerequisites

- **Go 1.24+** - Application development
- **Python 3.8+** - vLLM inference server
- **curl** - API testing
- **jq** - JSON processing (recommended)
- **NVIDIA GPU** - Optional, for GPU acceleration

### Quick Verification
```bash
make quick-check
```

## 🛠 Development Commands

### Essential Daily Commands
```bash
make morning          # Morning health check
make quick-dev         # Start development server  
make quick-test        # Fast essential tests
make pre-push          # Pre-push validation
```

### Build & Test
```bash
make build            # Optimized production build
make build-dev        # Development build with debugging
make build-fast       # Fast build (no optimization)

make test             # Unit tests with race detection
make test-integration # Integration tests with vLLM
make test-all         # All tests (unit + integration)
make test-coverage    # Coverage report
```

### vLLM Operations  
```bash
make vllm-setup       # Setup Python environment
make vllm-start       # Start lightweight vLLM server
make vllm-health      # Health check
make vllm-test        # Test with sample request
make vllm-logs        # Follow server logs
make vllm-stop        # Stop server
```

### Monitoring & Profiling
```bash
make monitor          # Interactive system monitor
make monitor-continuous # Continuous monitoring + logging
make profile          # Capture 30s profiling data
make profile-cpu      # Interactive CPU profiling
make profile-heap     # Interactive memory profiling
```

### Code Quality
```bash
make fmt              # Format code
make lint             # Fast linting
make lint-fix         # Auto-fix lint issues
make security-scan    # Security vulnerability scan
make ci               # Full CI pipeline locally
make ci-quick         # Essential CI checks (fast)
```

## 📊 Development Workflow

### New Developer Setup
1. **Quick Setup**: `make quick-start`
2. **Verify Setup**: `make status`  
3. **Start Coding**: Edit files, hot reload handles the rest
4. **Test Changes**: `make quick-test`
5. **Before Pushing**: `make pre-push`

### Daily Development Loop
1. **Morning Check**: `make morning`
2. **Start Development**: `make quick-dev`
3. **Code & Test**: Files auto-reload, run `make quick-test` as needed
4. **Performance Check**: `make monitor` (separate terminal)
5. **Before Commits**: `make pre-push`

### Performance Analysis
1. **Start with Profiling**: `CRUSH_PROFILE=true make quick-dev`
2. **Monitor Resources**: `make monitor` (separate terminal)  
3. **CPU Analysis**: `make profile-cpu`
4. **Memory Analysis**: `make profile-heap`
5. **Integration Testing**: `make test-integration`

## 🔧 Configuration

### Environment Variables
```bash
# Core development
CRUSH_DEBUG=true              # Enable debug logging
CRUSH_PROFILE=true           # Enable profiling endpoints
VLLM_PORT=8000               # vLLM server port
PROFILE_PORT=6060            # Profiling port

# Performance tuning
GPU_MEMORY_UTILIZATION=0.3   # GPU memory usage (0.1-0.9)
GOMAXPROCS=$(nproc)          # Go max processes
GOGC=100                     # Go GC percentage
```

### Configuration Files
- **~/.config/crush/crush.json** - Global Crush configuration
- **.env.development** - Local environment variables
- **.envrc** - Directory environment (direnv)

## 📁 Project Structure

```
crush/
├── cmd/                     # CLI commands
├── internal/                # Private packages
│   ├── app/                # Application logic
│   ├── vllm/               # vLLM integration
│   └── ...
├── scripts/                # Development scripts
│   ├── dev/               # Development helpers
│   ├── quick-setup.sh     # Ultra-fast setup
│   ├── test-integration-native.sh  # Integration tests
│   └── monitor-system.sh  # System monitoring
├── logs/                   # Application logs
├── cache/                  # Model cache
├── venv/                   # Python virtual environment
├── coverage/               # Test coverage reports
├── benchmarks/             # Performance benchmarks
└── Makefile               # Development commands
```

## 🐛 Debugging

### Common Issues

#### vLLM Server Won't Start
```bash
# Check logs
make vllm-logs

# Check Python environment
source venv/bin/activate && python -c "import vllm"

# Reset and retry
make vllm-stop && make vllm-start
```

#### Build Failures
```bash
# Clean and rebuild
make clean && make build

# Check dependencies
go mod verify && go mod tidy
```

#### Performance Issues
```bash
# Check system resources
make perf-info

# Start monitoring
make monitor

# Profile the application
make profile-cpu
```

#### Integration Test Failures
```bash
# Quick integration test
make test-integration-quick

# Check vLLM health
make vllm-health

# Check logs
tail -f logs/integration-vllm.log
```

### Debugging Tools

#### Built-in Profiling
```bash
# Start with profiling enabled
CRUSH_PROFILE=true make quick-dev

# Access profiling endpoints
curl http://localhost:6060/debug/pprof/
```

#### System Monitoring
```bash
# Interactive monitor
make monitor

# Continuous monitoring with logging  
make monitor-continuous

# Performance snapshot
make perf-info
```

## 🧪 Testing

### Test Categories
- **Unit Tests**: `make test` - Fast, isolated tests
- **Integration Tests**: `make test-integration` - Full vLLM integration
- **Quick Tests**: `make quick-test` - Essential tests only
- **Benchmark Tests**: `make benchmark` - Performance benchmarks

### CI/CD Testing
```bash
# Local CI simulation
make ci

# Essential checks only (fast)
make ci-quick

# Full test suite
make full-test
```

### Manual Testing
```bash
# Test vLLM directly
make vllm-test

# Integration testing
make test-integration

# Performance testing
make benchmark
```

## 📈 Performance Optimization

### Build Optimization
```bash
# Production build
make build

# Development build (debugging)
make build-dev

# Fast build (development speed)
make build-fast
```

### Runtime Optimization
- **GPU Memory**: Adjust `GPU_MEMORY_UTILIZATION` (0.1-0.9)
- **Go Runtime**: Tune `GOMAXPROCS` and `GOGC`
- **Model Selection**: Use lightweight models for development

### Monitoring Performance
```bash
# Real-time monitoring
make monitor

# Profile capture
make profile

# System info
make perf-info
```

## 🔒 Security

### Security Scanning
```bash
make security-scan        # Vulnerability scanning
```

### Best Practices
- Never commit API keys or secrets
- Use environment variables for configuration
- Regularly update dependencies: `make update-deps`
- Run security scans before releases

## 🚀 Deployment

### Build for Production
```bash
make build               # Optimized binary
make schema              # Generate configuration schema  
make docs                # Update documentation
```

### CI/CD Pipeline
The project includes GitHub Actions workflow (`.github/workflows/ci-native.yml`) with:
- Multi-platform builds
- Comprehensive testing
- Security scanning
- Performance benchmarks

## 📚 Additional Resources

### Useful Commands
```bash
make status              # Development environment status
make version             # Version information
make deps                # Dependency information
make help                # All available commands
```

### Maintenance
```bash
make update-deps         # Update dependencies
make reset               # Reset development environment  
make deep-clean          # Deep clean (including caches)
```

### Troubleshooting
- **Logs**: `tail -f logs/*.log`
- **vLLM Issues**: Check `logs/vllm.log`
- **Integration Issues**: Check `logs/integration-vllm.log`
- **System Issues**: Run `make perf-info`

---

## 💡 Pro Tips

1. **Use `make morning`** at the start of each day for health check
2. **Keep `make monitor` running** in a separate terminal during development
3. **Use `make quick-test`** frequently for rapid feedback
4. **Run `make pre-push`** before committing changes
5. **Use GPU acceleration** when available for faster inference
6. **Monitor resource usage** to avoid OOM issues
7. **Keep models lightweight** during development (use `facebook/opt-125m`)

---

*Generated automatically by Crush development tooling*
EOF

log "✅ Generated developer guide: ${OUTPUT_FILE}"

# Generate API documentation from comments
if [[ -x "./crush" ]]; then
    log "Generating API schema documentation..."
    ./crush schema > "${DOCS_DIR}/api-schema.json" 2>/dev/null || warn "Could not generate API schema"
fi

# Generate command reference
log "Generating command reference..."
cat > "${DOCS_DIR}/COMMANDS.md" << 'EOF'
# Crush Development Commands Reference

## Quick Commands (Most Used)

| Command | Description | Usage |
|---------|-------------|-------|
| `make quick-start` | Complete setup + dev server | New developers |
| `make quick-dev` | Start development server | Daily development |
| `make quick-test` | Essential tests (< 30s) | Rapid feedback |  
| `make quick-check` | Environment health check | Verification |

## Build Commands

| Command | Description | Output |
|---------|-------------|--------|
| `make build` | Optimized production build | `./crush` |
| `make build-dev` | Development build with race detection | `./crush-dev` |
| `make build-fast` | Fast build (no optimization) | `./crush` |

## Test Commands

| Command | Description | Duration |
|---------|-------------|----------|
| `make test` | Unit tests with race detection | ~30s |
| `make test-integration` | Full integration tests | ~2-5min |
| `make test-all` | All tests (unit + integration) | ~3-6min |
| `make test-coverage` | Coverage report | ~1min |

## vLLM Commands

| Command | Description | Notes |
|---------|-------------|-------|
| `make vllm-setup` | Setup Python environment | One-time setup |
| `make vllm-start` | Start vLLM server | Uses lightweight model |
| `make vllm-health` | Health check | Returns 0/1 exit code |
| `make vllm-test` | Test with sample request | Requires running server |
| `make vllm-stop` | Stop vLLM server | Cleanup |

## Monitoring Commands

| Command | Description | Interface |
|---------|-------------|-----------|
| `make monitor` | Interactive system monitor | Terminal UI |
| `make monitor-continuous` | Continuous monitoring + logs | Background |
| `make profile-cpu` | CPU profiling | Web UI (:6061) |
| `make profile-heap` | Memory profiling | Web UI (:6061) |

## Workflow Commands

| Command | Description | When to Use |
|---------|-------------|-------------|
| `make morning` | Morning health check | Start of day |
| `make pre-push` | Pre-push validation | Before git push |
| `make ci` | Full CI pipeline locally | Before major commits |
| `make new-dev` | New developer setup | Fresh environment |

## Maintenance Commands

| Command | Description | Impact |
|---------|-------------|--------|
| `make clean` | Clean build artifacts | Safe |
| `make reset` | Reset dev environment | Removes venv, logs |
| `make deep-clean` | Deep clean + caches | Removes downloads |
| `make update-deps` | Update dependencies | Updates go.mod |

---

*Run `make help` for the complete list with descriptions*
EOF

# Generate troubleshooting guide
log "Generating troubleshooting guide..."
cat > "${DOCS_DIR}/TROUBLESHOOTING.md" << 'EOF'
# Troubleshooting Guide

## Common Issues & Solutions

### Setup Issues

#### "Go not installed"
```bash
# Install Go 1.24+
curl -LO https://go.dev/dl/go1.24.linux-amd64.tar.gz
sudo tar -C /usr/local -xzf go1.24.linux-amd64.tar.gz
export PATH=$PATH:/usr/local/go/bin
```

#### "Python not found"  
```bash
# Ubuntu/Debian
sudo apt update && sudo apt install python3 python3-venv python3-pip

# RHEL/CentOS/Fedora  
sudo dnf install python3 python3-pip
```

#### "vLLM installation failed"
```bash
# Check Python version (need 3.8+)
python3 --version

# Try CPU-only installation
source venv/bin/activate
pip install vllm torch --cpu-only
```

### Development Server Issues

#### "Port already in use"
```bash
# Find and kill process using port 8000
sudo lsof -ti:8000 | xargs kill -9

# Or use different port
VLLM_PORT=8001 make vllm-start
```

#### "vLLM server won't start"
```bash
# Check logs
make vllm-logs

# Common fixes
make vllm-stop
make clean
make vllm-setup
make vllm-start
```

#### "Hot reload not working"
```bash
# Install file watcher (Ubuntu/Debian)
sudo apt install inotify-tools

# Or use manual restart
# Ctrl+C to stop, then `make quick-dev` again
```

### Build Issues

#### "Build failed"
```bash
# Clean and retry
make clean
go mod tidy
make build
```

#### "Race condition detected"
```bash
# This is good - race detector found an issue
# Fix the race condition in the code
# Or use non-race build for testing
make build-fast
```

### Test Issues

#### "Tests timeout"
```bash
# Run with longer timeout
go test -timeout=300s ./...

# Or run specific package
go test -v ./internal/app
```

#### "Integration tests fail"
```bash
# Check vLLM is running
make vllm-health

# Run quick integration test
make test-integration-quick

# Check integration logs
tail -f logs/integration-vllm.log
```

### Performance Issues

#### "High memory usage"
```bash
# Check memory usage
make perf-info

# Reduce GPU memory usage
GPU_MEMORY_UTILIZATION=0.1 make vllm-start

# Use CPU-only mode
# Edit venv/lib/python*/site-packages/vllm config
```

#### "Slow performance"
```bash
# Enable profiling
CRUSH_PROFILE=true make quick-dev

# Profile CPU
make profile-cpu

# Check system resources
make monitor
```

#### "GPU not detected"
```bash
# Check NVIDIA drivers
nvidia-smi

# Check CUDA
nvcc --version

# Use CPU-only mode if no GPU
pip install vllm torch --cpu-only
```

### Network Issues

#### "Connection refused"
```bash
# Check if server is running
make vllm-health

# Check port binding
netstat -tulpn | grep :8000

# Check firewall
sudo ufw status
```

#### "DNS resolution issues"
```bash
# Use localhost instead of domain names
curl http://localhost:8000/health

# Check /etc/hosts
grep localhost /etc/hosts
```

## Debugging Techniques

### Enable Debug Logging
```bash
export CRUSH_DEBUG=true
make quick-dev
```

### Capture Profiling Data
```bash
# Start with profiling
CRUSH_PROFILE=true make quick-dev

# In another terminal
make profile

# Analyze with pprof
go tool pprof profiling/*/cpu.prof
```

### Monitor System Resources
```bash
# Interactive monitoring
make monitor

# Check specific process
ps aux | grep -E "(crush|vllm)"
```

### Check Logs
```bash
# Application logs
tail -f logs/crush.log

# vLLM logs  
tail -f logs/vllm.log

# Integration test logs
tail -f logs/integration-vllm.log

# System monitoring logs
tail -f logs/system-monitor.log
```

## Getting Help

### Log Analysis
1. **Check recent logs**: `tail -50 logs/*.log`
2. **Search for errors**: `grep -i error logs/*.log`
3. **Check timestamps**: Look for patterns around failure time

### System Information
```bash
# Environment status
make status

# Dependencies
make deps

# Version info
make version
```

### Debugging Commands
```bash
# Health check
make quick-check

# System info
make perf-info

# Test specific component
make vllm-test
make test-integration-quick
```

### Reset Options
```bash
# Soft reset (keep downloads)
make clean

# Hard reset (remove venv)
make reset

# Nuclear reset (remove all caches)
make deep-clean
```

---

**Still having issues?**
1. Check the logs: `tail -f logs/*.log`
2. Run health check: `make morning`
3. Try reset: `make reset && make setup`
4. Check system resources: `make monitor`
EOF

# Generate final summary
log "Documentation generation complete!"

echo ""
echo -e "${CYAN}📚 Generated Documentation:${RESET}"
echo "  • ${OUTPUT_FILE} - Comprehensive developer guide"
echo "  • ${DOCS_DIR}/COMMANDS.md - Command reference"
echo "  • ${DOCS_DIR}/TROUBLESHOOTING.md - Troubleshooting guide"
if [[ -f "${DOCS_DIR}/api-schema.json" ]]; then
    echo "  • ${DOCS_DIR}/api-schema.json - API schema"
fi

echo ""
echo -e "${GREEN}📖 Quick Links:${RESET}"
echo "  • View guide: cat ${OUTPUT_FILE}"
echo "  • All commands: make help"
echo "  • Get started: make quick-start"

log "✅ Ready for development!"