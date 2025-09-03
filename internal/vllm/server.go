package vllm

import (
	"bufio"
	"context"
	"fmt"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"sync"
	"time"
)

// ServerManager manages the vLLM server lifecycle
type ServerManager struct {
	mu            sync.Mutex
	currentModel  string
	serverProcess *exec.Cmd
	cancelFunc    context.CancelFunc
	venvPath      string
	port          int
	isRunning     bool
}

// NewServerManager creates a new vLLM server manager
func NewServerManager(venvPath string, port int) *ServerManager {
	if venvPath == "" {
		homeDir, _ := os.UserHomeDir()
		venvPath = filepath.Join(homeDir, "venvs", "vllm_uv")
	}
	if port == 0 {
		port = 8000
	}
	return &ServerManager{
		venvPath: venvPath,
		port:     port,
	}
}

// RestartWithModel restarts the vLLM server with a new model
func (sm *ServerManager) RestartWithModel(modelID string) error {
	sm.mu.Lock()
	defer sm.mu.Unlock()

	// Check if already running the requested model
	if sm.currentModel == modelID && sm.isRunning {
		return nil // Already running with this model
	}

	// Stop current server if running
	if err := sm.stopServer(); err != nil {
		return fmt.Errorf("failed to stop current server: %w", err)
	}

	// Wait a moment for port to be released
	time.Sleep(2 * time.Second)

	// Start new server with the model
	if err := sm.startServer(modelID); err != nil {
		return fmt.Errorf("failed to start server with model %s: %w", modelID, err)
	}

	sm.currentModel = modelID
	return nil
}

// stopServer stops the currently running vLLM server
func (sm *ServerManager) stopServer() error {
	if sm.serverProcess != nil && sm.cancelFunc != nil {
		sm.cancelFunc()
		
		// Give the process a moment to terminate gracefully
		done := make(chan error, 1)
		go func() {
			done <- sm.serverProcess.Wait()
		}()
		
		select {
		case <-done:
			// Process terminated gracefully
		case <-time.After(5 * time.Second):
			// Force kill if it doesn't stop
			if sm.serverProcess.Process != nil {
				sm.serverProcess.Process.Kill()
			}
		}
		
		sm.serverProcess = nil
		sm.cancelFunc = nil
		sm.isRunning = false
	}
	return nil
}

// startServer starts a new vLLM server with the specified model
func (sm *ServerManager) startServer(modelID string) error {
	ctx, cancel := context.WithCancel(context.Background())
	sm.cancelFunc = cancel

	// Build the command
	vllmCmd := filepath.Join(sm.venvPath, "bin", "vllm")
	
	cmd := exec.CommandContext(ctx, vllmCmd, "serve", modelID,
		"--host", "0.0.0.0",
		"--port", fmt.Sprintf("%d", sm.port),
		"--gpu-memory-utilization", "0.85",
		"--dtype", "auto",
		"--enforce-eager",
		"--trust-remote-code",
	)

	// Set environment variables for GPU optimization
	cmd.Env = append(os.Environ(),
		"VLLM_USE_FLASH_ATTN=0",
		"VLLM_ATTENTION_BACKEND=XFORMERS",
		"VLLM_GPU_MEMORY_UTILIZATION=0.85",
		"TOKENIZERS_PARALLELISM=true",
		"PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True",
	)

	// Create pipes for stdout and stderr
	stdout, err := cmd.StdoutPipe()
	if err != nil {
		return fmt.Errorf("failed to create stdout pipe: %w", err)
	}
	stderr, err := cmd.StderrPipe()
	if err != nil {
		return fmt.Errorf("failed to create stderr pipe: %w", err)
	}

	// Start the command
	if err := cmd.Start(); err != nil {
		return fmt.Errorf("failed to start vLLM server: %w", err)
	}

	sm.serverProcess = cmd

	// Monitor server output to detect when it's ready
	readyChan := make(chan bool, 1)
	errorChan := make(chan error, 1)

	// Monitor stdout for ready signal
	go func() {
		scanner := bufio.NewScanner(stdout)
		for scanner.Scan() {
			line := scanner.Text()
			// Check for server ready indicators
			if strings.Contains(line, "Uvicorn running on") || 
			   strings.Contains(line, "Started server process") ||
			   strings.Contains(line, "Application startup complete") {
				select {
				case readyChan <- true:
				default:
				}
			}
		}
	}()

	// Monitor stderr for errors
	go func() {
		scanner := bufio.NewScanner(stderr)
		for scanner.Scan() {
			line := scanner.Text()
			// Check for critical errors
			if strings.Contains(line, "CUDA out of memory") ||
			   strings.Contains(line, "Failed to allocate") {
				select {
				case errorChan <- fmt.Errorf("vLLM server error: %s", line):
				default:
				}
			}
		}
	}()

	// Wait for server to be ready or timeout
	select {
	case <-readyChan:
		// Perform a health check before declaring success
		if err := sm.healthCheck(); err != nil {
			sm.stopServer()
			return fmt.Errorf("health check failed: %w", err)
		}
		sm.isRunning = true
		return nil
	case err := <-errorChan:
		sm.stopServer()
		return err
	case <-time.After(60 * time.Second):
		sm.stopServer()
		return fmt.Errorf("timeout waiting for vLLM server to start")
	}
}

// IsRunning returns whether the server is currently running
func (sm *ServerManager) IsRunning() bool {
	sm.mu.Lock()
	defer sm.mu.Unlock()
	return sm.isRunning
}

// GetCurrentModel returns the currently loaded model
func (sm *ServerManager) GetCurrentModel() string {
	sm.mu.Lock()
	defer sm.mu.Unlock()
	return sm.currentModel
}

// healthCheck verifies the server is responding
func (sm *ServerManager) healthCheck() error {
	client := &http.Client{Timeout: 5 * time.Second}
	resp, err := client.Get(fmt.Sprintf("http://localhost:%d/health", sm.port))
	if err != nil {
		// Try the models endpoint as fallback
		resp, err = client.Get(fmt.Sprintf("http://localhost:%d/v1/models", sm.port))
		if err != nil {
			return fmt.Errorf("server not responding: %w", err)
		}
	}
	defer resp.Body.Close()
	
	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("server returned status %d", resp.StatusCode)
	}
	return nil
}

// Shutdown gracefully shuts down the server
func (sm *ServerManager) Shutdown() error {
	sm.mu.Lock()
	defer sm.mu.Unlock()
	return sm.stopServer()
}