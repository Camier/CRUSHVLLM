//go:build ignore
// +build ignore

package provider

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"
	"sync"
	"time"

	"github.com/charmbracelet/crush/internal/config"
	"github.com/charmbracelet/crush/internal/llm/tools"
	"github.com/charmbracelet/crush/internal/log"
	"github.com/charmbracelet/crush/internal/message"
	"github.com/charmbracelet/x/ansi"
)

// VLLMProvider implements the Provider interface for vLLM local inference.
type VLLMProvider struct {
	baseURL         string
	apiKey          string // Optional, for secured vLLM deployments
	httpClient      *http.Client
	config          *VLLMConfig
	mu              sync.RWMutex
	metricsCollector *MetricsCollector
	// Legacy metrics for backward compatibility
	metrics         *VLLMMetrics
	// Background context for metrics collection
	backgroundCtx    context.Context
	cancelBackground context.CancelFunc
}

// VLLMConfig holds vLLM-specific configuration.
type VLLMConfig struct {
	// Performance settings
	GPUMemoryUtilization float32 `json:"gpu_memory_utilization,omitempty"`
	MaxModelLen          int     `json:"max_model_len,omitempty"`
	MaxNumSeqs           int     `json:"max_num_seqs,omitempty"`
	TensorParallelSize   int     `json:"tensor_parallel_size,omitempty"`
	
	// Quantization settings
	Quantization string `json:"quantization,omitempty"` // "awq", "gptq", "squeezellm"
	
	// Request settings
	Timeout           time.Duration `json:"timeout,omitempty"`
	MaxRetries        int           `json:"max_retries,omitempty"`
	EnableStreaming   bool          `json:"enable_streaming,omitempty"`
	EnableCaching     bool          `json:"enable_caching,omitempty"`
	
	// Model-specific settings
	ModelPath         string `json:"model_path,omitempty"`
	TokenizerPath     string `json:"tokenizer_path,omitempty"`
	TrustRemoteCode   bool   `json:"trust_remote_code,omitempty"`
	
	// Monitoring settings
	EnableMonitoring  bool          `json:"enable_monitoring,omitempty"`
	MetricsConfig     *MetricsConfig `json:"metrics_config,omitempty"`
}

// VLLMMetrics tracks performance metrics.
type VLLMMetrics struct {
	mu               sync.RWMutex
	RequestCount     int64
	TotalLatency     time.Duration
	TokensGenerated  int64
	TokensPrompt     int64
	CacheHits        int64
	CacheMisses      int64
	LastRequestTime  time.Time
}

// VLLMRequest represents a request to the vLLM API.
type VLLMRequest struct {
	Model            string                 `json:"model"`
	Messages         []map[string]any       `json:"messages,omitempty"`
	Prompt           string                 `json:"prompt,omitempty"`
	Temperature      float32                `json:"temperature,omitempty"`
	TopP             float32                `json:"top_p,omitempty"`
	MaxTokens        int                    `json:"max_tokens,omitempty"`
	Stream           bool                   `json:"stream,omitempty"`
	Stop             []string               `json:"stop,omitempty"`
	PresencePenalty  float32                `json:"presence_penalty,omitempty"`
	FrequencyPenalty float32                `json:"frequency_penalty,omitempty"`
	N                int                    `json:"n,omitempty"`
	BestOf           int                    `json:"best_of,omitempty"`
	UseBeamSearch    bool                   `json:"use_beam_search,omitempty"`
	Tools            []tools.Tool           `json:"tools,omitempty"`
	ToolChoice       any                    `json:"tool_choice,omitempty"`
	
	// vLLM-specific parameters
	IgnoreEOS        bool                   `json:"ignore_eos,omitempty"`
	SkipSpecialTokens bool                  `json:"skip_special_tokens,omitempty"`
	Spaces           map[string]int         `json:"spaces_between_special_tokens,omitempty"`
}

// VLLMResponse represents a response from the vLLM API.
type VLLMResponse struct {
	ID      string `json:"id"`
	Object  string `json:"object"`
	Created int64  `json:"created"`
	Model   string `json:"model"`
	Choices []struct {
		Index        int    `json:"index"`
		Message      any    `json:"message,omitempty"`
		Text         string `json:"text,omitempty"`
		FinishReason string `json:"finish_reason"`
		Logprobs     any    `json:"logprobs,omitempty"`
	} `json:"choices"`
	Usage struct {
		PromptTokens     int `json:"prompt_tokens"`
		CompletionTokens int `json:"completion_tokens"`
		TotalTokens      int `json:"total_tokens"`
	} `json:"usage"`
}

// NewVLLMProvider creates a new vLLM provider instance.
func NewVLLMProvider(cfg config.Provider) (*VLLMProvider, error) {
	baseURL := cfg.BaseURL
	if baseURL == "" {
		baseURL = "http://localhost:8000/v1" // Default vLLM server URL
	}
	
	// Parse vLLM-specific configuration
	vllmConfig := &VLLMConfig{
		GPUMemoryUtilization: 0.9,
		MaxModelLen:          8192,
		MaxNumSeqs:           256,
		TensorParallelSize:   1,
		Timeout:              5 * time.Minute,
		MaxRetries:           3,
		EnableStreaming:      true,
		EnableCaching:        true,
		EnableMonitoring:     true, // Enable monitoring by default
	}
	
	// Override with user configuration if provided
	if cfg.ExtraConfig != nil {
		if gpu, ok := cfg.ExtraConfig["gpu_memory_utilization"].(float64); ok {
			vllmConfig.GPUMemoryUtilization = float32(gpu)
		}
		if maxLen, ok := cfg.ExtraConfig["max_model_len"].(float64); ok {
			vllmConfig.MaxModelLen = int(maxLen)
		}
		if quant, ok := cfg.ExtraConfig["quantization"].(string); ok {
			vllmConfig.Quantization = quant
		}
		if modelPath, ok := cfg.ExtraConfig["model_path"].(string); ok {
			vllmConfig.ModelPath = modelPath
		}
		if enableMonitoring, ok := cfg.ExtraConfig["enable_monitoring"].(bool); ok {
			vllmConfig.EnableMonitoring = enableMonitoring
		}
	}
	
	// Create background context for metrics collection
	backgroundCtx, cancelBackground := context.WithCancel(context.Background())
	
	// Initialize metrics collector if monitoring is enabled
	var metricsCollector *MetricsCollector
	if vllmConfig.EnableMonitoring {
		metricsConfig := vllmConfig.MetricsConfig
		if metricsConfig == nil {
			// Use default metrics configuration
			metricsConfig = &MetricsConfig{
				GPUMetricsInterval:    5 * time.Second,
				SystemMetricsInterval: 10 * time.Second,
				MaxLatencyThreshold:   30 * time.Second,
				MaxGPUUtilization:     0.95,
				MaxMemoryUsage:        16 * 1024 * 1024 * 1024, // 16GB
				MinCacheHitRate:       0.80,
				EnablePrometheus:      true,
				PrometheusPort:        9090,
				PrometheusPath:        "/metrics",
				EnableLogging:         true,
				LogInterval:           60 * time.Second,
				EnableAlerts:          true,
			}
			vllmConfig.MetricsConfig = metricsConfig
		}
		
		metricsCollector = NewMetricsCollector(metricsConfig)
		
		// Start background metrics collection
		metricsCollector.StartBackgroundCollection(backgroundCtx)
		
		log.Info("vLLM monitoring enabled", 
			"prometheus_enabled", metricsConfig.EnablePrometheus,
			"gpu_metrics_interval", metricsConfig.GPUMetricsInterval,
			"log_interval", metricsConfig.LogInterval)
	}
	
	provider := &VLLMProvider{
		baseURL: baseURL,
		apiKey:  cfg.APIKey,
		httpClient: &http.Client{
			Timeout: vllmConfig.Timeout,
			Transport: &http.Transport{
				MaxIdleConns:        100,
				MaxIdleConnsPerHost: 10,
				IdleConnTimeout:     90 * time.Second,
			},
		},
		config:           vllmConfig,
		metrics:          &VLLMMetrics{}, // Keep legacy metrics for compatibility
		metricsCollector: metricsCollector,
		backgroundCtx:    backgroundCtx,
		cancelBackground: cancelBackground,
	}
	
	return provider, nil
}

// ID returns the provider ID.
func (p *VLLMProvider) ID() string {
	return "vllm"
}

// Name returns the provider name.
func (p *VLLMProvider) Name() string {
	return "vLLM (Local)"
}

// Send sends a request to the vLLM server.
func (p *VLLMProvider) Send(ctx context.Context, msgs []message.Message, opts ...RequestOption) (*Response, error) {
	req := p.buildRequest(msgs, opts...)
	
	// Start request tracking
	startTime := time.Now()
	if p.metricsCollector != nil {
		p.metricsCollector.StartRequest()
	}
	
	// Defer metrics recording and request completion
	var tokensPrompt, tokensGenerated int
	var cacheHit bool
	defer func() {
		latency := time.Since(startTime)
		
		// Record legacy metrics for backward compatibility
		p.recordMetrics(latency, req)
		
		// Record enhanced metrics
		if p.metricsCollector != nil {
			p.metricsCollector.EndRequest()
			p.metricsCollector.RecordRequest(latency, tokensPrompt, tokensGenerated, cacheHit, 1) // Single request batch size
		}
	}()
	
	// Create HTTP request
	body, err := json.Marshal(req)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal request: %w", err)
	}
	
	httpReq, err := http.NewRequestWithContext(ctx, "POST", p.baseURL+"/chat/completions", bytes.NewReader(body))
	if err != nil {
		return nil, fmt.Errorf("failed to create request: %w", err)
	}
	
	httpReq.Header.Set("Content-Type", "application/json")
	if p.apiKey != "" {
		httpReq.Header.Set("Authorization", "Bearer "+p.apiKey)
	}
	
	// Send request with retries
	var resp *http.Response
	for attempt := 0; attempt <= p.config.MaxRetries; attempt++ {
		resp, err = p.httpClient.Do(httpReq)
		if err == nil && resp.StatusCode < 500 {
			break
		}
		
		if attempt < p.config.MaxRetries {
			// Exponential backoff
			time.Sleep(time.Duration(1<<uint(attempt)) * time.Second)
			log.Debug("Retrying vLLM request", "attempt", attempt+1, "error", err)
		}
	}
	
	if err != nil {
		return nil, fmt.Errorf("failed to send request: %w", err)
	}
	defer resp.Body.Close()
	
	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(resp.Body)
		return nil, fmt.Errorf("vLLM API error (status %d): %s", resp.StatusCode, body)
	}
	
	// Parse response
	var vllmResp VLLMResponse
	if err := json.NewDecoder(resp.Body).Decode(&vllmResp); err != nil {
		return nil, fmt.Errorf("failed to decode response: %w", err)
	}
	
	// Extract token counts for metrics
	tokensPrompt = vllmResp.Usage.PromptTokens
	tokensGenerated = vllmResp.Usage.CompletionTokens
	
	// Determine cache hit (this would need to be implemented based on vLLM response headers or metadata)
	cacheHit = p.detectCacheHit(resp.Header)
	
	return p.convertResponse(&vllmResp), nil
}

// detectCacheHit attempts to detect if the response was served from cache
func (p *VLLMProvider) detectCacheHit(headers http.Header) bool {
	// Check for vLLM-specific cache headers
	if cacheStatus := headers.Get("X-vLLM-Cache-Hit"); cacheStatus == "true" {
		return true
	}
	
	// Check for standard cache headers
	if cacheStatus := headers.Get("X-Cache-Status"); cacheStatus == "HIT" {
		return true
	}
	
	// For now, assume cache miss if no headers indicate otherwise
	// This could be enhanced based on vLLM's actual cache implementation
	return false
}

// SendStream sends a streaming request to the vLLM server.
func (p *VLLMProvider) SendStream(ctx context.Context, msgs []message.Message, w io.Writer, opts ...RequestOption) (*Response, error) {
	req := p.buildRequest(msgs, opts...)
	req.Stream = true
	
	// Start request tracking
	startTime := time.Now()
	if p.metricsCollector != nil {
		p.metricsCollector.StartRequest()
	}
	
	// Defer metrics recording and request completion
	var tokensPrompt, tokensGenerated int
	var cacheHit bool
	defer func() {
		latency := time.Since(startTime)
		
		// Record legacy metrics for backward compatibility
		p.recordMetrics(latency, req)
		
		// Record enhanced metrics
		if p.metricsCollector != nil {
			p.metricsCollector.EndRequest()
			p.metricsCollector.RecordRequest(latency, tokensPrompt, tokensGenerated, cacheHit, 1) // Single request batch size
		}
	}()
	
	// Create HTTP request
	body, err := json.Marshal(req)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal request: %w", err)
	}
	
	httpReq, err := http.NewRequestWithContext(ctx, "POST", p.baseURL+"/chat/completions", bytes.NewReader(body))
	if err != nil {
		return nil, fmt.Errorf("failed to create request: %w", err)
	}
	
	httpReq.Header.Set("Content-Type", "application/json")
	httpReq.Header.Set("Accept", "text/event-stream")
	if p.apiKey != "" {
		httpReq.Header.Set("Authorization", "Bearer "+p.apiKey)
	}
	
	// Send request
	resp, err := p.httpClient.Do(httpReq)
	if err != nil {
		return nil, fmt.Errorf("failed to send request: %w", err)
	}
	defer resp.Body.Close()
	
	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(resp.Body)
		return nil, fmt.Errorf("vLLM API error (status %d): %s", resp.StatusCode, body)
	}
	
	// Process SSE stream
	scanner := bufio.NewScanner(resp.Body)
	var fullResponse strings.Builder
	var usage struct {
		PromptTokens     int
		CompletionTokens int
		TotalTokens      int
	}
	
	for scanner.Scan() {
		line := scanner.Text()
		
		// Skip empty lines and comments
		if line == "" || strings.HasPrefix(line, ":") {
			continue
		}
		
		// Parse SSE data
		if strings.HasPrefix(line, "data: ") {
			data := strings.TrimPrefix(line, "data: ")
			
			// Check for end of stream
			if data == "[DONE]" {
				break
			}
			
			// Parse JSON chunk
			var chunk VLLMResponse
			if err := json.Unmarshal([]byte(data), &chunk); err != nil {
				log.Debug("Failed to parse SSE chunk", "error", err, "data", data)
				continue
			}
			
			// Write chunk to output
			if len(chunk.Choices) > 0 && chunk.Choices[0].Message != nil {
				if msg, ok := chunk.Choices[0].Message.(map[string]any); ok {
					if content, ok := msg["content"].(string); ok {
						fmt.Fprint(w, content)
						fullResponse.WriteString(content)
					}
				} else if chunk.Choices[0].Text != "" {
					fmt.Fprint(w, chunk.Choices[0].Text)
					fullResponse.WriteString(chunk.Choices[0].Text)
				}
			}
			
			// Update usage stats
			if chunk.Usage.TotalTokens > 0 {
				usage = chunk.Usage
			}
		}
	}
	
	if err := scanner.Err(); err != nil {
		return nil, fmt.Errorf("error reading stream: %w", err)
	}
	
	// Extract token counts for metrics (set outside defer for access in defer)
	tokensPrompt = usage.PromptTokens
	tokensGenerated = usage.CompletionTokens
	
	// Determine cache hit from response headers
	cacheHit = p.detectCacheHit(resp.Header)
	
	// Update legacy metrics
	p.mu.Lock()
	p.metrics.TokensGenerated += int64(usage.CompletionTokens)
	p.metrics.TokensPrompt += int64(usage.PromptTokens)
	p.mu.Unlock()
	
	return &Response{
		Content: fullResponse.String(),
		Usage: &Usage{
			InputTokens:  usage.PromptTokens,
			OutputTokens: usage.CompletionTokens,
		},
	}, nil
}

// buildRequest builds a vLLM request from messages and options.
func (p *VLLMProvider) buildRequest(msgs []message.Message, opts ...RequestOption) *VLLMRequest {
	req := &VLLMRequest{
		Model:             "default", // vLLM uses the loaded model
		Temperature:       0.7,
		TopP:              0.9,
		MaxTokens:         2048,
		Stream:            false,
		SkipSpecialTokens: true,
		N:                 1,
	}
	
	// Apply options
	for _, opt := range opts {
		if ro, ok := opt.(*requestOptions); ok {
			if ro.temperature != nil {
				req.Temperature = *ro.temperature
			}
			if ro.topP != nil {
				req.TopP = *ro.topP
			}
			if ro.maxTokens != nil {
				req.MaxTokens = *ro.maxTokens
			}
			if ro.stop != nil {
				req.Stop = ro.stop
			}
			if ro.tools != nil {
				req.Tools = ro.tools
			}
			if ro.toolChoice != nil {
				req.ToolChoice = ro.toolChoice
			}
		}
	}
	
	// Convert messages
	req.Messages = make([]map[string]any, len(msgs))
	for i, msg := range msgs {
		m := map[string]any{
			"role":    msg.Role(),
			"content": msg.Content(),
		}
		
		// Add tool calls if present
		if toolCalls := msg.ToolCalls(); len(toolCalls) > 0 {
			m["tool_calls"] = toolCalls
		}
		
		// Add tool call ID for tool responses
		if toolCallID := msg.ToolCallID(); toolCallID != "" {
			m["tool_call_id"] = toolCallID
		}
		
		req.Messages[i] = m
	}
	
	return req
}

// convertResponse converts a vLLM response to the internal Response type.
func (p *VLLMProvider) convertResponse(vr *VLLMResponse) *Response {
	if len(vr.Choices) == 0 {
		return &Response{
			Content: "",
		}
	}
	
	choice := vr.Choices[0]
	resp := &Response{
		Usage: &Usage{
			InputTokens:  vr.Usage.PromptTokens,
			OutputTokens: vr.Usage.CompletionTokens,
		},
		FinishReason: choice.FinishReason,
	}
	
	// Extract content
	if choice.Message != nil {
		if msg, ok := choice.Message.(map[string]any); ok {
			if content, ok := msg["content"].(string); ok {
				resp.Content = content
			}
			
			// Extract tool calls if present
			if toolCalls, ok := msg["tool_calls"].([]any); ok {
				resp.ToolCalls = make([]ToolCall, len(toolCalls))
				for i, tc := range toolCalls {
					if tcMap, ok := tc.(map[string]any); ok {
						resp.ToolCalls[i] = ToolCall{
							ID:   tcMap["id"].(string),
							Name: tcMap["function"].(map[string]any)["name"].(string),
							Args: tcMap["function"].(map[string]any)["arguments"].(string),
						}
					}
				}
			}
		}
	} else if choice.Text != "" {
		resp.Content = choice.Text
	}
	
	return resp
}

// recordMetrics records performance metrics.
func (p *VLLMProvider) recordMetrics(latency time.Duration, req *VLLMRequest) {
	p.mu.Lock()
	defer p.mu.Unlock()
	
	p.metrics.RequestCount++
	p.metrics.TotalLatency += latency
	p.metrics.LastRequestTime = time.Now()
	
	// Log performance if enabled
	if log.IsDebug() {
		avgLatency := p.metrics.TotalLatency / time.Duration(p.metrics.RequestCount)
		log.Debug("vLLM performance",
			"requests", p.metrics.RequestCount,
			"avg_latency", avgLatency,
			"last_latency", latency,
			"tokens_generated", p.metrics.TokensGenerated,
			"cache_hit_rate", p.getCacheHitRate(),
		)
	}
}

// getCacheHitRate returns the cache hit rate as a percentage.
func (p *VLLMProvider) getCacheHitRate() float64 {
	total := p.metrics.CacheHits + p.metrics.CacheMisses
	if total == 0 {
		return 0
	}
	return float64(p.metrics.CacheHits) / float64(total) * 100
}

// GetMetrics returns current performance metrics.
func (p *VLLMProvider) GetMetrics() map[string]any {
	// If enhanced monitoring is enabled, use the comprehensive metrics collector
	if p.metricsCollector != nil {
		enhancedMetrics := p.metricsCollector.GetMetrics()
		
		// Also include legacy metrics for backward compatibility
		p.mu.RLock()
		legacyAvgLatency := time.Duration(0)
		if p.metrics.RequestCount > 0 {
			legacyAvgLatency = p.metrics.TotalLatency / time.Duration(p.metrics.RequestCount)
		}
		
		legacyMetrics := map[string]any{
			"legacy_request_count":    p.metrics.RequestCount,
			"legacy_avg_latency_ms":   legacyAvgLatency.Milliseconds(),
			"legacy_tokens_generated": p.metrics.TokensGenerated,
			"legacy_tokens_prompt":    p.metrics.TokensPrompt,
			"legacy_cache_hit_rate":   p.getCacheHitRate(),
			"legacy_last_request":     p.metrics.LastRequestTime.Format(time.RFC3339),
		}
		p.mu.RUnlock()
		
		// Merge enhanced and legacy metrics
		combined := make(map[string]any)
		for k, v := range enhancedMetrics {
			combined[k] = v
		}
		for k, v := range legacyMetrics {
			combined[k] = v
		}
		
		return combined
	}
	
	// Fallback to legacy metrics only
	p.mu.RLock()
	defer p.mu.RUnlock()
	
	avgLatency := time.Duration(0)
	if p.metrics.RequestCount > 0 {
		avgLatency = p.metrics.TotalLatency / time.Duration(p.metrics.RequestCount)
	}
	
	return map[string]any{
		"request_count":    p.metrics.RequestCount,
		"avg_latency_ms":   avgLatency.Milliseconds(),
		"tokens_generated": p.metrics.TokensGenerated,
		"tokens_prompt":    p.metrics.TokensPrompt,
		"cache_hit_rate":   p.getCacheHitRate(),
		"last_request":     p.metrics.LastRequestTime.Format(time.RFC3339),
	}
}

// Health checks if the vLLM server is healthy.
func (p *VLLMProvider) Health(ctx context.Context) error {
	req, err := http.NewRequestWithContext(ctx, "GET", p.baseURL+"/health", nil)
	if err != nil {
		return fmt.Errorf("failed to create health check request: %w", err)
	}
	
	resp, err := p.httpClient.Do(req)
	if err != nil {
		return fmt.Errorf("vLLM server unavailable: %w", err)
	}
	defer resp.Body.Close()
	
	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("vLLM server unhealthy (status %d)", resp.StatusCode)
	}
	
	return nil
}

// GetModels returns available models from the vLLM server.
func (p *VLLMProvider) GetModels(ctx context.Context) ([]string, error) {
	req, err := http.NewRequestWithContext(ctx, "GET", p.baseURL+"/models", nil)
	if err != nil {
		return nil, fmt.Errorf("failed to create models request: %w", err)
	}
	
	if p.apiKey != "" {
		req.Header.Set("Authorization", "Bearer "+p.apiKey)
	}
	
	resp, err := p.httpClient.Do(req)
	if err != nil {
		return nil, fmt.Errorf("failed to get models: %w", err)
	}
	defer resp.Body.Close()
	
	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("failed to get models (status %d)", resp.StatusCode)
	}
	
	var modelsResp struct {
		Data []struct {
			ID string `json:"id"`
		} `json:"data"`
	}
	
	if err := json.NewDecoder(resp.Body).Decode(&modelsResp); err != nil {
		return nil, fmt.Errorf("failed to decode models response: %w", err)
	}
	
	models := make([]string, len(modelsResp.Data))
	for i, m := range modelsResp.Data {
		models[i] = m.ID
	}
	
	return models, nil
}

// SupportsTools returns true if the provider supports tool usage.
func (p *VLLMProvider) SupportsTools() bool {
	// vLLM supports tools through compatible models (e.g., models fine-tuned for function calling)
	return true
}

// String returns a string representation of the provider.
func (p *VLLMProvider) String() string {
	return fmt.Sprintf("vLLM Provider (%s)", p.baseURL)
}

// Close closes any resources held by the provider.
func (p *VLLMProvider) Close() error {
	// Cancel background monitoring goroutines
	if p.cancelBackground != nil {
		p.cancelBackground()
	}
	
	// Close HTTP connections
	p.httpClient.CloseIdleConnections()
	
	log.Debug("vLLM provider closed", "final_metrics", p.GetMetrics())
	
	return nil
}

// GetPrometheusMetrics returns metrics in Prometheus format
func (p *VLLMProvider) GetPrometheusMetrics() string {
	if p.metricsCollector != nil {
		return p.metricsCollector.ExportPrometheusMetrics()
	}
	return ""
}

// EnableMonitoring enables or disables monitoring at runtime
func (p *VLLMProvider) EnableMonitoring(enable bool) error {
	if enable == p.config.EnableMonitoring {
		return nil // No change needed
	}
	
	p.config.EnableMonitoring = enable
	
	if enable {
		// Start monitoring if not already running
		if p.metricsCollector == nil {
			metricsConfig := p.config.MetricsConfig
			if metricsConfig == nil {
				metricsConfig = &MetricsConfig{
					GPUMetricsInterval:    5 * time.Second,
					SystemMetricsInterval: 10 * time.Second,
					MaxLatencyThreshold:   30 * time.Second,
					MaxGPUUtilization:     0.95,
					MaxMemoryUsage:        16 * 1024 * 1024 * 1024, // 16GB
					MinCacheHitRate:       0.80,
					EnablePrometheus:      true,
					PrometheusPort:        9090,
					PrometheusPath:        "/metrics",
					EnableLogging:         true,
					LogInterval:           60 * time.Second,
					EnableAlerts:          true,
				}
			}
			
			p.metricsCollector = NewMetricsCollector(metricsConfig)
			
			// Create new background context
			p.backgroundCtx, p.cancelBackground = context.WithCancel(context.Background())
			
			// Start background collection
			p.metricsCollector.StartBackgroundCollection(p.backgroundCtx)
			
			log.Info("vLLM monitoring enabled at runtime")
		}
	} else {
		// Stop monitoring
		if p.cancelBackground != nil {
			p.cancelBackground()
		}
		p.metricsCollector = nil
		log.Info("vLLM monitoring disabled at runtime")
	}
	
	return nil
}

// UpdateMetricsConfig updates the metrics configuration at runtime
func (p *VLLMProvider) UpdateMetricsConfig(config *MetricsConfig) error {
	if p.metricsCollector == nil {
		return fmt.Errorf("monitoring not enabled")
	}
	
	// Store the new configuration
	p.config.MetricsConfig = config
	
	// Restart monitoring with new config
	if p.cancelBackground != nil {
		p.cancelBackground()
	}
	
	p.metricsCollector = NewMetricsCollector(config)
	p.backgroundCtx, p.cancelBackground = context.WithCancel(context.Background())
	p.metricsCollector.StartBackgroundCollection(p.backgroundCtx)
	
	log.Info("vLLM metrics configuration updated", 
		"gpu_interval", config.GPUMetricsInterval,
		"system_interval", config.SystemMetricsInterval,
		"prometheus_enabled", config.EnablePrometheus)
	
	return nil
}

// GetLatencyPercentiles returns latency percentiles if monitoring is enabled
func (p *VLLMProvider) GetLatencyPercentiles(percentiles []float64) map[float64]time.Duration {
	if p.metricsCollector != nil && p.metricsCollector.latencyHistogram != nil {
		return p.metricsCollector.latencyHistogram.GetPercentiles(percentiles)
	}
	return make(map[float64]time.Duration)
}

// StartPrometheusServer starts an HTTP server to expose Prometheus metrics
func (p *VLLMProvider) StartPrometheusServer() error {
	if p.metricsCollector == nil {
		return fmt.Errorf("monitoring not enabled")
	}
	
	config := p.config.MetricsConfig
	if !config.EnablePrometheus {
		return fmt.Errorf("Prometheus export not enabled in configuration")
	}
	
	mux := http.NewServeMux()
	mux.HandleFunc(config.PrometheusPath, func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/plain")
		metrics := p.GetPrometheusMetrics()
		if metrics == "" {
			w.WriteHeader(http.StatusNoContent)
			return
		}
		w.WriteHeader(http.StatusOK)
		fmt.Fprint(w, metrics)
	})
	
	// Add health check endpoint
	mux.HandleFunc("/health", func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)
		fmt.Fprint(w, `{"status":"healthy","provider":"vllm"}`)
	})
	
	// Add metrics endpoint in JSON format
	mux.HandleFunc("/api/metrics", func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		metrics := p.GetMetrics()
		w.WriteHeader(http.StatusOK)
		json.NewEncoder(w).Encode(metrics)
	})
	
	server := &http.Server{
		Addr:    fmt.Sprintf(":%d", config.PrometheusPort),
		Handler: mux,
	}
	
	go func() {
		log.Info("Starting Prometheus metrics server", 
			"port", config.PrometheusPort, 
			"path", config.PrometheusPath)
		
		if err := server.ListenAndServe(); err != nil && err != http.ErrServerClosed {
			log.Error("Prometheus server error", "error", err)
		}
	}()
	
	return nil
}

// For color output support
func init() {
	// Register vLLM color scheme
	ansi.DefaultColorProfile = ansi.TrueColor
}