package main

import (
	"context"
	"encoding/json"
	"fmt"
	"log"
	"net/http"
	"os"
	"strconv"
	"strings"
	"time"

	"github.com/mark3labs/mcp-go/mcp"
	"github.com/mark3labs/mcp-go/server"
)

const (
	serverName    = "vllm-mcp-server"
	serverVersion = "1.0.0"
	defaultVLLMURL = "http://localhost:8000"
)

type VLLMServer struct {
	baseURL    string
	httpClient *http.Client
	logger     *log.Logger
}

type VLLMClient interface {
	ListModels(ctx context.Context) ([]Model, error)
	GetModelInfo(ctx context.Context, modelName string) (*ModelInfo, error)
	GenerateCompletion(ctx context.Context, req CompletionRequest) (*CompletionResponse, error)
	GetMetrics(ctx context.Context) (*MetricsResponse, error)
	ClearCache(ctx context.Context) error
	GetCacheStats(ctx context.Context) (*CacheStatsResponse, error)
}

type Model struct {
	ID      string `json:"id"`
	Object  string `json:"object"`
	Created int64  `json:"created"`
	OwnedBy string `json:"owned_by"`
}

type ModelInfo struct {
	ModelName    string  `json:"model_name"`
	MaxTokens    int     `json:"max_tokens"`
	GPUMemoryMB  int     `json:"gpu_memory_mb"`
	IsLoaded     bool    `json:"is_loaded"`
	LoadProgress float64 `json:"load_progress"`
}

type CompletionRequest struct {
	Model       string  `json:"model"`
	Prompt      string  `json:"prompt"`
	MaxTokens   int     `json:"max_tokens,omitempty"`
	Temperature float64 `json:"temperature,omitempty"`
	Stream      bool    `json:"stream,omitempty"`
}

type CompletionResponse struct {
	ID      string   `json:"id"`
	Object  string   `json:"object"`
	Created int64    `json:"created"`
	Model   string   `json:"model"`
	Choices []Choice `json:"choices"`
	Usage   Usage    `json:"usage"`
}

type Choice struct {
	Index        int    `json:"index"`
	Text         string `json:"text"`
	FinishReason string `json:"finish_reason"`
}

type Usage struct {
	PromptTokens     int `json:"prompt_tokens"`
	CompletionTokens int `json:"completion_tokens"`
	TotalTokens      int `json:"total_tokens"`
}

type MetricsResponse struct {
	GPUUtilization   float64 `json:"gpu_utilization"`
	GPUMemoryUsed    int64   `json:"gpu_memory_used"`
	GPUMemoryTotal   int64   `json:"gpu_memory_total"`
	RequestsPerSec   float64 `json:"requests_per_sec"`
	TokensPerSec     float64 `json:"tokens_per_sec"`
	ActiveRequests   int     `json:"active_requests"`
	QueuedRequests   int     `json:"queued_requests"`
	CacheHitRate     float64 `json:"cache_hit_rate"`
	AverageLatencyMS float64 `json:"average_latency_ms"`
}

type CacheStatsResponse struct {
	TotalKeys   int64   `json:"total_keys"`
	UsedMemoryMB int64  `json:"used_memory_mb"`
	HitRate     float64 `json:"hit_rate"`
	LastCleared string  `json:"last_cleared"`
}

func NewVLLMServer(baseURL string) *VLLMServer {
	if baseURL == "" {
		baseURL = defaultVLLMURL
	}
	
	return &VLLMServer{
		baseURL: strings.TrimSuffix(baseURL, "/"),
		httpClient: &http.Client{
			Timeout: 30 * time.Second,
		},
		logger: log.New(os.Stderr, "[vLLM-MCP] ", log.LstdFlags|log.Lshortfile),
	}
}

func (v *VLLMServer) ListModels(ctx context.Context) ([]Model, error) {
	url := fmt.Sprintf("%s/v1/models", v.baseURL)
	
	req, err := http.NewRequestWithContext(ctx, "GET", url, nil)
	if err != nil {
		return nil, fmt.Errorf("creating request: %w", err)
	}
	
	resp, err := v.httpClient.Do(req)
	if err != nil {
		return nil, fmt.Errorf("making request: %w", err)
	}
	defer resp.Body.Close()
	
	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("unexpected status code: %d", resp.StatusCode)
	}
	
	var result struct {
		Data []Model `json:"data"`
	}
	
	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		return nil, fmt.Errorf("decoding response: %w", err)
	}
	
	return result.Data, nil
}

func (v *VLLMServer) GetModelInfo(ctx context.Context, modelName string) (*ModelInfo, error) {
	url := fmt.Sprintf("%s/v1/models/%s", v.baseURL, modelName)
	
	req, err := http.NewRequestWithContext(ctx, "GET", url, nil)
	if err != nil {
		return nil, fmt.Errorf("creating request: %w", err)
	}
	
	resp, err := v.httpClient.Do(req)
	if err != nil {
		return nil, fmt.Errorf("making request: %w", err)
	}
	defer resp.Body.Close()
	
	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("unexpected status code: %d", resp.StatusCode)
	}
	
	var info ModelInfo
	if err := json.NewDecoder(resp.Body).Decode(&info); err != nil {
		return nil, fmt.Errorf("decoding response: %w", err)
	}
	
	return &info, nil
}

func (v *VLLMServer) GenerateCompletion(ctx context.Context, req CompletionRequest) (*CompletionResponse, error) {
	url := fmt.Sprintf("%s/v1/completions", v.baseURL)
	
	reqBody, err := json.Marshal(req)
	if err != nil {
		return nil, fmt.Errorf("encoding request: %w", err)
	}
	
	httpReq, err := http.NewRequestWithContext(ctx, "POST", url, strings.NewReader(string(reqBody)))
	if err != nil {
		return nil, fmt.Errorf("creating request: %w", err)
	}
	
	httpReq.Header.Set("Content-Type", "application/json")
	
	resp, err := v.httpClient.Do(httpReq)
	if err != nil {
		return nil, fmt.Errorf("making request: %w", err)
	}
	defer resp.Body.Close()
	
	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("unexpected status code: %d", resp.StatusCode)
	}
	
	var completion CompletionResponse
	if err := json.NewDecoder(resp.Body).Decode(&completion); err != nil {
		return nil, fmt.Errorf("decoding response: %w", err)
	}
	
	return &completion, nil
}

func (v *VLLMServer) GetMetrics(ctx context.Context) (*MetricsResponse, error) {
	url := fmt.Sprintf("%s/metrics", v.baseURL)
	
	req, err := http.NewRequestWithContext(ctx, "GET", url, nil)
	if err != nil {
		return nil, fmt.Errorf("creating request: %w", err)
	}
	
	resp, err := v.httpClient.Do(req)
	if err != nil {
		return nil, fmt.Errorf("making request: %w", err)
	}
	defer resp.Body.Close()
	
	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("unexpected status code: %d", resp.StatusCode)
	}
	
	var metrics MetricsResponse
	if err := json.NewDecoder(resp.Body).Decode(&metrics); err != nil {
		return nil, fmt.Errorf("decoding response: %w", err)
	}
	
	return &metrics, nil
}

func (v *VLLMServer) ClearCache(ctx context.Context) error {
	url := fmt.Sprintf("%s/cache/clear", v.baseURL)
	
	req, err := http.NewRequestWithContext(ctx, "POST", url, nil)
	if err != nil {
		return fmt.Errorf("creating request: %w", err)
	}
	
	resp, err := v.httpClient.Do(req)
	if err != nil {
		return fmt.Errorf("making request: %w", err)
	}
	defer resp.Body.Close()
	
	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("unexpected status code: %d", resp.StatusCode)
	}
	
	return nil
}

func (v *VLLMServer) GetCacheStats(ctx context.Context) (*CacheStatsResponse, error) {
	url := fmt.Sprintf("%s/cache/stats", v.baseURL)
	
	req, err := http.NewRequestWithContext(ctx, "GET", url, nil)
	if err != nil {
		return nil, fmt.Errorf("creating request: %w", err)
	}
	
	resp, err := v.httpClient.Do(req)
	if err != nil {
		return nil, fmt.Errorf("making request: %w", err)
	}
	defer resp.Body.Close()
	
	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("unexpected status code: %d", resp.StatusCode)
	}
	
	var stats CacheStatsResponse
	if err := json.NewDecoder(resp.Body).Decode(&stats); err != nil {
		return nil, fmt.Errorf("decoding response: %w", err)
	}
	
	return &stats, nil
}

func setupMCPServer(vllmClient VLLMClient) *server.MCPServer {
	s := server.NewMCPServer(serverName, serverVersion)
	
	// Tool: list_models
	s.AddTool(mcp.Tool{
		Name:        "list_models",
		Description: "List all available models in vLLM",
		InputSchema: mcp.ToolInputSchema{
			Type:       "object",
			Properties: map[string]interface{}{},
		},
	}, func(ctx context.Context, request mcp.CallToolRequest) (*mcp.CallToolResult, error) {
		models, err := vllmClient.ListModels(ctx)
		if err != nil {
			return mcp.NewToolResultError(fmt.Sprintf("Failed to list models: %v", err)), nil
		}
		
		modelsJSON, _ := json.Marshal(models)
		return mcp.NewToolResultText(string(modelsJSON)), nil
	})
	
	// Tool: get_model_info
	s.AddTool(mcp.Tool{
		Name:        "get_model_info",
		Description: "Get detailed information about a specific model",
		InputSchema: mcp.ToolInputSchema{
			Type: "object",
			Properties: map[string]interface{}{
				"model_name": map[string]interface{}{
					"type":        "string",
					"description": "Name of the model to get information about",
				},
			},
			Required: []string{"model_name"},
		},
	}, func(ctx context.Context, request mcp.CallToolRequest) (*mcp.CallToolResult, error) {
		modelName, ok := request.Params.Arguments["model_name"].(string)
		if !ok {
			return mcp.NewToolResultError("model_name must be a string"), nil
		}
		
		info, err := vllmClient.GetModelInfo(ctx, modelName)
		if err != nil {
			return mcp.NewToolResultError(fmt.Sprintf("Failed to get model info: %v", err)), nil
		}
		
		infoJSON, _ := json.Marshal(info)
		return mcp.NewToolResultText(string(infoJSON)), nil
	})
	
	// Tool: generate_completion
	s.AddTool(mcp.Tool{
		Name:        "generate_completion",
		Description: "Generate text completion using vLLM",
		InputSchema: mcp.ToolInputSchema{
			Type: "object",
			Properties: map[string]interface{}{
				"model": map[string]interface{}{
					"type":        "string",
					"description": "Model name to use for completion",
				},
				"prompt": map[string]interface{}{
					"type":        "string",
					"description": "Text prompt for completion",
				},
				"max_tokens": map[string]interface{}{
					"type":        "integer",
					"description": "Maximum number of tokens to generate",
					"default":     100,
				},
				"temperature": map[string]interface{}{
					"type":        "number",
					"description": "Temperature for sampling (0.0 to 2.0)",
					"default":     0.7,
				},
			},
			Required: []string{"model", "prompt"},
		},
	}, func(ctx context.Context, request mcp.CallToolRequest) (*mcp.CallToolResult, error) {
		model, ok := request.Params.Arguments["model"].(string)
		if !ok {
			return mcp.NewToolResultError("model must be a string"), nil
		}
		
		prompt, ok := request.Params.Arguments["prompt"].(string)
		if !ok {
			return mcp.NewToolResultError("prompt must be a string"), nil
		}
		
		completionReq := CompletionRequest{
			Model:       model,
			Prompt:      prompt,
			MaxTokens:   100,
			Temperature: 0.7,
		}
		
		if maxTokens, ok := request.Params.Arguments["max_tokens"]; ok {
			if mt, ok := maxTokens.(float64); ok {
				completionReq.MaxTokens = int(mt)
			}
		}
		
		if temp, ok := request.Params.Arguments["temperature"]; ok {
			if t, ok := temp.(float64); ok {
				completionReq.Temperature = t
			}
		}
		
		completion, err := vllmClient.GenerateCompletion(ctx, completionReq)
		if err != nil {
			return mcp.NewToolResultError(fmt.Sprintf("Failed to generate completion: %v", err)), nil
		}
		
		completionJSON, _ := json.Marshal(completion)
		return mcp.NewToolResultText(string(completionJSON)), nil
	})
	
	// Tool: get_metrics
	s.AddTool(mcp.Tool{
		Name:        "get_metrics",
		Description: "Get performance metrics from vLLM server",
		InputSchema: mcp.ToolInputSchema{
			Type:       "object",
			Properties: map[string]interface{}{},
		},
	}, func(ctx context.Context, request mcp.CallToolRequest) (*mcp.CallToolResult, error) {
		metrics, err := vllmClient.GetMetrics(ctx)
		if err != nil {
			return mcp.NewToolResultError(fmt.Sprintf("Failed to get metrics: %v", err)), nil
		}
		
		metricsJSON, _ := json.Marshal(metrics)
		return mcp.NewToolResultText(string(metricsJSON)), nil
	})
	
	// Tool: clear_cache
	s.AddTool(mcp.Tool{
		Name:        "clear_cache",
		Description: "Clear the vLLM cache",
		InputSchema: mcp.ToolInputSchema{
			Type:       "object",
			Properties: map[string]interface{}{},
		},
	}, func(ctx context.Context, request mcp.CallToolRequest) (*mcp.CallToolResult, error) {
		if err := vllmClient.ClearCache(ctx); err != nil {
			return mcp.NewToolResultError(fmt.Sprintf("Failed to clear cache: %v", err)), nil
		}
		
		return mcp.NewToolResultText("Cache cleared successfully"), nil
	})
	
	// Tool: get_cache_stats
	s.AddTool(mcp.Tool{
		Name:        "get_cache_stats",
		Description: "Get cache statistics from vLLM server",
		InputSchema: mcp.ToolInputSchema{
			Type:       "object",
			Properties: map[string]interface{}{},
		},
	}, func(ctx context.Context, request mcp.CallToolRequest) (*mcp.CallToolResult, error) {
		stats, err := vllmClient.GetCacheStats(ctx)
		if err != nil {
			return mcp.NewToolResultError(fmt.Sprintf("Failed to get cache stats: %v", err)), nil
		}
		
		statsJSON, _ := json.Marshal(stats)
		return mcp.NewToolResultText(string(statsJSON)), nil
	})
	
	// Tool: sequential_thinking
	s.AddTool(mcp.Tool{
		Name:        "sequential_thinking",
		Description: "Execute sequential thinking with step-by-step reasoning",
		InputSchema: mcp.ToolInputSchema{
			Type: "object",
			Properties: map[string]interface{}{
				"model": map[string]interface{}{
					"type":        "string",
					"description": "Model name to use for thinking",
				},
				"problem": map[string]interface{}{
					"type":        "string",
					"description": "Problem to solve with sequential thinking",
				},
				"steps": map[string]interface{}{
					"type":        "integer",
					"description": "Maximum number of thinking steps",
					"default":     5,
				},
			},
			Required: []string{"model", "problem"},
		},
	}, func(ctx context.Context, request mcp.CallToolRequest) (*mcp.CallToolResult, error) {
		model, ok := request.Params.Arguments["model"].(string)
		if !ok {
			return mcp.NewToolResultError("model must be a string"), nil
		}
		
		problem, ok := request.Params.Arguments["problem"].(string)
		if !ok {
			return mcp.NewToolResultError("problem must be a string"), nil
		}
		
		maxSteps := 5
		if steps, ok := request.Params.Arguments["steps"]; ok {
			if s, ok := steps.(float64); ok {
				maxSteps = int(s)
			}
		}
		
		result, err := executeSequentialThinking(ctx, vllmClient, model, problem, maxSteps)
		if err != nil {
			return mcp.NewToolResultError(fmt.Sprintf("Sequential thinking failed: %v", err)), nil
		}
		
		return mcp.NewToolResultText(result), nil
	})
	
	// Tool: batch_inference
	s.AddTool(mcp.Tool{
		Name:        "batch_inference",
		Description: "Execute optimized batch inference for multiple prompts",
		InputSchema: mcp.ToolInputSchema{
			Type: "object",
			Properties: map[string]interface{}{
				"model": map[string]interface{}{
					"type":        "string",
					"description": "Model name to use for batch inference",
				},
				"prompts": map[string]interface{}{
					"type": "array",
					"items": map[string]interface{}{
						"type": "string",
					},
					"description": "List of prompts to process",
				},
				"max_tokens": map[string]interface{}{
					"type":        "integer",
					"description": "Maximum tokens per completion",
					"default":     100,
				},
				"temperature": map[string]interface{}{
					"type":        "number",
					"description": "Temperature for sampling",
					"default":     0.7,
				},
			},
			Required: []string{"model", "prompts"},
		},
	}, func(ctx context.Context, request mcp.CallToolRequest) (*mcp.CallToolResult, error) {
		model, ok := request.Params.Arguments["model"].(string)
		if !ok {
			return mcp.NewToolResultError("model must be a string"), nil
		}
		
		promptsInterface, ok := request.Params.Arguments["prompts"].([]interface{})
		if !ok {
			return mcp.NewToolResultError("prompts must be an array"), nil
		}
		
		prompts := make([]string, len(promptsInterface))
		for i, p := range promptsInterface {
			prompt, ok := p.(string)
			if !ok {
				return mcp.NewToolResultError("all prompts must be strings"), nil
			}
			prompts[i] = prompt
		}
		
		maxTokens := 100
		if mt, ok := request.Params.Arguments["max_tokens"]; ok {
			if tokens, ok := mt.(float64); ok {
				maxTokens = int(tokens)
			}
		}
		
		temperature := 0.7
		if temp, ok := request.Params.Arguments["temperature"]; ok {
			if t, ok := temp.(float64); ok {
				temperature = t
			}
		}
		
		results, err := executeBatchInference(ctx, vllmClient, model, prompts, maxTokens, temperature)
		if err != nil {
			return mcp.NewToolResultError(fmt.Sprintf("Batch inference failed: %v", err)), nil
		}
		
		resultsJSON, _ := json.Marshal(results)
		return mcp.NewToolResultText(string(resultsJSON)), nil
	})
	
	return s
}

func executeSequentialThinking(ctx context.Context, client VLLMClient, model, problem string, maxSteps int) (string, error) {
	thinkingSteps := []string{}
	currentProblem := problem
	
	for i := 0; i < maxSteps; i++ {
		prompt := fmt.Sprintf(`Think step by step about this problem:
%s

Previous steps:
%s

What is the next logical step in solving this problem? Provide just the next step, not the full solution.`,
			currentProblem, strings.Join(thinkingSteps, "\n"))
		
		req := CompletionRequest{
			Model:       model,
			Prompt:      prompt,
			MaxTokens:   200,
			Temperature: 0.3,
		}
		
		resp, err := client.GenerateCompletion(ctx, req)
		if err != nil {
			return "", fmt.Errorf("step %d failed: %w", i+1, err)
		}
		
		if len(resp.Choices) == 0 {
			return "", fmt.Errorf("no response from model at step %d", i+1)
		}
		
		step := strings.TrimSpace(resp.Choices[0].Text)
		thinkingSteps = append(thinkingSteps, fmt.Sprintf("Step %d: %s", i+1, step))
		
		// Check if we've reached a conclusion
		if strings.Contains(strings.ToLower(step), "conclusion") ||
			strings.Contains(strings.ToLower(step), "final answer") ||
			strings.Contains(strings.ToLower(step), "therefore") {
			break
		}
	}
	
	// Generate final synthesis
	synthesisPrompt := fmt.Sprintf(`Based on these thinking steps, provide a comprehensive solution:

Original problem: %s

Thinking steps:
%s

Provide a clear, complete answer based on this sequential reasoning.`,
		problem, strings.Join(thinkingSteps, "\n"))
	
	req := CompletionRequest{
		Model:       model,
		Prompt:      synthesisPrompt,
		MaxTokens:   500,
		Temperature: 0.2,
	}
	
	resp, err := client.GenerateCompletion(ctx, req)
	if err != nil {
		return "", fmt.Errorf("synthesis failed: %w", err)
	}
	
	if len(resp.Choices) == 0 {
		return "", fmt.Errorf("no synthesis response from model")
	}
	
	result := fmt.Sprintf("Sequential Thinking Process:\n\n%s\n\nFinal Answer:\n%s",
		strings.Join(thinkingSteps, "\n"), strings.TrimSpace(resp.Choices[0].Text))
	
	return result, nil
}

func executeBatchInference(ctx context.Context, client VLLMClient, model string, prompts []string, maxTokens int, temperature float64) ([]CompletionResponse, error) {
	results := make([]CompletionResponse, 0, len(prompts))
	
	// Process in parallel with controlled concurrency
	const maxConcurrency = 5
	semaphore := make(chan struct{}, maxConcurrency)
	resultChan := make(chan struct {
		index int
		resp  *CompletionResponse
		err   error
	}, len(prompts))
	
	for i, prompt := range prompts {
		go func(index int, p string) {
			semaphore <- struct{}{} // Acquire
			defer func() { <-semaphore }() // Release
			
			req := CompletionRequest{
				Model:       model,
				Prompt:      p,
				MaxTokens:   maxTokens,
				Temperature: temperature,
			}
			
			resp, err := client.GenerateCompletion(ctx, req)
			resultChan <- struct {
				index int
				resp  *CompletionResponse
				err   error
			}{index, resp, err}
		}(i, prompt)
	}
	
	// Collect results in order
	orderedResults := make([]*CompletionResponse, len(prompts))
	var errs []string
	
	for i := 0; i < len(prompts); i++ {
		result := <-resultChan
		if result.err != nil {
			errs = append(errs, fmt.Sprintf("prompt %d: %v", result.index, result.err))
		} else {
			orderedResults[result.index] = result.resp
		}
	}
	
	if len(errs) > 0 {
		return nil, fmt.Errorf("batch inference errors: %s", strings.Join(errs, "; "))
	}
	
	for _, resp := range orderedResults {
		if resp != nil {
			results = append(results, *resp)
		}
	}
	
	return results, nil
}

func main() {
	// Get vLLM URL from environment or use default
	vllmURL := os.Getenv("VLLM_URL")
	if vllmURL == "" {
		vllmURL = defaultVLLMURL
	}
	
	// Get port from environment or use default
	port := os.Getenv("MCP_PORT")
	if port == "" {
		port = "8080"
	}
	
	// Validate port
	if _, err := strconv.Atoi(port); err != nil {
		log.Fatalf("Invalid port: %v", err)
	}
	
	logger := log.New(os.Stderr, "[vLLM-MCP] ", log.LstdFlags|log.Lshortfile)
	logger.Printf("Starting vLLM MCP Server on port %s, connecting to vLLM at %s", port, vllmURL)
	
	// Create vLLM client
	vllmClient := NewVLLMServer(vllmURL)
	
	// Test connection to vLLM
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()
	
	if _, err := vllmClient.ListModels(ctx); err != nil {
		logger.Printf("Warning: Cannot connect to vLLM server at %s: %v", vllmURL, err)
		logger.Printf("Server will start but tools may not work until vLLM is available")
	} else {
		logger.Printf("Successfully connected to vLLM server")
	}
	
	// Setup MCP server
	mcpServer := setupMCPServer(vllmClient)
	
	// Start server
	logger.Printf("vLLM MCP Server ready - Tools: list_models, get_model_info, generate_completion, get_metrics, clear_cache, get_cache_stats, sequential_thinking, batch_inference")
	
	if err := mcpServer.Serve(ctx); err != nil {
		logger.Fatalf("Server error: %v", err)
	}
}