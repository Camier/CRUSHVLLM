package main

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"
)

// Mock vLLM server for testing
func createMockVLLMServer() *httptest.Server {
	mux := http.NewServeMux()
	
	// Mock models endpoint
	mux.HandleFunc("/v1/models", func(w http.ResponseWriter, r *http.Request) {
		models := struct {
			Data []Model `json:"data"`
		}{
			Data: []Model{
				{
					ID:      "meta-llama/Llama-2-7b-hf",
					Object:  "model",
					Created: time.Now().Unix(),
					OwnedBy: "vllm",
				},
			},
		}
		json.NewEncoder(w).Encode(models)
	})
	
	// Mock model info endpoint
	mux.HandleFunc("/v1/models/", func(w http.ResponseWriter, r *http.Request) {
		info := ModelInfo{
			ModelName:    "meta-llama/Llama-2-7b-hf",
			MaxTokens:    4096,
			GPUMemoryMB:  6144,
			IsLoaded:     true,
			LoadProgress: 1.0,
		}
		json.NewEncoder(w).Encode(info)
	})
	
	// Mock completions endpoint
	mux.HandleFunc("/v1/completions", func(w http.ResponseWriter, r *http.Request) {
		completion := CompletionResponse{
			ID:      "cmpl-test",
			Object:  "text_completion",
			Created: time.Now().Unix(),
			Model:   "meta-llama/Llama-2-7b-hf",
			Choices: []Choice{
				{
					Index:        0,
					Text:         "This is a test completion response.",
					FinishReason: "length",
				},
			},
			Usage: Usage{
				PromptTokens:     10,
				CompletionTokens: 8,
				TotalTokens:      18,
			},
		}
		json.NewEncoder(w).Encode(completion)
	})
	
	// Mock metrics endpoint
	mux.HandleFunc("/metrics", func(w http.ResponseWriter, r *http.Request) {
		metrics := MetricsResponse{
			GPUUtilization:   75.5,
			GPUMemoryUsed:    4096,
			GPUMemoryTotal:   8192,
			RequestsPerSec:   12.5,
			TokensPerSec:     150.0,
			ActiveRequests:   3,
			QueuedRequests:   1,
			CacheHitRate:     0.85,
			AverageLatencyMS: 120.5,
		}
		json.NewEncoder(w).Encode(metrics)
	})
	
	// Mock cache endpoints
	mux.HandleFunc("/cache/clear", func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusOK)
	})
	
	mux.HandleFunc("/cache/stats", func(w http.ResponseWriter, r *http.Request) {
		stats := CacheStatsResponse{
			TotalKeys:    1000,
			UsedMemoryMB: 512,
			HitRate:      0.85,
			LastCleared:  time.Now().Format(time.RFC3339),
		}
		json.NewEncoder(w).Encode(stats)
	})
	
	return httptest.NewServer(mux)
}

func TestVLLMServer_ListModels(t *testing.T) {
	mockServer := createMockVLLMServer()
	defer mockServer.Close()
	
	vllmServer := NewVLLMServer(mockServer.URL)
	
	ctx := context.Background()
	models, err := vllmServer.ListModels(ctx)
	
	if err != nil {
		t.Fatalf("ListModels failed: %v", err)
	}
	
	if len(models) != 1 {
		t.Fatalf("Expected 1 model, got %d", len(models))
	}
	
	if models[0].ID != "meta-llama/Llama-2-7b-hf" {
		t.Fatalf("Expected model ID 'meta-llama/Llama-2-7b-hf', got '%s'", models[0].ID)
	}
}

func TestVLLMServer_GetModelInfo(t *testing.T) {
	mockServer := createMockVLLMServer()
	defer mockServer.Close()
	
	vllmServer := NewVLLMServer(mockServer.URL)
	
	ctx := context.Background()
	info, err := vllmServer.GetModelInfo(ctx, "meta-llama/Llama-2-7b-hf")
	
	if err != nil {
		t.Fatalf("GetModelInfo failed: %v", err)
	}
	
	if info.ModelName != "meta-llama/Llama-2-7b-hf" {
		t.Fatalf("Expected model name 'meta-llama/Llama-2-7b-hf', got '%s'", info.ModelName)
	}
	
	if info.MaxTokens != 4096 {
		t.Fatalf("Expected max tokens 4096, got %d", info.MaxTokens)
	}
	
	if !info.IsLoaded {
		t.Fatalf("Expected model to be loaded")
	}
}

func TestVLLMServer_GenerateCompletion(t *testing.T) {
	mockServer := createMockVLLMServer()
	defer mockServer.Close()
	
	vllmServer := NewVLLMServer(mockServer.URL)
	
	ctx := context.Background()
	req := CompletionRequest{
		Model:       "meta-llama/Llama-2-7b-hf",
		Prompt:      "Test prompt",
		MaxTokens:   100,
		Temperature: 0.7,
	}
	
	completion, err := vllmServer.GenerateCompletion(ctx, req)
	
	if err != nil {
		t.Fatalf("GenerateCompletion failed: %v", err)
	}
	
	if completion.Model != "meta-llama/Llama-2-7b-hf" {
		t.Fatalf("Expected model 'meta-llama/Llama-2-7b-hf', got '%s'", completion.Model)
	}
	
	if len(completion.Choices) != 1 {
		t.Fatalf("Expected 1 choice, got %d", len(completion.Choices))
	}
	
	if completion.Choices[0].Text != "This is a test completion response." {
		t.Fatalf("Unexpected completion text: %s", completion.Choices[0].Text)
	}
}

func TestVLLMServer_GetMetrics(t *testing.T) {
	mockServer := createMockVLLMServer()
	defer mockServer.Close()
	
	vllmServer := NewVLLMServer(mockServer.URL)
	
	ctx := context.Background()
	metrics, err := vllmServer.GetMetrics(ctx)
	
	if err != nil {
		t.Fatalf("GetMetrics failed: %v", err)
	}
	
	if metrics.GPUUtilization != 75.5 {
		t.Fatalf("Expected GPU utilization 75.5, got %f", metrics.GPUUtilization)
	}
	
	if metrics.ActiveRequests != 3 {
		t.Fatalf("Expected 3 active requests, got %d", metrics.ActiveRequests)
	}
}

func TestVLLMServer_CacheOperations(t *testing.T) {
	mockServer := createMockVLLMServer()
	defer mockServer.Close()
	
	vllmServer := NewVLLMServer(mockServer.URL)
	ctx := context.Background()
	
	// Test clear cache
	err := vllmServer.ClearCache(ctx)
	if err != nil {
		t.Fatalf("ClearCache failed: %v", err)
	}
	
	// Test get cache stats
	stats, err := vllmServer.GetCacheStats(ctx)
	if err != nil {
		t.Fatalf("GetCacheStats failed: %v", err)
	}
	
	if stats.TotalKeys != 1000 {
		t.Fatalf("Expected 1000 cache keys, got %d", stats.TotalKeys)
	}
	
	if stats.HitRate != 0.85 {
		t.Fatalf("Expected hit rate 0.85, got %f", stats.HitRate)
	}
}

func TestExecuteBatchInference(t *testing.T) {
	mockServer := createMockVLLMServer()
	defer mockServer.Close()
	
	vllmClient := NewVLLMServer(mockServer.URL)
	
	ctx := context.Background()
	prompts := []string{
		"Test prompt 1",
		"Test prompt 2",
		"Test prompt 3",
	}
	
	results, err := executeBatchInference(ctx, vllmClient, "meta-llama/Llama-2-7b-hf", prompts, 100, 0.7)
	
	if err != nil {
		t.Fatalf("executeBatchInference failed: %v", err)
	}
	
	if len(results) != 3 {
		t.Fatalf("Expected 3 results, got %d", len(results))
	}
	
	for i, result := range results {
		if result.Model != "meta-llama/Llama-2-7b-hf" {
			t.Fatalf("Result %d: expected model 'meta-llama/Llama-2-7b-hf', got '%s'", i, result.Model)
		}
		
		if len(result.Choices) != 1 {
			t.Fatalf("Result %d: expected 1 choice, got %d", i, len(result.Choices))
		}
	}
}

func TestExecuteSequentialThinking(t *testing.T) {
	mockServer := createMockVLLMServer()
	defer mockServer.Close()
	
	vllmClient := NewVLLMServer(mockServer.URL)
	
	ctx := context.Background()
	problem := "How do we solve world hunger?"
	
	result, err := executeSequentialThinking(ctx, vllmClient, "meta-llama/Llama-2-7b-hf", problem, 3)
	
	if err != nil {
		t.Fatalf("executeSequentialThinking failed: %v", err)
	}
	
	if result == "" {
		t.Fatalf("Expected non-empty result")
	}
	
	// Check that result contains expected sections
	if !contains(result, "Sequential Thinking Process") {
		t.Fatalf("Result should contain 'Sequential Thinking Process'")
	}
	
	if !contains(result, "Final Answer") {
		t.Fatalf("Result should contain 'Final Answer'")
	}
}

func TestNewVLLMServer(t *testing.T) {
	// Test with custom URL
	server := NewVLLMServer("http://custom:8000")
	if server.baseURL != "http://custom:8000" {
		t.Fatalf("Expected baseURL 'http://custom:8000', got '%s'", server.baseURL)
	}
	
	// Test with default URL
	server = NewVLLMServer("")
	if server.baseURL != defaultVLLMURL {
		t.Fatalf("Expected default baseURL '%s', got '%s'", defaultVLLMURL, server.baseURL)
	}
	
	// Test that trailing slash is removed
	server = NewVLLMServer("http://localhost:8000/")
	if server.baseURL != "http://localhost:8000" {
		t.Fatalf("Expected baseURL without trailing slash, got '%s'", server.baseURL)
	}
}

// Helper function
func contains(s, substr string) bool {
	return len(s) >= len(substr) && (s == substr || len(substr) > 0 && 
		(s[:len(substr)] == substr || s[len(s)-len(substr):] == substr || 
		 strings.Contains(s, substr)))
}

// Import the strings package for the contains helper
import "strings"

// Benchmark tests
func BenchmarkListModels(b *testing.B) {
	mockServer := createMockVLLMServer()
	defer mockServer.Close()
	
	vllmServer := NewVLLMServer(mockServer.URL)
	ctx := context.Background()
	
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, err := vllmServer.ListModels(ctx)
		if err != nil {
			b.Fatalf("ListModels failed: %v", err)
		}
	}
}

func BenchmarkGenerateCompletion(b *testing.B) {
	mockServer := createMockVLLMServer()
	defer mockServer.Close()
	
	vllmServer := NewVLLMServer(mockServer.URL)
	ctx := context.Background()
	req := CompletionRequest{
		Model:       "meta-llama/Llama-2-7b-hf",
		Prompt:      "Test prompt",
		MaxTokens:   100,
		Temperature: 0.7,
	}
	
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, err := vllmServer.GenerateCompletion(ctx, req)
		if err != nil {
			b.Fatalf("GenerateCompletion failed: %v", err)
		}
	}
}