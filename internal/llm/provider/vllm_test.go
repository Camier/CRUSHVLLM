//go:build ignore
// +build ignore

package provider

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"

	"github.com/charmbracelet/crush/internal/config"
	"github.com/charmbracelet/crush/internal/message"
	"github.com/stretchr/testify/require"
)

func TestVLLMProvider_NewProvider(t *testing.T) {
	tests := []struct {
		name    string
		cfg     config.Provider
		wantURL string
		wantErr bool
	}{
		{
			name: "default configuration",
			cfg:  config.Provider{},
			wantURL: "http://localhost:8000/v1",
			wantErr: false,
		},
		{
			name: "custom base URL",
			cfg: config.Provider{
				BaseURL: "http://gpu-server:9000/v1",
			},
			wantURL: "http://gpu-server:9000/v1",
			wantErr: false,
		},
		{
			name: "with extra config",
			cfg: config.Provider{
				BaseURL: "http://localhost:8000/v1",
				ExtraConfig: map[string]any{
					"gpu_memory_utilization": 0.95,
					"max_model_len":          16384,
					"quantization":           "awq",
					"model_path":             "/models/qwen-32b",
				},
			},
			wantURL: "http://localhost:8000/v1",
			wantErr: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			p, err := NewVLLMProvider(tt.cfg)
			if tt.wantErr {
				require.Error(t, err)
				return
			}
			require.NoError(t, err)
			require.NotNil(t, p)
			require.Equal(t, tt.wantURL, p.baseURL)
			require.Equal(t, "vllm", p.ID())
			require.Equal(t, "vLLM (Local)", p.Name())
		})
	}
}

func TestVLLMProvider_Send(t *testing.T) {
	// Create mock vLLM server
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		require.Equal(t, "/v1/chat/completions", r.URL.Path)
		require.Equal(t, "POST", r.Method)

		// Decode request
		var req VLLMRequest
		err := json.NewDecoder(r.Body).Decode(&req)
		require.NoError(t, err)

		// Validate request
		require.NotEmpty(t, req.Messages)
		require.Equal(t, float32(0.7), req.Temperature)
		require.Equal(t, 2048, req.MaxTokens)

		// Send mock response
		resp := VLLMResponse{
			ID:      "test-123",
			Object:  "chat.completion",
			Created: time.Now().Unix(),
			Model:   "test-model",
			Choices: []struct {
				Index        int    `json:"index"`
				Message      any    `json:"message,omitempty"`
				Text         string `json:"text,omitempty"`
				FinishReason string `json:"finish_reason"`
				Logprobs     any    `json:"logprobs,omitempty"`
			}{
				{
					Index: 0,
					Message: map[string]any{
						"role":    "assistant",
						"content": "Test response from vLLM",
					},
					FinishReason: "stop",
				},
			},
			Usage: struct {
				PromptTokens     int `json:"prompt_tokens"`
				CompletionTokens int `json:"completion_tokens"`
				TotalTokens      int `json:"total_tokens"`
			}{
				PromptTokens:     10,
				CompletionTokens: 5,
				TotalTokens:      15,
			},
		}

		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(resp)
	}))
	defer server.Close()

	// Create provider with mock server URL
	p, err := NewVLLMProvider(config.Provider{
		BaseURL: server.URL + "/v1",
	})
	require.NoError(t, err)

	// Create test messages
	msgs := []message.Message{
		message.NewSystem("You are a helpful assistant"),
		message.NewUser("Hello, how are you?"),
	}

	// Send request
	ctx := context.Background()
	resp, err := p.Send(ctx, msgs, WithTemperature(0.7), WithMaxTokens(2048))
	require.NoError(t, err)
	require.NotNil(t, resp)
	require.Equal(t, "Test response from vLLM", resp.Content)
	require.Equal(t, 10, resp.Usage.InputTokens)
	require.Equal(t, 5, resp.Usage.OutputTokens)
}

func TestVLLMProvider_Health(t *testing.T) {
	// Create mock server with health endpoint
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/v1/health" {
			w.WriteHeader(http.StatusOK)
			w.Write([]byte("healthy"))
			return
		}
		w.WriteHeader(http.StatusNotFound)
	}))
	defer server.Close()

	// Create provider
	p, err := NewVLLMProvider(config.Provider{
		BaseURL: server.URL + "/v1",
	})
	require.NoError(t, err)

	// Check health
	ctx := context.Background()
	err = p.Health(ctx)
	require.NoError(t, err)
}

func TestVLLMProvider_GetModels(t *testing.T) {
	// Create mock server with models endpoint
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/v1/models" {
			resp := map[string]any{
				"data": []map[string]string{
					{"id": "qwen-32b"},
					{"id": "mistral-7b"},
					{"id": "deepseek-33b"},
				},
			}
			w.Header().Set("Content-Type", "application/json")
			json.NewEncoder(w).Encode(resp)
			return
		}
		w.WriteHeader(http.StatusNotFound)
	}))
	defer server.Close()

	// Create provider
	p, err := NewVLLMProvider(config.Provider{
		BaseURL: server.URL + "/v1",
	})
	require.NoError(t, err)

	// Get models
	ctx := context.Background()
	models, err := p.GetModels(ctx)
	require.NoError(t, err)
	require.Len(t, models, 3)
	require.Contains(t, models, "qwen-32b")
	require.Contains(t, models, "mistral-7b")
	require.Contains(t, models, "deepseek-33b")
}

func TestVLLMProvider_Metrics(t *testing.T) {
	// Create provider
	p, err := NewVLLMProvider(config.Provider{})
	require.NoError(t, err)

	// Simulate some requests
	p.metrics.RequestCount = 100
	p.metrics.TotalLatency = 10 * time.Second
	p.metrics.TokensGenerated = 5000
	p.metrics.TokensPrompt = 2000
	p.metrics.CacheHits = 30
	p.metrics.CacheMisses = 70

	// Get metrics
	metrics := p.GetMetrics()
	require.Equal(t, int64(100), metrics["request_count"])
	require.Equal(t, int64(100), metrics["avg_latency_ms"]) // 10s / 100 = 100ms
	require.Equal(t, int64(5000), metrics["tokens_generated"])
	require.Equal(t, int64(2000), metrics["tokens_prompt"])
	require.Equal(t, 30.0, metrics["cache_hit_rate"])
}

func TestVLLMProvider_BuildRequest(t *testing.T) {
	p, err := NewVLLMProvider(config.Provider{})
	require.NoError(t, err)

	msgs := []message.Message{
		message.NewSystem("System prompt"),
		message.NewUser("User message"),
		message.NewAssistant("Assistant response"),
	}

	req := p.buildRequest(msgs,
		WithTemperature(0.5),
		WithTopP(0.95),
		WithMaxTokens(1024),
		WithStop([]string{"END", "STOP"}),
	)

	require.Equal(t, float32(0.5), req.Temperature)
	require.Equal(t, float32(0.95), req.TopP)
	require.Equal(t, 1024, req.MaxTokens)
	require.Equal(t, []string{"END", "STOP"}, req.Stop)
	require.Len(t, req.Messages, 3)
	require.Equal(t, "system", req.Messages[0]["role"])
	require.Equal(t, "System prompt", req.Messages[0]["content"])
}

func TestVLLMProvider_Retry(t *testing.T) {
	attempts := 0
	// Create mock server that fails first 2 times
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		attempts++
		if attempts < 3 {
			w.WriteHeader(http.StatusInternalServerError)
			return
		}
		
		// Success on third attempt
		resp := VLLMResponse{
			ID: "success",
			Choices: []struct {
				Index        int    `json:"index"`
				Message      any    `json:"message,omitempty"`
				Text         string `json:"text,omitempty"`
				FinishReason string `json:"finish_reason"`
				Logprobs     any    `json:"logprobs,omitempty"`
			}{
				{
					Message: map[string]any{
						"content": "Success after retries",
					},
				},
			},
		}
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(resp)
	}))
	defer server.Close()

	// Create provider with retry config
	p, err := NewVLLMProvider(config.Provider{
		BaseURL: server.URL + "/v1",
	})
	require.NoError(t, err)
	p.config.MaxRetries = 3

	// Send request - should succeed after retries
	ctx := context.Background()
	msgs := []message.Message{message.NewUser("Test")}
	resp, err := p.Send(ctx, msgs)
	require.NoError(t, err)
	require.Equal(t, "Success after retries", resp.Content)
	require.Equal(t, 3, attempts)
}