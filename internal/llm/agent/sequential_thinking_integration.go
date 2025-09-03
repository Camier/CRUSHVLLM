package agent

import (
	"context"
	"fmt"
	"log/slog"

	"github.com/charmbracelet/crush/internal/config"
	"github.com/charmbracelet/crush/internal/llm/provider"
	"github.com/charmbracelet/crush/internal/message"
)

// Example integration showing how to use the Sequential Thinking Agent with vLLM

// CreateSequentialThinkingWithVLLM creates a sequential thinking agent configured for vLLM.
func CreateSequentialThinkingWithVLLM(ctx context.Context) (*SequentialThinkingAgent, error) {
	// vLLM provider configuration
	vllmConfig := &config.ProviderConfig{
		ID:      config.ProviderID("vllm"),
		Type:    "openai", // vLLM is OpenAI-compatible
		BaseURL: "http://localhost:8000/v1", // Default vLLM server
		APIKey:  "", // Usually not needed for local vLLM
		ExtraConfig: map[string]any{
			"gpu_memory_utilization": 0.9,
			"max_model_len":          8192,
			"tensor_parallel_size":   1,
			"quantization":           "", // Can be "awq", "gptq", etc.
			"enable_caching":         true,
		},
	}

	// Agent configuration optimized for local inference
	agentConfig := config.Agent{
		ID:    "sequential_thinker",
		Name:  "Sequential Thinking Agent (vLLM)",
		Model: config.SelectedModelTypeLarge,
		ExtraConfig: map[string]any{
			"reasoning_depth":      3,    // Balanced depth for local inference
			"parallel_stages":      false, // Sequential for better cache utilization
			"max_stage_time":       60,    // Longer timeout for local models
			"enable_reflection":    true,
			"thinking_temperature": 0.3,   // Lower for more focused reasoning
			"max_thinking_tokens":  2048,  // Reasonable limit for local models
			"verbose_thinking":     true,  // Enable for debugging
		},
	}

	// Create the sequential thinking agent
	agent, err := NewSequentialThinkingAgent(ctx, agentConfig, vllmConfig)
	if err != nil {
		return nil, fmt.Errorf("failed to create sequential thinking agent: %w", err)
	}

	slog.Info("Created Sequential Thinking Agent with vLLM",
		"provider", vllmConfig.ID,
		"base_url", vllmConfig.BaseURL,
		"reasoning_depth", agentConfig.ExtraConfig["reasoning_depth"],
	)

	return agent, nil
}

// OptimizedVLLMConfig returns an optimized configuration for different model sizes.
func OptimizedVLLMConfig(modelSize string) map[string]any {
	configs := map[string]map[string]any{
		"7b": {
			"reasoning_depth":      2,
			"max_thinking_tokens":  1024,
			"thinking_temperature": 0.4,
			"parallel_stages":      false,
			"max_stage_time":       30,
		},
		"13b": {
			"reasoning_depth":      3,
			"max_thinking_tokens":  1536,
			"thinking_temperature": 0.35,
			"parallel_stages":      false,
			"max_stage_time":       45,
		},
		"30b": {
			"reasoning_depth":      4,
			"max_thinking_tokens":  2048,
			"thinking_temperature": 0.3,
			"parallel_stages":      true,
			"max_stage_time":       60,
		},
		"70b": {
			"reasoning_depth":      5,
			"max_thinking_tokens":  3072,
			"thinking_temperature": 0.25,
			"parallel_stages":      true,
			"max_stage_time":       90,
		},
	}

	if config, ok := configs[modelSize]; ok {
		return config
	}
	
	// Default configuration
	return configs["13b"]
}

// ExampleUsage demonstrates how to use the sequential thinking agent.
func ExampleUsage() {
	ctx := context.Background()

	// Create agent with vLLM
	agent, err := CreateSequentialThinkingWithVLLM(ctx)
	if err != nil {
		slog.Error("Failed to create agent", "error", err)
		return
	}

	// Example 1: Problem solving
	problemSolving := `
	I have a web application that's experiencing intermittent 503 errors during peak traffic.
	The errors seem to correlate with database connection timeouts.
	Help me diagnose and solve this issue.
	`

	result1, err := agent.Think(ctx, problemSolving)
	if err != nil {
		slog.Error("Failed to process problem", "error", err)
		return
	}

	fmt.Printf("Problem Solving Result:\n%s\n\n", result1.FinalOutput)
	fmt.Printf("Reasoning Chain:\n%s\n", result1.Reasoning)
	fmt.Printf("Tokens Used: %d\n", result1.TokensUsed)
	fmt.Printf("Time Taken: %s\n\n", result1.TotalTime)

	// Example 2: Code analysis
	codeAnalysis := `
	Analyze this function for potential improvements:
	
	func processData(items []Item) []Result {
		var results []Result
		for _, item := range items {
			if item.Valid {
				result := Result{
					ID: item.ID,
					Value: item.Process(),
				}
				results = append(results, result)
			}
		}
		return results
	}
	`

	result2, err := agent.Think(ctx, codeAnalysis)
	if err != nil {
		slog.Error("Failed to analyze code", "error", err)
		return
	}

	fmt.Printf("Code Analysis Result:\n%s\n\n", result2.FinalOutput)

	// Example 3: Architecture design
	architectureDesign := `
	Design a microservices architecture for an e-commerce platform that needs to handle:
	- 100k concurrent users
	- Real-time inventory updates
	- Multiple payment providers
	- International shipping calculations
	`

	result3, err := agent.Think(ctx, architectureDesign)
	if err != nil {
		slog.Error("Failed to design architecture", "error", err)
		return
	}

	fmt.Printf("Architecture Design:\n%s\n\n", result3.FinalOutput)

	// Print metrics
	metrics := agent.GetMetrics()
	fmt.Printf("\nAgent Metrics:\n")
	for key, value := range metrics {
		fmt.Printf("  %s: %v\n", key, value)
	}
}

// IntegrateWithCrushSession shows how to integrate with Crush's session system.
func IntegrateWithCrushSession(
	ctx context.Context,
	sessionID string,
	messages message.Service,
) error {
	// Create the agent
	agent, err := CreateSequentialThinkingWithVLLM(ctx)
	if err != nil {
		return err
	}

	// Wrap as a Crush service
	service := agent.AsService()

	// Process a user query
	userQuery := "Explain the trade-offs between microservices and monolithic architectures"
	
	events, err := service.Run(ctx, sessionID, userQuery)
	if err != nil {
		return fmt.Errorf("failed to run agent: %w", err)
	}

	// Process events
	for event := range events {
		switch event.Type {
		case AgentEventTypeResponse:
			// Save the message
			_, err := messages.Create(ctx, sessionID, message.CreateMessageParams{
				Role:     event.Message.Role,
				Parts:    event.Message.Parts,
				Model:    service.Model().ID,
				Provider: "vllm",
			})
			if err != nil {
				return fmt.Errorf("failed to save message: %w", err)
			}
			
			slog.Info("Agent response saved",
				"session_id", sessionID,
				"content_length", len(event.Message.Content().String()),
			)

		case AgentEventTypeError:
			return fmt.Errorf("agent error: %w", event.Error)
		}
	}

	return nil
}

// BenchmarkConfigurations provides configurations for different use cases.
func BenchmarkConfigurations() map[string]*SequentialThinkingConfig {
	return map[string]*SequentialThinkingConfig{
		"fast": {
			ReasoningDepth:      1,
			ParallelStages:      false,
			MaxStageTime:        10,
			EnableReflection:    false,
			ThinkingTemperature: 0.5,
			MaxThinkingTokens:   512,
			VerboseThinking:     false,
		},
		"balanced": {
			ReasoningDepth:      3,
			ParallelStages:      false,
			MaxStageTime:        30,
			EnableReflection:    false,
			ThinkingTemperature: 0.35,
			MaxThinkingTokens:   1536,
			VerboseThinking:     false,
		},
		"thorough": {
			ReasoningDepth:      5,
			ParallelStages:      true,
			MaxStageTime:        60,
			EnableReflection:    true,
			ThinkingTemperature: 0.25,
			MaxThinkingTokens:   3072,
			VerboseThinking:     true,
		},
		"debug": {
			ReasoningDepth:      3,
			ParallelStages:      false,
			MaxStageTime:        120,
			EnableReflection:    true,
			ThinkingTemperature: 0.3,
			MaxThinkingTokens:   2048,
			VerboseThinking:     true,
		},
	}
}

// PerformanceOptimizationTips provides tips for optimizing the agent with vLLM.
func PerformanceOptimizationTips() []string {
	return []string{
		"1. Use quantization (AWQ/GPTQ) for larger models to reduce memory usage",
		"2. Enable KV cache in vLLM for better performance on sequential stages",
		"3. Adjust gpu_memory_utilization based on available VRAM",
		"4. Use tensor parallelism for models larger than 30B parameters",
		"5. Disable parallel_stages for better cache utilization on single GPU",
		"6. Lower thinking_temperature for more deterministic reasoning",
		"7. Adjust max_thinking_tokens based on model size and VRAM",
		"8. Use continuous batching in vLLM for handling multiple requests",
		"9. Enable PagedAttention in vLLM for efficient memory management",
		"10. Monitor GPU memory usage and adjust max_model_len accordingly",
	}
}

// ModelRecommendations suggests models for different use cases.
func ModelRecommendations() map[string]string {
	return map[string]string{
		"codellama-7b":        "Fast code analysis and generation",
		"llama-2-13b-chat":    "Balanced general reasoning",
		"mixtral-8x7b":        "High-quality reasoning with MoE efficiency",
		"yi-34b":              "Deep reasoning with large context",
		"deepseek-coder-33b":  "Advanced code understanding",
		"qwen-72b":            "Comprehensive analysis (requires multiple GPUs)",
		"mistral-7b-instruct": "Efficient reasoning for edge deployment",
	}
}