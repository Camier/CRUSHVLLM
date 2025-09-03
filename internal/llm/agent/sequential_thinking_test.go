package agent

import (
	"context"
	"testing"
	"time"

	"github.com/charmbracelet/crush/internal/config"
	"github.com/charmbracelet/crush/internal/message"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestSequentialThinkingAgent_InitializeStages(t *testing.T) {
	tests := []struct {
		name           string
		reasoningDepth int
		expectedStages []string
	}{
		{
			name:           "Minimal depth",
			reasoningDepth: 1,
			expectedStages: []string{
				"problem_decomposition",
				"component_analysis",
				"synthesis",
			},
		},
		{
			name:           "Medium depth",
			reasoningDepth: 3,
			expectedStages: []string{
				"problem_decomposition",
				"component_analysis",
				"hypothesis_generation",
				"hypothesis_evaluation",
				"synthesis",
			},
		},
		{
			name:           "Maximum depth with reflection",
			reasoningDepth: 5,
			expectedStages: []string{
				"problem_decomposition",
				"component_analysis",
				"hypothesis_generation",
				"hypothesis_evaluation",
				"validation_planning",
				"synthesis",
				"reflection",
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			agent := &SequentialThinkingAgent{
				config: &SequentialThinkingConfig{
					ReasoningDepth:   tt.reasoningDepth,
					EnableReflection: tt.reasoningDepth >= 5,
				},
			}

			stages := agent.initializeStages(tt.reasoningDepth)
			
			assert.Equal(t, len(tt.expectedStages), len(stages))
			
			for i, expectedName := range tt.expectedStages {
				assert.Equal(t, expectedName, stages[i].Name)
			}
		})
	}
}

func TestSequentialThinkingAgent_DependencyResolution(t *testing.T) {
	agent := &SequentialThinkingAgent{
		config: &SequentialThinkingConfig{
			ReasoningDepth: 3,
		},
	}

	stages := agent.initializeStages(3)
	
	// Check dependencies are correct
	stageMap := make(map[string]ThinkingStage)
	for _, stage := range stages {
		stageMap[stage.Name] = stage
	}
	
	// problem_decomposition should have no dependencies
	assert.Empty(t, stageMap["problem_decomposition"].DependsOn)
	
	// component_analysis should depend on problem_decomposition
	assert.Contains(t, stageMap["component_analysis"].DependsOn, "problem_decomposition")
	
	// synthesis should depend on all previous stages except reflection
	synthDeps := stageMap["synthesis"].DependsOn
	assert.Contains(t, synthDeps, "problem_decomposition")
	assert.Contains(t, synthDeps, "component_analysis")
	assert.Contains(t, synthDeps, "hypothesis_generation")
	assert.Contains(t, synthDeps, "hypothesis_evaluation")
}

func TestSequentialThinkingAgent_StageExecution(t *testing.T) {
	// Mock stage result
	mockResult := StageResult{
		Stage:     "test_stage",
		Content:   "Test content",
		Reasoning: "Test reasoning",
		StartTime: time.Now(),
		EndTime:   time.Now().Add(100 * time.Millisecond),
		Tokens:    50,
		Metadata: map[string]any{
			"duration_ms": int64(100),
		},
	}

	agent := &SequentialThinkingAgent{
		config: &SequentialThinkingConfig{
			ReasoningDepth:   2,
			MaxStageTime:     10,
			VerboseThinking:  true,
		},
	}

	// Test stage with mock handler
	stage := ThinkingStage{
		Name:        "test_stage",
		Description: "Test stage",
		Required:    true,
		Handler: func(ctx context.Context, input string, prev map[string]StageResult) (StageResult, error) {
			return mockResult, nil
		},
	}

	results := make(map[string]StageResult)
	err := agent.executeStage(context.Background(), "test input", stage, results)
	
	require.NoError(t, err)
	assert.Equal(t, mockResult, results["test_stage"])
}

func TestSequentialThinkingAgent_BuildReasoningChain(t *testing.T) {
	agent := &SequentialThinkingAgent{}

	results := map[string]StageResult{
		"problem_decomposition": {
			Stage:     "problem_decomposition",
			Content:   "Decomposed the problem",
			Reasoning: "Breaking down into parts",
			Metadata: map[string]any{
				"duration_ms": int64(150),
			},
		},
		"component_analysis": {
			Stage:     "component_analysis",
			Content:   "Analyzed components",
			Reasoning: "Looking at each part",
			Metadata: map[string]any{
				"duration_ms": int64(200),
			},
		},
		"synthesis": {
			Stage:   "synthesis",
			Content: "Final synthesis",
			Metadata: map[string]any{
				"duration_ms": int64(100),
			},
		},
	}

	order := []string{"problem_decomposition", "component_analysis", "synthesis"}
	
	chain := agent.buildReasoningChain(results, order)
	
	assert.Contains(t, chain, "Sequential Thinking Process")
	assert.Contains(t, chain, "Problem Decomposition")
	assert.Contains(t, chain, "Component Analysis")
	assert.Contains(t, chain, "Synthesis")
	assert.Contains(t, chain, "Breaking down into parts")
	assert.Contains(t, chain, "Looking at each part")
	assert.Contains(t, chain, "Final synthesis")
}

func TestSequentialThinkingAgent_Metrics(t *testing.T) {
	agent := &SequentialThinkingAgent{
		config: &SequentialThinkingConfig{
			ReasoningDepth: 3,
		},
		metrics: &ThinkingMetrics{},
	}

	// Simulate some results
	results := map[string]StageResult{
		"stage1": {
			StartTime: time.Now(),
			EndTime:   time.Now().Add(100 * time.Millisecond),
			Tokens:    100,
			Content:   "content",
		},
		"stage2": {
			StartTime: time.Now(),
			EndTime:   time.Now().Add(150 * time.Millisecond),
			Tokens:    150,
			Content:   "content",
		},
	}

	agent.updateMetrics(results, 250*time.Millisecond)
	
	metrics := agent.GetMetrics()
	
	assert.Equal(t, int64(1), metrics["total_requests"])
	assert.Equal(t, int64(2), metrics["total_stages"])
	assert.Equal(t, int64(250), metrics["tokens_used"])
	assert.Equal(t, 1.0, metrics["stage_success_rate"])
}

func TestSequentialThinkingConfig_Defaults(t *testing.T) {
	cfg := &SequentialThinkingConfig{
		ReasoningDepth:      3,
		ParallelStages:      true,
		MaxStageTime:        30,
		EnableReflection:    true,
		ThinkingTemperature: 0.3,
		MaxThinkingTokens:   1024,
		VerboseThinking:     false,
	}

	assert.Equal(t, 3, cfg.ReasoningDepth)
	assert.True(t, cfg.ParallelStages)
	assert.Equal(t, 30, cfg.MaxStageTime)
	assert.True(t, cfg.EnableReflection)
	assert.Equal(t, float32(0.3), cfg.ThinkingTemperature)
	assert.Equal(t, 1024, cfg.MaxThinkingTokens)
	assert.False(t, cfg.VerboseThinking)
}

func TestSequentialThinkingService_Integration(t *testing.T) {
	// This would require a mock provider
	t.Skip("Integration test requires mock provider setup")
	
	// Example of how the service would be used:
	/*
	providerCfg := &config.ProviderConfig{
		ID:      "vllm",
		Type:    "vllm",
		BaseURL: "http://localhost:8000/v1",
	}
	
	agentCfg := config.Agent{
		ID:    "sequential_thinker",
		Name:  "Sequential Thinking Agent",
		Model: config.SelectedModelTypeLarge,
		ExtraConfig: map[string]any{
			"reasoning_depth": 3,
			"parallel_stages": true,
		},
	}
	
	agent, err := NewSequentialThinkingAgent(context.Background(), agentCfg, providerCfg)
	require.NoError(t, err)
	
	service := agent.AsService()
	
	events, err := service.Run(context.Background(), "session-123", "Explain how photosynthesis works")
	require.NoError(t, err)
	
	for event := range events {
		if event.Type == AgentEventTypeResponse {
			assert.NotEmpty(t, event.Message.Content())
			assert.Contains(t, event.Message.Content(), "photosynthesis")
		}
	}
	*/
}

func TestCalculateTotalTokens(t *testing.T) {
	agent := &SequentialThinkingAgent{}
	
	results := map[string]StageResult{
		"stage1": {Tokens: 100},
		"stage2": {Tokens: 200},
		"stage3": {Tokens: 150},
	}
	
	total := agent.calculateTotalTokens(results)
	assert.Equal(t, 450, total)
}

func TestGetSynthesisDependencies(t *testing.T) {
	agent := &SequentialThinkingAgent{}
	
	stages := []ThinkingStage{
		{Name: "problem_decomposition"},
		{Name: "component_analysis"},
		{Name: "hypothesis_generation"},
		{Name: "synthesis"},
		{Name: "reflection"},
	}
	
	deps := agent.getSynthesisDependencies(stages)
	
	assert.Contains(t, deps, "problem_decomposition")
	assert.Contains(t, deps, "component_analysis")
	assert.Contains(t, deps, "hypothesis_generation")
	assert.NotContains(t, deps, "synthesis")
	assert.NotContains(t, deps, "reflection")
	assert.Equal(t, 3, len(deps))
}

func TestBuildSequentialThinkingPrompt(t *testing.T) {
	tests := []struct {
		name     string
		depth    int
		expected string
	}{
		{
			name:     "Quick reasoning",
			depth:    1,
			expected: "quick reasoning process",
		},
		{
			name:     "Standard reasoning",
			depth:    3,
			expected: "standard reasoning process",
		},
		{
			name:     "Deep reasoning",
			depth:    5,
			expected: "deep reasoning process",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			cfg := &SequentialThinkingConfig{
				ReasoningDepth:      tt.depth,
				ThinkingTemperature: 0.3,
				MaxThinkingTokens:   1024,
			}
			
			prompt := buildSequentialThinkingPrompt(cfg)
			
			assert.Contains(t, prompt, tt.expected)
			assert.Contains(t, prompt, "Sequential Thinking Agent")
			assert.Contains(t, prompt, "Temperature: 0.30")
			assert.Contains(t, prompt, "Max tokens per stage: 1024")
		})
	}
}