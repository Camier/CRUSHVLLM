//go:build ignore
// +build ignore

package agent

import (
	"context"
	"errors"
	"fmt"
	"log/slog"
	"strings"
	"sync"
	"time"

	"github.com/charmbracelet/catwalk/pkg/catwalk"
	"github.com/charmbracelet/crush/internal/config"
	"github.com/charmbracelet/crush/internal/llm/provider"
	"github.com/charmbracelet/crush/internal/llm/tools"
	"github.com/charmbracelet/crush/internal/message"
	"github.com/charmbracelet/crush/internal/pubsub"
)

// SequentialThinkingAgent implements structured, step-by-step reasoning
// optimized for local inference with vLLM provider.
type SequentialThinkingAgent struct {
	*pubsub.Broker[AgentEvent]
	
	provider   provider.Provider
	providerID string
	config     *SequentialThinkingConfig
	
	// Thinking stages
	stages []ThinkingStage
	
	// Metrics
	mu      sync.RWMutex
	metrics *ThinkingMetrics
}

// SequentialThinkingConfig holds configuration for the sequential thinking agent.
type SequentialThinkingConfig struct {
	// Reasoning depth (1-5, where 5 is most thorough)
	ReasoningDepth int `json:"reasoning_depth"`
	
	// Enable parallel stage execution where possible
	ParallelStages bool `json:"parallel_stages"`
	
	// Maximum thinking time per stage (seconds)
	MaxStageTime int `json:"max_stage_time"`
	
	// Enable reflection after each stage
	EnableReflection bool `json:"enable_reflection"`
	
	// Temperature for thinking stages (lower = more focused)
	ThinkingTemperature float32 `json:"thinking_temperature"`
	
	// Maximum tokens per thinking stage
	MaxThinkingTokens int `json:"max_thinking_tokens"`
	
	// Enable verbose thinking output
	VerboseThinking bool `json:"verbose_thinking"`
}

// ThinkingStage represents a step in the sequential thinking process.
type ThinkingStage struct {
	Name        string
	Description string
	Prompt      string
	Required    bool
	DependsOn   []string // Names of stages this depends on
	Handler     StageHandler
}

// StageHandler processes a thinking stage.
type StageHandler func(ctx context.Context, input string, previousResults map[string]StageResult) (StageResult, error)

// StageResult holds the output of a thinking stage.
type StageResult struct {
	Stage     string
	Content   string
	Reasoning string
	Metadata  map[string]any
	StartTime time.Time
	EndTime   time.Time
	Tokens    int
}

// ThinkingMetrics tracks agent performance.
type ThinkingMetrics struct {
	TotalRequests     int64
	TotalStages       int64
	AverageStageTime  time.Duration
	TotalThinkingTime time.Duration
	StageSuccessRate  float64
	ReflectionCount   int64
	TokensUsed        int64
}

// NewSequentialThinkingAgent creates a new sequential thinking agent.
func NewSequentialThinkingAgent(
	ctx context.Context,
	agentCfg config.Agent,
	providerCfg *config.ProviderConfig,
) (*SequentialThinkingAgent, error) {
	// Default configuration
	cfg := &SequentialThinkingConfig{
		ReasoningDepth:      3,
		ParallelStages:      true,
		MaxStageTime:        30,
		EnableReflection:    true,
		ThinkingTemperature: 0.3, // Lower temperature for more focused reasoning
		MaxThinkingTokens:   1024,
		VerboseThinking:     false,
	}
	
	// Override with agent configuration if provided
	if agentCfg.ExtraConfig != nil {
		if depth, ok := agentCfg.ExtraConfig["reasoning_depth"].(int); ok {
			cfg.ReasoningDepth = depth
		}
		if parallel, ok := agentCfg.ExtraConfig["parallel_stages"].(bool); ok {
			cfg.ParallelStages = parallel
		}
		if verbose, ok := agentCfg.ExtraConfig["verbose_thinking"].(bool); ok {
			cfg.VerboseThinking = verbose
		}
	}
	
	// Create provider with sequential thinking system prompt
	systemPrompt := buildSequentialThinkingPrompt(cfg)
	opts := []provider.ProviderClientOption{
		provider.WithModel(agentCfg.Model),
		provider.WithSystemMessage(systemPrompt),
		provider.WithMaxTokens(int64(cfg.MaxThinkingTokens)),
	}
	
	agentProvider, err := provider.NewProvider(*providerCfg, opts...)
	if err != nil {
		return nil, fmt.Errorf("failed to create provider: %w", err)
	}
	
	agent := &SequentialThinkingAgent{
		Broker:     pubsub.NewBroker[AgentEvent](),
		provider:   agentProvider,
		providerID: string(providerCfg.ID),
		config:     cfg,
		metrics:    &ThinkingMetrics{},
	}
	
	// Initialize thinking stages based on reasoning depth
	agent.stages = agent.initializeStages(cfg.ReasoningDepth)
	
	return agent, nil
}

// initializeStages creates thinking stages based on reasoning depth.
func (a *SequentialThinkingAgent) initializeStages(depth int) []ThinkingStage {
	stages := []ThinkingStage{
		{
			Name:        "problem_decomposition",
			Description: "Break down the problem into components",
			Prompt:      "Analyze and decompose this problem into its core components and requirements:",
			Required:    true,
			Handler:     a.handleProblemDecomposition,
		},
		{
			Name:        "component_analysis",
			Description: "Analyze each component in detail",
			Prompt:      "For each component identified, analyze its properties, constraints, and relationships:",
			Required:    true,
			DependsOn:   []string{"problem_decomposition"},
			Handler:     a.handleComponentAnalysis,
		},
	}
	
	if depth >= 2 {
		stages = append(stages, ThinkingStage{
			Name:        "hypothesis_generation",
			Description: "Generate potential solutions or approaches",
			Prompt:      "Based on the analysis, generate multiple hypotheses or solution approaches:",
			Required:    true,
			DependsOn:   []string{"component_analysis"},
			Handler:     a.handleHypothesisGeneration,
		})
	}
	
	if depth >= 3 {
		stages = append(stages, ThinkingStage{
			Name:        "hypothesis_evaluation",
			Description: "Evaluate and rank hypotheses",
			Prompt:      "Evaluate each hypothesis for feasibility, effectiveness, and potential issues:",
			Required:    true,
			DependsOn:   []string{"hypothesis_generation"},
			Handler:     a.handleHypothesisEvaluation,
		})
	}
	
	if depth >= 4 {
		stages = append(stages, ThinkingStage{
			Name:        "validation_planning",
			Description: "Plan validation steps",
			Prompt:      "Design validation steps to verify the chosen approach:",
			Required:    false,
			DependsOn:   []string{"hypothesis_evaluation"},
			Handler:     a.handleValidationPlanning,
		})
	}
	
	// Always include synthesis
	stages = append(stages, ThinkingStage{
		Name:        "synthesis",
		Description: "Synthesize findings into a coherent response",
		Prompt:      "Synthesize all findings into a comprehensive solution:",
		Required:    true,
		DependsOn:   a.getSynthesisDependencies(stages),
		Handler:     a.handleSynthesis,
	})
	
	if depth >= 5 || a.config.EnableReflection {
		stages = append(stages, ThinkingStage{
			Name:        "reflection",
			Description: "Reflect on the reasoning process",
			Prompt:      "Reflect on the reasoning process and identify any gaps or improvements:",
			Required:    false,
			DependsOn:   []string{"synthesis"},
			Handler:     a.handleReflection,
		})
	}
	
	return stages
}

// getSynthesisDependencies returns the stages that synthesis depends on.
func (a *SequentialThinkingAgent) getSynthesisDependencies(stages []ThinkingStage) []string {
	deps := []string{}
	for _, stage := range stages {
		if stage.Name != "synthesis" && stage.Name != "reflection" {
			deps = append(deps, stage.Name)
		}
	}
	return deps
}

// Think performs sequential thinking on the input.
func (a *SequentialThinkingAgent) Think(ctx context.Context, input string, attachments ...message.Attachment) (*ThinkingResult, error) {
	startTime := time.Now()
	
	// Update metrics
	a.mu.Lock()
	a.metrics.TotalRequests++
	a.mu.Unlock()
	
	// Execute stages
	results := make(map[string]StageResult)
	var stageOrder []string
	
	// Determine execution order based on dependencies
	executed := make(map[string]bool)
	
	for len(executed) < len(a.stages) {
		// Find stages that can be executed
		readyStages := []ThinkingStage{}
		for _, stage := range a.stages {
			if executed[stage.Name] {
				continue
			}
			
			// Check if all dependencies are satisfied
			ready := true
			for _, dep := range stage.DependsOn {
				if !executed[dep] {
					ready = false
					break
				}
			}
			
			if ready {
				readyStages = append(readyStages, stage)
			}
		}
		
		if len(readyStages) == 0 {
			return nil, errors.New("circular dependency detected in thinking stages")
		}
		
		// Execute ready stages (potentially in parallel)
		if a.config.ParallelStages && len(readyStages) > 1 {
			a.executeStagesParallel(ctx, input, readyStages, results, executed)
		} else {
			for _, stage := range readyStages {
				if err := a.executeStage(ctx, input, stage, results); err != nil {
					if stage.Required {
						return nil, fmt.Errorf("required stage %s failed: %w", stage.Name, err)
					}
					slog.Warn("Optional stage failed", "stage", stage.Name, "error", err)
				}
				executed[stage.Name] = true
				stageOrder = append(stageOrder, stage.Name)
			}
		}
	}
	
	// Update metrics
	a.updateMetrics(results, time.Since(startTime))
	
	// Build final result
	return &ThinkingResult{
		Input:       input,
		Stages:      results,
		StageOrder:  stageOrder,
		TotalTime:   time.Since(startTime),
		TokensUsed:  a.calculateTotalTokens(results),
		FinalOutput: results["synthesis"].Content,
		Reasoning:   a.buildReasoningChain(results, stageOrder),
	}, nil
}

// executeStage executes a single thinking stage.
func (a *SequentialThinkingAgent) executeStage(
	ctx context.Context,
	input string,
	stage ThinkingStage,
	results map[string]StageResult,
) error {
	// Set timeout for stage
	stageCtx, cancel := context.WithTimeout(ctx, time.Duration(a.config.MaxStageTime)*time.Second)
	defer cancel()
	
	if a.config.VerboseThinking {
		slog.Info("Executing thinking stage", "stage", stage.Name, "description", stage.Description)
	}
	
	// Execute stage handler
	result, err := stage.Handler(stageCtx, input, results)
	if err != nil {
		return err
	}
	
	// Store result
	results[stage.Name] = result
	
	// Publish progress event
	a.Publish(pubsub.CreatedEvent, AgentEvent{
		Type:     AgentEventTypeResponse,
		Progress: fmt.Sprintf("Completed stage: %s", stage.Name),
	})
	
	return nil
}

// executeStagesParallel executes multiple stages in parallel.
func (a *SequentialThinkingAgent) executeStagesParallel(
	ctx context.Context,
	input string,
	stages []ThinkingStage,
	results map[string]StageResult,
	executed map[string]bool,
) {
	var wg sync.WaitGroup
	mu := sync.Mutex{}
	
	for _, stage := range stages {
		wg.Add(1)
		go func(s ThinkingStage) {
			defer wg.Done()
			
			if err := a.executeStage(ctx, input, s, results); err != nil {
				if s.Required {
					slog.Error("Required stage failed", "stage", s.Name, "error", err)
				} else {
					slog.Warn("Optional stage failed", "stage", s.Name, "error", err)
				}
			}
			
			mu.Lock()
			executed[s.Name] = true
			mu.Unlock()
		}(stage)
	}
	
	wg.Wait()
}

// Stage handlers

func (a *SequentialThinkingAgent) handleProblemDecomposition(ctx context.Context, input string, _ map[string]StageResult) (StageResult, error) {
	prompt := fmt.Sprintf(`%s

Input: %s

Provide a structured decomposition with:
1. Core problem statement
2. Key components/subproblems
3. Constraints and requirements
4. Success criteria`, a.stages[0].Prompt, input)
	
	return a.executeThinkingStep(ctx, "problem_decomposition", prompt)
}

func (a *SequentialThinkingAgent) handleComponentAnalysis(ctx context.Context, input string, prev map[string]StageResult) (StageResult, error) {
	decomp := prev["problem_decomposition"]
	
	prompt := fmt.Sprintf(`%s

Previous decomposition:
%s

Input: %s

Analyze each component with:
1. Properties and characteristics
2. Dependencies and relationships
3. Potential challenges
4. Required resources or knowledge`, a.stages[1].Prompt, decomp.Content, input)
	
	return a.executeThinkingStep(ctx, "component_analysis", prompt)
}

func (a *SequentialThinkingAgent) handleHypothesisGeneration(ctx context.Context, input string, prev map[string]StageResult) (StageResult, error) {
	analysis := prev["component_analysis"]
	
	prompt := fmt.Sprintf(`%s

Based on analysis:
%s

Input: %s

Generate 3-5 distinct hypotheses or solution approaches with:
1. Core approach description
2. Key assumptions
3. Expected outcomes
4. Potential risks`, a.stages[2].Prompt, analysis.Content, input)
	
	return a.executeThinkingStep(ctx, "hypothesis_generation", prompt)
}

func (a *SequentialThinkingAgent) handleHypothesisEvaluation(ctx context.Context, input string, prev map[string]StageResult) (StageResult, error) {
	hypotheses := prev["hypothesis_generation"]
	
	prompt := fmt.Sprintf(`%s

Hypotheses to evaluate:
%s

Input: %s

Evaluate each hypothesis on:
1. Feasibility (1-10)
2. Effectiveness (1-10)
3. Risk level (Low/Medium/High)
4. Implementation complexity
5. Recommendation with justification`, a.stages[3].Prompt, hypotheses.Content, input)
	
	return a.executeThinkingStep(ctx, "hypothesis_evaluation", prompt)
}

func (a *SequentialThinkingAgent) handleValidationPlanning(ctx context.Context, input string, prev map[string]StageResult) (StageResult, error) {
	evaluation := prev["hypothesis_evaluation"]
	
	prompt := fmt.Sprintf(`%s

Based on evaluation:
%s

Input: %s

Design validation steps:
1. Test cases or experiments
2. Success metrics
3. Failure indicators
4. Contingency plans`, a.stages[4].Prompt, evaluation.Content, input)
	
	return a.executeThinkingStep(ctx, "validation_planning", prompt)
}

func (a *SequentialThinkingAgent) handleSynthesis(ctx context.Context, input string, prev map[string]StageResult) (StageResult, error) {
	// Collect all previous results
	var previousWork strings.Builder
	for stage, result := range prev {
		if stage != "synthesis" && stage != "reflection" {
			previousWork.WriteString(fmt.Sprintf("\n[%s]:\n%s\n", stage, result.Content))
		}
	}
	
	prompt := fmt.Sprintf(`Based on all previous analysis:
%s

Original input: %s

Synthesize a comprehensive solution that:
1. Addresses all identified components
2. Implements the best approach
3. Includes validation steps
4. Provides clear, actionable steps
5. Highlights key insights and recommendations`, previousWork.String(), input)
	
	return a.executeThinkingStep(ctx, "synthesis", prompt)
}

func (a *SequentialThinkingAgent) handleReflection(ctx context.Context, input string, prev map[string]StageResult) (StageResult, error) {
	synthesis := prev["synthesis"]
	
	prompt := fmt.Sprintf(`Reflect on the reasoning process and solution:

Solution:
%s

Original input: %s

Consider:
1. Are there any gaps in the reasoning?
2. What assumptions were made?
3. What alternative approaches were not explored?
4. How confident are we in the solution?
5. What could be improved in future analysis?`, synthesis.Content, input)
	
	result, err := a.executeThinkingStep(ctx, "reflection", prompt)
	if err == nil {
		a.mu.Lock()
		a.metrics.ReflectionCount++
		a.mu.Unlock()
	}
	
	return result, err
}

// executeThinkingStep sends a prompt to the provider and returns the result.
func (a *SequentialThinkingAgent) executeThinkingStep(ctx context.Context, stageName, prompt string) (StageResult, error) {
	startTime := time.Now()
	
	// Create message for provider
	msgs := []message.Message{
		{
			Role: message.User,
			Parts: []message.ContentPart{
				message.TextContent{Text: prompt},
			},
		},
	}
	
	// Use lower temperature for thinking
	response := a.provider.StreamResponse(ctx, msgs, nil)
	
	var content strings.Builder
	var reasoning strings.Builder
	var tokens int
	inReasoning := false
	
	for event := range response {
		if event.Error != nil {
			return StageResult{}, event.Error
		}
		
		if event.Type == provider.EventThinkingDelta {
			reasoning.WriteString(event.Thinking)
			inReasoning = true
		} else if event.Type == provider.EventContentDelta {
			inReasoning = false
			content.WriteString(event.Content)
		} else if event.Type == provider.EventComplete && event.Response != nil {
			tokens = int(event.Response.Usage.OutputTokens)
		}
	}
	
	return StageResult{
		Stage:     stageName,
		Content:   content.String(),
		Reasoning: reasoning.String(),
		StartTime: startTime,
		EndTime:   time.Now(),
		Tokens:    tokens,
		Metadata: map[string]any{
			"duration_ms": time.Since(startTime).Milliseconds(),
		},
	}, nil
}

// buildReasoningChain builds a narrative of the reasoning process.
func (a *SequentialThinkingAgent) buildReasoningChain(results map[string]StageResult, order []string) string {
	var chain strings.Builder
	
	chain.WriteString("## Sequential Thinking Process\n\n")
	
	for _, stageName := range order {
		result := results[stageName]
		chain.WriteString(fmt.Sprintf("### %s\n", strings.Title(strings.ReplaceAll(stageName, "_", " "))))
		
		if result.Reasoning != "" {
			chain.WriteString("**Reasoning:**\n")
			chain.WriteString(result.Reasoning)
			chain.WriteString("\n\n")
		}
		
		chain.WriteString("**Result:**\n")
		chain.WriteString(result.Content)
		chain.WriteString("\n\n")
		
		if result.Metadata != nil {
			if duration, ok := result.Metadata["duration_ms"].(int64); ok {
				chain.WriteString(fmt.Sprintf("*Stage completed in %dms*\n\n", duration))
			}
		}
	}
	
	return chain.String()
}

// calculateTotalTokens sums tokens across all stages.
func (a *SequentialThinkingAgent) calculateTotalTokens(results map[string]StageResult) int {
	total := 0
	for _, result := range results {
		total += result.Tokens
	}
	return total
}

// updateMetrics updates agent metrics.
func (a *SequentialThinkingAgent) updateMetrics(results map[string]StageResult, totalTime time.Duration) {
	a.mu.Lock()
	defer a.mu.Unlock()
	
	a.metrics.TotalStages += int64(len(results))
	a.metrics.TotalThinkingTime += totalTime
	
	// Calculate average stage time
	var totalStageTime time.Duration
	successCount := 0
	for _, result := range results {
		stageDuration := result.EndTime.Sub(result.StartTime)
		totalStageTime += stageDuration
		if result.Content != "" {
			successCount++
		}
		a.metrics.TokensUsed += int64(result.Tokens)
	}
	
	if len(results) > 0 {
		a.metrics.AverageStageTime = totalStageTime / time.Duration(len(results))
		a.metrics.StageSuccessRate = float64(successCount) / float64(len(results))
	}
}

// GetMetrics returns current metrics.
func (a *SequentialThinkingAgent) GetMetrics() map[string]any {
	a.mu.RLock()
	defer a.mu.RUnlock()
	
	avgThinkingTime := time.Duration(0)
	if a.metrics.TotalRequests > 0 {
		avgThinkingTime = a.metrics.TotalThinkingTime / time.Duration(a.metrics.TotalRequests)
	}
	
	return map[string]any{
		"total_requests":      a.metrics.TotalRequests,
		"total_stages":        a.metrics.TotalStages,
		"avg_stage_time_ms":   a.metrics.AverageStageTime.Milliseconds(),
		"avg_thinking_time_ms": avgThinkingTime.Milliseconds(),
		"stage_success_rate":  a.metrics.StageSuccessRate,
		"reflection_count":    a.metrics.ReflectionCount,
		"tokens_used":         a.metrics.TokensUsed,
		"reasoning_depth":     a.config.ReasoningDepth,
		"parallel_stages":     a.config.ParallelStages,
	}
}

// Model returns the model being used.
func (a *SequentialThinkingAgent) Model() catwalk.Model {
	return a.provider.Model()
}

// ThinkingResult contains the complete result of sequential thinking.
type ThinkingResult struct {
	Input       string
	Stages      map[string]StageResult
	StageOrder  []string
	TotalTime   time.Duration
	TokensUsed  int
	FinalOutput string
	Reasoning   string
}

// buildSequentialThinkingPrompt creates the system prompt for sequential thinking.
func buildSequentialThinkingPrompt(cfg *SequentialThinkingConfig) string {
	depth := "standard"
	if cfg.ReasoningDepth >= 4 {
		depth = "deep"
	} else if cfg.ReasoningDepth <= 2 {
		depth = "quick"
	}
	
	return fmt.Sprintf(`You are a Sequential Thinking Agent that processes problems through structured, step-by-step reasoning.

## Thinking Framework

You will analyze problems using a %s reasoning process with the following principles:

1. **Decomposition**: Break complex problems into manageable components
2. **Analysis**: Examine each component systematically
3. **Hypothesis**: Generate multiple solution approaches
4. **Evaluation**: Assess approaches based on criteria
5. **Synthesis**: Combine insights into coherent solutions
6. **Reflection**: Consider gaps and improvements

## Reasoning Style

- Be systematic and thorough
- Show your work explicitly
- Question assumptions
- Consider edge cases
- Prioritize clarity over brevity
- Use structured formats when helpful

## Temperature: %.2f (focused reasoning)
## Max tokens per stage: %d

When thinking through problems:
- Start with the big picture
- Progressively increase detail
- Connect insights across stages
- Build on previous findings
- Validate conclusions

Your goal is to provide well-reasoned, comprehensive solutions through sequential analysis.`,
		depth,
		cfg.ThinkingTemperature,
		cfg.MaxThinkingTokens,
	)
}

// Integration with Crush's agent system

// AsService wraps the sequential thinking agent as a Crush Service.
func (a *SequentialThinkingAgent) AsService() Service {
	return &sequentialThinkingService{
		agent: a,
	}
}

type sequentialThinkingService struct {
	agent *SequentialThinkingAgent
	activeRequests map[string]context.CancelFunc
	mu sync.Mutex
}

func (s *sequentialThinkingService) Model() catwalk.Model {
	return s.agent.Model()
}

func (s *sequentialThinkingService) Run(ctx context.Context, sessionID string, content string, attachments ...message.Attachment) (<-chan AgentEvent, error) {
	events := make(chan AgentEvent)
	
	cancelCtx, cancel := context.WithCancel(ctx)
	s.mu.Lock()
	if s.activeRequests == nil {
		s.activeRequests = make(map[string]context.CancelFunc)
	}
	s.activeRequests[sessionID] = cancel
	s.mu.Unlock()
	
	go func() {
		defer close(events)
		defer func() {
			s.mu.Lock()
			delete(s.activeRequests, sessionID)
			s.mu.Unlock()
		}()
		
		// Perform sequential thinking
		result, err := s.agent.Think(cancelCtx, content, attachments...)
		if err != nil {
			events <- AgentEvent{
				Type:  AgentEventTypeError,
				Error: err,
			}
			return
		}
		
		// Create response message with reasoning
		msg := message.Message{
			Role: message.Assistant,
			Parts: []message.ContentPart{
				message.ReasoningContent{
					Thinking:  result.Reasoning,
					Signature: fmt.Sprintf("Sequential Thinking (depth=%d)", s.agent.config.ReasoningDepth),
				},
				message.TextContent{
					Text: result.FinalOutput,
				},
				message.Finish{
					Reason: message.FinishReasonEndTurn,
					Time:   time.Now().Unix(),
				},
			},
		}
		
		events <- AgentEvent{
			Type:    AgentEventTypeResponse,
			Message: msg,
			Done:    true,
		}
	}()
	
	return events, nil
}

func (s *sequentialThinkingService) Cancel(sessionID string) {
	s.mu.Lock()
	defer s.mu.Unlock()
	
	if cancel, ok := s.activeRequests[sessionID]; ok {
		cancel()
		delete(s.activeRequests, sessionID)
	}
}

func (s *sequentialThinkingService) CancelAll() {
	s.mu.Lock()
	defer s.mu.Unlock()
	
	for _, cancel := range s.activeRequests {
		cancel()
	}
	s.activeRequests = make(map[string]context.CancelFunc)
}

func (s *sequentialThinkingService) IsSessionBusy(sessionID string) bool {
	s.mu.Lock()
	defer s.mu.Unlock()
	
	_, busy := s.activeRequests[sessionID]
	return busy
}

func (s *sequentialThinkingService) IsBusy() bool {
	s.mu.Lock()
	defer s.mu.Unlock()
	
	return len(s.activeRequests) > 0
}

func (s *sequentialThinkingService) Summarize(ctx context.Context, sessionID string) error {
	// Sequential thinking agent doesn't support summarization
	return errors.New("summarization not supported by sequential thinking agent")
}

func (s *sequentialThinkingService) UpdateModel() error {
	// Model updates handled by provider
	return nil
}

func (s *sequentialThinkingService) QueuedPrompts(sessionID string) int {
	// No queue support in this implementation
	return 0
}

func (s *sequentialThinkingService) ClearQueue(sessionID string) {
	// No queue support in this implementation
}

func (s *sequentialThinkingService) Subscribe(event pubsub.Event, handler func(AgentEvent)) func() {
	return s.agent.Subscribe(event, handler)
}