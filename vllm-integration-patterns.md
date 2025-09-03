# vLLM Integration Patterns for Crush CLI

This document outlines integration patterns for vLLM with Crush CLI using the Model Context Protocol (MCP).

## Architecture Overview

```
┌─────────────────┐    MCP Protocol    ┌─────────────────┐    HTTP API    ┌─────────────────┐
│   Crush CLI     │◄------------------►│  vLLM MCP       │◄--------------►│  vLLM Server    │
│                 │                    │  Servers        │                │                 │
├─────────────────┤                    ├─────────────────┤                ├─────────────────┤
│ • Model Mgmt    │                    │ • Model Manager │                │ • Model Engine  │
│ • Conversations │                    │ • State Persist │                │ • GPU Manager   │
│ • Tool Calls    │                    │ • Perf Monitor  │                │ • API Server    │
└─────────────────┘                    └─────────────────┘                └─────────────────┘
        │                                       │                                    │
        │ Local State                          │ Persistent Storage                 │ Model Files
        ▼                                       ▼                                    ▼
┌─────────────────┐                    ┌─────────────────┐                ┌─────────────────┐
│ .crush/         │                    │ SQLite Database │                │ HuggingFace     │
│ • sessions/     │                    │ • Model States  │                │ Model Cache     │
│ • configs/      │                    │ • Metrics       │                │                 │
│ • logs/         │                    │ • Conversations │                │                 │
└─────────────────┘                    └─────────────────┘                └─────────────────┘
```

## Integration Components

### 1. Provider Configuration

The vLLM integration appears as a custom provider in Crush configuration:

```json
{
  "providers": {
    "vllm": {
      "name": "vLLM Local",
      "base_url": "http://localhost:8000/v1/",
      "type": "openai",
      "auto_discovery": true,
      "health_check": {
        "enabled": true,
        "interval": 30000,
        "timeout": 5000
      },
      "models": []  // Auto-populated via MCP
    }
  }
}
```

### 2. MCP Server Management

Three specialized MCP servers handle different aspects:

#### Model Manager MCP
```typescript
interface ModelManagerMCP {
  // Model lifecycle
  listModels(): Promise<ModelInfo[]>;
  loadModel(config: ModelConfig): Promise<LoadResult>;
  unloadModel(modelId: string): Promise<UnloadResult>;
  
  // Configuration
  getModelInfo(modelId: string): Promise<ModelInfo>;
  setModelConfig(modelId: string, config: RuntimeConfig): Promise<void>;
  
  // Resources
  getModelConfig(modelId: string): Resource;
  getModelLogs(modelId: string): Resource;
}
```

#### State Persistence MCP
```typescript
interface StatePersistenceMCP {
  // Session management
  saveState(sessionId: string, options: SaveOptions): Promise<StateSnapshot>;
  loadState(sessionId: string, stateId: string): Promise<void>;
  clearState(sessionId: string, filters: ClearFilters): Promise<void>;
  
  // Resources
  getSessions(): Resource;
  getSessionSummary(sessionId: string): Resource;
}
```

#### Performance Monitor MCP
```typescript
interface PerformanceMonitorMCP {
  // Metrics collection
  getMetrics(duration: number, types: MetricType[]): Promise<Metrics>;
  getGpuStats(gpuIds: number[]): Promise<GpuStats>;
  getMemoryStats(): Promise<MemoryStats>;
  
  // Resources
  getCurrentMetrics(): Resource;
  getMetricsHistory(duration: string): Resource;
  getActiveAlerts(): Resource;
}
```

### 3. State Synchronization

State synchronization occurs at multiple levels:

#### Model State
- **Crush → vLLM**: Model selection triggers load operations
- **vLLM → Crush**: Model status updates reflected in UI
- **Bidirectional**: Configuration changes synchronized

#### Conversation State
- **Persistent Storage**: Conversations saved with model context
- **Session Recovery**: Full state restoration across restarts
- **History Management**: Efficient conversation history pruning

#### Performance State
- **Real-time Monitoring**: Continuous metrics collection
- **Threshold Alerts**: Performance degradation notifications
- **Resource Management**: Automatic model unloading under pressure

## Implementation Patterns

### 1. Auto-Discovery Pattern

vLLM models are automatically discovered and configured:

```typescript
class VLLMAutoDiscovery {
  async discoverModels(): Promise<ModelInfo[]> {
    // Query vLLM API for available models
    const apiModels = await this.vllmClient.listModels();
    
    // Enhance with metadata from HuggingFace
    const enhancedModels = await Promise.all(
      apiModels.map(model => this.enhanceModelMetadata(model))
    );
    
    // Update Crush configuration
    await this.updateCrushConfig(enhancedModels);
    
    return enhancedModels;
  }
  
  async enhanceModelMetadata(model: ApiModel): Promise<ModelInfo> {
    const hfMetadata = await this.fetchHuggingFaceMetadata(model.id);
    
    return {
      id: model.id,
      name: hfMetadata.name || model.id,
      context_window: hfMetadata.context_window || 4096,
      default_max_tokens: Math.min(hfMetadata.context_window / 4, 2048),
      supports_attachments: hfMetadata.modality?.includes('text'),
      can_reason: hfMetadata.capabilities?.reasoning || false,
      // vLLM specific properties
      tensor_parallel_size: model.tensor_parallel_size,
      pipeline_parallel_size: model.pipeline_parallel_size,
      memory_requirements: model.memory_usage,
      quantization: model.quantization
    };
  }
}
```

### 2. Lazy Loading Pattern

Models are loaded on-demand to optimize resource usage:

```typescript
class LazyModelLoader {
  private loadingPromises = new Map<string, Promise<void>>();
  
  async ensureModelLoaded(modelId: string): Promise<void> {
    // Check if model is already loaded
    const status = await this.getModelStatus(modelId);
    if (status === 'loaded') return;
    
    // Check if loading is already in progress
    if (this.loadingPromises.has(modelId)) {
      return this.loadingPromises.get(modelId);
    }
    
    // Start loading process
    const loadingPromise = this.loadModel(modelId);
    this.loadingPromises.set(modelId, loadingPromise);
    
    try {
      await loadingPromise;
    } finally {
      this.loadingPromises.delete(modelId);
    }
  }
  
  async loadModel(modelId: string): Promise<void> {
    // Determine optimal configuration
    const config = await this.optimizeModelConfig(modelId);
    
    // Check resource availability
    await this.ensureResourcesAvailable(config);
    
    // Load model via MCP
    const result = await this.mcpClient.invoke('load-model', {
      model_id: modelId,
      ...config
    });
    
    if (!result.success) {
      throw new Error(`Failed to load model ${modelId}: ${result.error.message}`);
    }
    
    // Wait for model to be ready
    await this.waitForModelReady(modelId);
  }
}
```

### 3. Resource Management Pattern

Intelligent resource management ensures optimal performance:

```typescript
class VLLMResourceManager {
  private readonly maxGpuUtilization = 0.9;
  private readonly maxMemoryUtilization = 0.85;
  
  async ensureResourcesAvailable(requiredConfig: ModelConfig): Promise<void> {
    const currentStats = await this.mcpClient.invoke('get-gpu-stats', {});
    const required = this.calculateRequirements(requiredConfig);
    
    // Check if resources are available
    if (this.hasAvailableResources(currentStats, required)) {
      return;
    }
    
    // Find models to unload
    const candidatesForUnload = await this.findUnloadCandidates(required);
    
    // Unload models in priority order
    for (const candidate of candidatesForUnload) {
      await this.mcpClient.invoke('unload-model', {
        model_id: candidate.model_id,
        force: false
      });
      
      // Check if we now have enough resources
      const updatedStats = await this.mcpClient.invoke('get-gpu-stats', {});
      if (this.hasAvailableResources(updatedStats, required)) {
        break;
      }
    }
  }
  
  private async findUnloadCandidates(required: ResourceRequirements): Promise<UnloadCandidate[]> {
    const loadedModels = await this.mcpClient.invoke('list-models', {});
    const candidates = [];
    
    for (const model of loadedModels.models) {
      if (model.status !== 'loaded') continue;
      
      // Calculate unload priority based on:
      // - Last usage time
      // - Memory usage
      // - Current session relevance
      const priority = await this.calculateUnloadPriority(model);
      
      candidates.push({
        model_id: model.id,
        priority,
        memory_savings: model.memory_usage_mb
      });
    }
    
    // Sort by priority (higher = more likely to unload)
    return candidates.sort((a, b) => b.priority - a.priority);
  }
}
```

### 4. Performance Monitoring Pattern

Continuous performance monitoring with intelligent alerting:

```typescript
class VLLMPerformanceMonitor {
  private metricsBuffer: MetricsPoint[] = [];
  private alertThresholds = {
    gpu_utilization: 0.95,
    memory_utilization: 0.90,
    latency_p99: 5000, // 5 seconds
    error_rate: 0.05   // 5%
  };
  
  async startMonitoring(): Promise<void> {
    setInterval(async () => {
      try {
        await this.collectMetrics();
        await this.analyzePerformance();
        await this.checkAlerts();
      } catch (error) {
        console.error('Performance monitoring error:', error);
      }
    }, 5000); // Every 5 seconds
  }
  
  private async collectMetrics(): Promise<void> {
    const [metrics, gpuStats, memoryStats] = await Promise.all([
      this.mcpClient.invoke('get-metrics', { duration_seconds: 5 }),
      this.mcpClient.invoke('get-gpu-stats', {}),
      this.mcpClient.invoke('get-memory-stats', {})
    ]);
    
    const metricsPoint: MetricsPoint = {
      timestamp: new Date(),
      throughput: metrics.throughput.tokens_per_second,
      latency_p99: metrics.latency.p99_ms,
      gpu_utilization: gpuStats.gpus[0]?.compute.utilization || 0,
      memory_utilization: memoryStats.gpu_memory.allocated_mb / memoryStats.gpu_memory.total_mb,
      error_rate: metrics.requests.error_rate
    };
    
    this.metricsBuffer.push(metricsPoint);
    
    // Keep only recent metrics
    const cutoff = new Date(Date.now() - 10 * 60 * 1000); // 10 minutes
    this.metricsBuffer = this.metricsBuffer.filter(p => p.timestamp > cutoff);
  }
  
  private async checkAlerts(): Promise<void> {
    const latest = this.metricsBuffer[this.metricsBuffer.length - 1];
    if (!latest) return;
    
    const alerts: Alert[] = [];
    
    // Check thresholds
    if (latest.gpu_utilization > this.alertThresholds.gpu_utilization) {
      alerts.push({
        type: 'gpu_high_utilization',
        severity: 'warning',
        message: `GPU utilization at ${(latest.gpu_utilization * 100).toFixed(1)}%`,
        value: latest.gpu_utilization,
        threshold: this.alertThresholds.gpu_utilization
      });
    }
    
    if (latest.latency_p99 > this.alertThresholds.latency_p99) {
      alerts.push({
        type: 'high_latency',
        severity: 'warning', 
        message: `P99 latency at ${latest.latency_p99}ms`,
        value: latest.latency_p99,
        threshold: this.alertThresholds.latency_p99
      });
    }
    
    // Process alerts
    for (const alert of alerts) {
      await this.handleAlert(alert);
    }
  }
}
```

### 5. State Persistence Pattern

Comprehensive state management for session continuity:

```typescript
class VLLMStatePersistence {
  private db: Database;
  
  async saveCurrentState(sessionId: string, options: SaveOptions = {}): Promise<StateSnapshot> {
    const timestamp = new Date();
    
    // Collect current state
    const state: SessionState = {
      timestamp,
      sessionId,
      models: options.include_models ? await this.collectModelState() : [],
      conversations: options.include_conversations ? await this.collectConversationState() : [],
      configurations: await this.collectConfigurationState(),
      performance: options.include_metrics ? await this.collectPerformanceState() : null
    };
    
    // Compress if requested
    const serializedState = options.compression 
      ? await this.compressState(state)
      : JSON.stringify(state);
    
    // Store in database
    const stateId = await this.storeState(sessionId, serializedState, {
      compressed: options.compression,
      size_bytes: serializedState.length,
      includes: {
        models: options.include_models,
        conversations: options.include_conversations,
        metrics: options.include_metrics
      }
    });
    
    return {
      success: true,
      session_id: sessionId,
      state_id: stateId,
      size_bytes: serializedState.length,
      timestamp
    };
  }
  
  async restoreState(sessionId: string, stateId: string, options: RestoreOptions = {}): Promise<void> {
    // Load state from database
    const stateRecord = await this.loadStateRecord(sessionId, stateId);
    if (!stateRecord) {
      throw new Error(`State ${stateId} not found for session ${sessionId}`);
    }
    
    // Deserialize state
    const state: SessionState = stateRecord.compressed
      ? await this.decompressState(stateRecord.data)
      : JSON.parse(stateRecord.data);
    
    // Restore components
    if (options.restore_models && state.models.length > 0) {
      await this.restoreModelState(state.models);
    }
    
    if (options.restore_conversations && state.conversations.length > 0) {
      await this.restoreConversationState(state.conversations);
    }
    
    await this.restoreConfigurationState(state.configurations);
  }
  
  private async collectModelState(): Promise<ModelState[]> {
    const models = await this.mcpClient.invoke('list-models', {});
    
    return Promise.all(
      models.models
        .filter(m => m.status === 'loaded')
        .map(async model => {
          const info = await this.mcpClient.invoke('get-model-info', {
            model_id: model.id
          });
          
          return {
            model_id: model.id,
            config: info.config,
            status: model.status,
            memory_usage: model.memory_usage_mb
          };
        })
    );
  }
}
```

## Tool Integration Examples

### Crush CLI Commands with vLLM MCP

```bash
# List available models
crush --tool mcp_vllm-model-manager_list-models

# Load a specific model with configuration
crush --tool mcp_vllm-model-manager_load-model \
  --model-id meta-llama/Llama-3.1-70B-Instruct \
  --tensor-parallel-size 4 \
  --gpu-memory-utilization 0.8

# Check current performance metrics
crush --tool mcp_vllm-performance-monitor_get-metrics \
  --duration-seconds 300 \
  --metrics throughput,latency,memory

# Save current session state
crush --tool mcp_vllm-state-persistence_save-state \
  --session-id $(crush --session-id) \
  --include-models true \
  --include-conversations true \
  --compression true

# Switch to a different model mid-conversation
crush "Switch to Llama 3.1 70B for this analysis" \
  --tool mcp_vllm-model-manager_load-model \
  --model-id meta-llama/Llama-3.1-70B-Instruct
```

### Programmatic Integration

```typescript
// Using vLLM MCP in Crush extensions
class CrushVLLMExtension {
  async switchModel(modelId: string, config?: ModelConfig): Promise<void> {
    // Ensure target model is loaded
    await this.crushAPI.callTool('mcp_vllm-model-manager_load-model', {
      model_id: modelId,
      ...config
    });
    
    // Update Crush to use the new model
    await this.crushAPI.setActiveModel('vllm', modelId);
    
    // Optionally save state
    await this.crushAPI.callTool('mcp_vllm-state-persistence_save-state', {
      session_id: this.crushAPI.getSessionId(),
      include_models: true
    });
  }
  
  async optimizeForTask(taskType: 'coding' | 'writing' | 'reasoning'): Promise<void> {
    const recommendations = await this.getModelRecommendations(taskType);
    
    for (const rec of recommendations) {
      const metrics = await this.crushAPI.callTool('mcp_vllm-performance-monitor_get-metrics', {
        duration_seconds: 60
      });
      
      if (this.meetsPerformanceRequirements(metrics, rec.requirements)) {
        await this.switchModel(rec.model_id, rec.config);
        break;
      }
    }
  }
}
```

## Best Practices

### 1. Resource Management
- Always check resource availability before loading models
- Implement graceful model unloading under resource pressure
- Use intelligent caching for model metadata

### 2. Error Handling
- Implement retry logic for transient failures
- Provide clear error messages with actionable suggestions
- Log all errors for debugging and monitoring

### 3. Performance Optimization
- Use connection pooling for vLLM API calls
- Batch operations where possible
- Implement intelligent prefetching for likely model switches

### 4. Security
- Secure API endpoints with authentication
- Validate all tool parameters
- Implement rate limiting to prevent abuse

### 5. Monitoring
- Continuous performance monitoring
- Alert on resource constraints
- Track model usage patterns for optimization

This integration pattern provides a comprehensive foundation for vLLM integration with Crush CLI, enabling seamless model management, state persistence, and performance monitoring through the Model Context Protocol.