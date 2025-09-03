# vLLM Model Switching Integration - Test Instructions

## ✅ Integration Complete!

The vLLM model switching functionality has been successfully integrated into Crush UI. When you select a vLLM model from the model switcher, Crush will automatically:

1. Stop the current vLLM server
2. Restart it with the new model
3. Show progress feedback in the UI
4. Notify you when the switch is complete

## Testing the Integration

### 1. Start Crush
```bash
./crush
```

### 2. Open Model Switcher
Press `⌘+M` or `Ctrl+M` to open the model selection dialog

### 3. Select a vLLM Model
You'll see "vLLM Local" provider with these models:
- OPT 125M (Fast Testing) - Quick for testing
- Llama 3.2 1B Instruct
- Qwen2.5 Coder 3B
- Qwen2.5 Coder 7B (Default Large)
- Qwen2.5 14B Instruct
- DeepSeek Coder 6.7B
- Mistral 7B Instruct v0.2

### 4. Watch the Restart Process
When you select a different vLLM model, you'll see:
- "Restarting vLLM server with new model..." message
- The server will automatically restart with the selected model
- "✓ vLLM server restarted with model X (took Xs)" when complete

## Key Features Implemented

### Server Management (`internal/vllm/server.go`)
- Automatic server lifecycle management
- Graceful shutdown of running server
- GPU-optimized configuration for RTX 5000
- Process monitoring and health checks

### UI Integration (`internal/tui/tui.go`)
- Model selection detection for vLLM provider
- Asynchronous server restart
- Real-time status updates
- Error handling and recovery

### Configuration (`~/.config/crush/config.json`)
- Pre-configured vLLM models
- OpenAI-compatible API endpoint
- Model metadata (context windows, max tokens)

## Architecture

```
User selects vLLM model → ModelSelectedMsg
                           ↓
                    Check if vLLM provider
                           ↓
                    Trigger server restart
                           ↓
                    ServerManager handles:
                    - Stop current server
                    - Start with new model
                    - Monitor startup
                           ↓
                    Send status updates
                           ↓
                    UI shows feedback
```

## Troubleshooting

If model switching fails:
1. Check if vLLM server is accessible at http://localhost:8000
2. Verify GPU memory is available (at least 4GB free)
3. Check server logs for model loading errors
4. Try with a smaller model first (OPT 125M)

## Next Steps

The integration is complete and ready for testing. The system will:
- Automatically manage vLLM server lifecycle
- Provide seamless model switching from the UI
- Show real-time feedback during transitions
- Handle errors gracefully

Test with different models to verify the integration works smoothly!