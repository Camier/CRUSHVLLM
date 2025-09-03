#!/usr/bin/env python3
"""
vLLM OpenAI-Compatible API Server
Optimized for Quadro RTX 5000 (16GB VRAM)
"""

import subprocess
import sys
import argparse
import os

def start_server(model="microsoft/DialoGPT-medium", port=8000, gpu_memory=0.8):
    """Start vLLM OpenAI-compatible API server"""
    
    cmd = [
        "python", "-m", "vllm.entrypoints.openai.api_server",
        "--model", model,
        "--port", str(port),
        "--gpu-memory-utilization", str(gpu_memory),
        "--host", "0.0.0.0",
        "--trust-remote-code",
    ]
    
    print(f"🚀 Starting vLLM server with model: {model}")
    print(f"🌐 Server will be available at: http://localhost:{port}")
    print(f"📱 OpenAI API endpoint: http://localhost:{port}/v1")
    print(f"💾 GPU memory utilization: {gpu_memory * 100}%")
    print("-" * 60)
    
    try:
        subprocess.run(cmd, check=True)
    except KeyboardInterrupt:
        print("\n🛑 Server stopped by user")
    except Exception as e:
        print(f"\n❌ Server error: {e}")

def main():
    parser = argparse.ArgumentParser(description="vLLM Server Manager")
    parser.add_argument("--model", "-m", 
                       default="microsoft/DialoGPT-medium",
                       help="Model to serve (default: microsoft/DialoGPT-medium)")
    parser.add_argument("--port", "-p", 
                       type=int, default=8000,
                       help="Port to serve on (default: 8000)")
    parser.add_argument("--gpu-memory", "-g", 
                       type=float, default=0.8,
                       help="GPU memory utilization (default: 0.8)")
    
    args = parser.parse_args()
    start_server(args.model, args.port, args.gpu_memory)

if __name__ == "__main__":
    main()