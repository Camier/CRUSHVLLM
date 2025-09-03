#!/usr/bin/env python3
"""
Simple vLLM test script for Quadro RTX 5000
Tests with a small model to verify GPU functionality
"""

import torch
from vllm import LLM, SamplingParams

def test_vllm():
    print("🔧 vLLM Environment Test")
    print("=" * 50)
    
    # Check environment
    print(f"PyTorch version: {torch.__version__}")
    print(f"vLLM version: {__import__('vllm').__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"GPU count: {torch.cuda.device_count()}")
    if torch.cuda.is_available():
        print(f"GPU name: {torch.cuda.get_device_name(0)}")
        print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    print()
    
    # Test with a small model
    print("🦙 Testing with small model...")
    try:
        # Use a small model that fits in 16GB
        llm = LLM(
            model="microsoft/DialoGPT-small",  # Very small model for testing
            gpu_memory_utilization=0.8,       # Use 80% of GPU memory
            max_model_len=512,                 # Limit context length
        )
        
        # Simple test prompt
        prompts = ["Hello, how are you?"]
        sampling_params = SamplingParams(temperature=0.8, top_p=0.95, max_tokens=50)
        
        print("Generating response...")
        outputs = llm.generate(prompts, sampling_params)
        
        print("\n✅ vLLM Test Results:")
        for output in outputs:
            prompt = output.prompt
            generated_text = output.outputs[0].text
            print(f"Prompt: {prompt}")
            print(f"Response: {generated_text}")
            
        print("\n🎉 vLLM is working correctly!")
        return True
        
    except Exception as e:
        print(f"\n❌ Error testing vLLM: {e}")
        return False

if __name__ == "__main__":
    success = test_vllm()
    exit(0 if success else 1)